"""The streaming pass itself: latent chunks in, finished video file out.

Kept free of ComfyUI imports so it can be exercised from a plain script.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Iterator

import torch


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:                                     # the wheel most ComfyUI installs pull in
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:                 # noqa: BLE001
        raise RuntimeError("ffmpeg not found on PATH and imageio-ffmpeg is not "
                           "installed") from exc


def latent_chunks(samples: torch.Tensor, per_chunk: int,
                  overlap: int = 1) -> Iterator[tuple[torch.Tensor, bool]]:
    """Slice a video latent along time, with overlap. Yields (slice, is_first).

    *** Slicing a video latent does NOT give clean frame boundaries. ***
    A temporal VAE decodes T latent frames into `(T-1) * r + 1` pixel frames,
    not `T * r`: the first latent frame of whatever you hand it becomes a single
    frame, and only the rest expand. So decoding [0..15] and [16..20]
    independently yields 61 + 17 = 78 frames where the whole latent yields 81.
    Three frames vanish, silently, and the loss grows with the chunk count.

    The fix is to overlap by `overlap` latent frames and drop the pixel frames
    the previous chunk already produced — see `_drop_count`. One frame of
    overlap is the minimum for correctness; more also restores the VAE's
    temporal context across the seam, which is the same reason
    `VAEDecodeTiled` has `temporal_overlap` (CLAUDE.md TRAP 5).
    """
    five = samples.dim() == 5
    n = samples.shape[2] if five else samples.shape[0]
    start = 0
    first = True
    while start < n:
        lo = start if first else max(0, start - overlap)
        hi = min(n, start + per_chunk)
        yield (samples[:, :, lo:hi] if five else samples[lo:hi]), first
        if hi >= n:
            break
        start = hi
        first = False


def _drop_count(ratio: int, overlap: int) -> int:
    """Pixel frames at the head of an overlapped chunk already emitted before.

    Decoding latents [a..b] gives frames f_0..f_{(b-a)*ratio} with f_0 at latent
    a and f_{k*ratio} at latent a+k. The previous chunk covered through latent
    a+overlap-1, so f_0 .. f_{ratio*(overlap-1)} are repeats.
    """
    return ratio * (overlap - 1) + 1


class Encoder:
    """ffmpeg writing rgb24 frames to disk, one pipe for the whole run."""

    def __init__(self, path, width, height, fps, codec, cq):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.proc = subprocess.Popen(
            [_ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", f"{fps}", "-i", "-",
             "-c:v", codec, "-pix_fmt", "yuv420p",
             *(["-preset", "p5", "-rc", "vbr", "-cq", str(cq)]
               if codec.endswith("nvenc") else ["-crf", str(cq)]),
             path],
            stdin=subprocess.PIPE)

    def write(self, chw: torch.Tensor) -> None:
        self.proc.stdin.write(
            chw.clamp(0, 1).mul(255.0).round().to(torch.uint8)
            .permute(1, 2, 0).contiguous().cpu().numpy().tobytes())

    def close(self) -> None:
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        self.proc.wait()


def run(decode_chunk, samples, *, out_path, fps, scale=4.0, quality="HIGH",
        strength=1.0, multiplier=2, mode="MEDIUM", shot_detect=True,
        frames_per_chunk=16, latent_overlap=1, codec="h264_nvenc", cq=19,
        log=print):
    """Decode -> super-resolution -> frame generation -> encode, streaming.

    `decode_chunk(latent_slice) -> IMAGE tensor [n, H, W, 3]` is passed in rather
    than imported so this file never needs ComfyUI.

    Live tensors at any moment: one decoded chunk, one upscaled frame, the
    previous upscaled frame, one generated frame. Nothing that scales with clip
    length, which is the entire point — the ComfyUI graph form of this chain
    materialises every frame at every edge and needs 53 GB for an 801-frame clip
    at 4x, against 31 GB of machine.
    """
    from nvvfx import VideoFrameGeneration, VideoSuperRes

    device = torch.device("cuda", torch.cuda.current_device())
    stream = torch.cuda.current_stream(device)
    ptr = int(stream.cuda_stream)

    sr = fg = enc = None
    prev = None
    n_in = n_out = 0
    w = h = ow = oh = 0
    t0 = time.time()
    try:
        ratio = None
        for chunk, is_first in latent_chunks(samples, frames_per_chunk,
                                             latent_overlap):
            n_lat = chunk.shape[2] if chunk.dim() == 5 else chunk.shape[0]
            images = decode_chunk(chunk)
            # A video VAE returns [B, T, H, W, 3]; core's VAEDecode folds the
            # batch and time axes together and so must we, or indexing yields a
            # 4-D slice and permute() fails with a message about sparse_coo that
            # has nothing to do with anything.
            if images.dim() == 5:
                images = images.reshape(-1, *images.shape[-3:])
            if ratio is None and n_lat > 1:
                # Derived, not assumed: 4:1 is Wan's, other video VAEs differ.
                ratio = (images.shape[0] - 1) // (n_lat - 1)
            skip = 0 if is_first else _drop_count(ratio or 4, latent_overlap)
            for i in range(skip, images.shape[0]):
                frame = (images[i].to(device=device, dtype=torch.float32)
                         .permute(2, 0, 1).contiguous())
                if sr is None:                       # first frame fixes the sizes
                    _, h, w = frame.shape
                    ow, oh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
                    if codec.startswith("h264") and max(ow, oh) > 4096:
                        raise ValueError(
                            f"{ow}x{oh} exceeds h264's 4096 px limit — "
                            "choose hevc_nvenc")
                    sr = VideoSuperRes(
                        quality=VideoSuperRes.QualityLevel[quality],
                        strength=float(strength), device=device.index,
                        image_encoding=VideoSuperRes.ImageEncoding.RGB8)
                    sr.input_width, sr.input_height = w, h
                    sr.output_width, sr.output_height = ow, oh
                    sr.load()
                    if multiplier > 1:
                        fg = VideoFrameGeneration(
                            mode=VideoFrameGeneration.Mode[mode],
                            automatic_shot_change_detection_enabled=shot_detect,
                            device=device.index,
                            image_encoding=VideoFrameGeneration.ImageEncoding.RGB8)
                        fg.input_width, fg.input_height = ow, oh
                        fg.frame_multiplier = multiplier
                        fg.load()
                    enc = Encoder(out_path, ow, oh, fps * multiplier, codec, cq)
                    log(f"StreamFinish: {w}x{h} -> {ow}x{oh}, "
                        f"fps {fps:g} -> {fps * multiplier:g}")

                # clone(): the capsule points into SDK-owned memory that the next
                # call reuses. Without it frame N comes back holding frame N+1.
                up = torch.from_dlpack(sr.run(frame, stream_ptr=ptr).image).clone()
                stream.synchronize()

                if prev is not None and fg is not None:
                    for k in range(1, multiplier):
                        mid = torch.from_dlpack(
                            fg.run(prev, up, frame_index=k, stream_ptr=ptr).image
                        ).clone()
                        stream.synchronize()
                        enc.write(mid)
                        n_out += 1
                enc.write(up)
                n_out += 1
                prev = up
                n_in += 1
            del images
    finally:
        if enc is not None:
            enc.close()
        if sr is not None:
            sr.close()
        if fg is not None:
            fg.close()

    dt = time.time() - t0
    report = (f"{n_in} decoded -> {n_out} written in {dt:.1f}s "
              f"({n_in / max(dt, 1e-6):.1f} src-fps), {ow}x{oh} @ "
              f"{fps * multiplier:g} fps, peak VRAM "
              f"{torch.cuda.max_memory_allocated() / 2 ** 20:.0f} MiB")
    log("StreamFinish: " + report)
    return out_path, report, n_out
