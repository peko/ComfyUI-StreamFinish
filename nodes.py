"""ComfyUI node: finish a video latent without ever holding the frame batch."""
from __future__ import annotations

import os
from importlib import import_module

import folder_paths

from .stream import run

QUALITY = ["BICUBIC", "LOW", "MEDIUM", "HIGH", "ULTRA",
           "HIGHBITRATE_LOW", "HIGHBITRATE_MEDIUM", "HIGHBITRATE_HIGH",
           "HIGHBITRATE_ULTRA", "STREAMING_MEDIUM", "STREAMING_ULTRA",
           "DENOISE_LOW", "DENOISE_MEDIUM", "DENOISE_HIGH", "DENOISE_ULTRA",
           "DEBLUR_LOW", "DEBLUR_MEDIUM", "DEBLUR_HIGH", "DEBLUR_ULTRA"]


def _video_from_path(path):
    """Native file-backed VIDEO, without decoding or copying a single frame.

    The import location moved between ComfyUI builds; try both and degrade to
    returning None rather than failing a render that has already succeeded.
    """
    for mod in ("comfy_api.latest._input_impl.video_types",
                "comfy_api.latest.input_impl.video_types",
                "comfy_api.input_impl.video_types"):
        try:
            return getattr(import_module(mod), "VideoFromFile")(str(path))
        except (ImportError, AttributeError):
            continue
    return None


class StreamFinishVideo:
    """LATENT + VAE -> upscaled, interpolated mp4, in one streaming pass.

    Takes the LATENT rather than an IMAGE batch on purpose. By the time a graph
    has an IMAGE, every frame already exists in system RAM at full resolution,
    and that is the thing this node is for avoiding: the same chain built from
    RTXVideoSuperResolution + RTXVideoFrameGeneration needs ~53 GB for an
    801-frame clip at 4x, because ComfyUI materialises the batch on every edge.
    Here the decode happens in chunks and each frame is written to ffmpeg as
    soon as it exists.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Video latent, straight from the sampler."}),
                "vae": ("VAE",),
                "filename_prefix": ("STRING", {"default": "StreamFinish/clip"}),
                "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 240.0, "step": 0.5,
                                  "tooltip": "Source frame rate. The output rate is this "
                                             "times the interpolation multiplier — the node "
                                             "does that for you."}),
                "scale": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.05,
                                    "tooltip": "1.0 keeps the size; use it with a DENOISE "
                                               "or DEBLUR quality to clean without resizing."}),
                "quality": (QUALITY, {"default": "HIGH"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "interpolate": ("INT", {"default": 2, "min": 1, "max": 16,
                                        "tooltip": "Frame multiplier. 1 disables interpolation. "
                                                   "Output frames are (N-1)*M + 1, not N*M."}),
                "interpolate_mode": (["LOW", "MEDIUM", "HIGH"], {"default": "MEDIUM"}),
                "shot_change_detection": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Do not interpolate across a cut. Matters for clips joined "
                               "end to end, as SVI and multipart output are."}),
                "latent_overlap": ("INT", {"default": 1, "min": 1, "max": 8,
                                           "tooltip": "Latent frames of overlap between "
                                                      "chunks. 1 is the minimum for a "
                                                      "correct frame count; raise it if the "
                                                      "chunk seam is visible, same idea as "
                                                      "VAEDecodeTiled's temporal_overlap."}),
                "frames_per_chunk": ("INT", {"default": 16, "min": 1, "max": 256,
                                             "tooltip": "Latent frames decoded at a time. "
                                                        "Lower it if the VAE decode is what "
                                                        "runs out of memory."}),
                "codec": (["h264_nvenc", "hevc_nvenc", "libx264"], {"default": "h264_nvenc"}),
                "cq": ("INT", {"default": 19, "min": 0, "max": 51}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "VIDEO")
    RETURN_NAMES = ("video_path", "report", "video")
    FUNCTION = "finish"
    CATEGORY = "video"
    OUTPUT_NODE = True
    DESCRIPTION = __doc__

    def finish(self, samples, vae, filename_prefix, fps, scale, quality, strength,
               interpolate, interpolate_mode, shot_change_detection,
               latent_overlap, frames_per_chunk, codec, cq):
        full, _, _, _, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory())
        base = os.path.basename(filename_prefix) or "clip"
        n = 1
        while os.path.exists(os.path.join(full, f"{base}_{n:05d}.mp4")):
            n += 1
        out_path = os.path.join(full, f"{base}_{n:05d}.mp4")

        path, report, _ = run(
            lambda chunk: vae.decode(chunk),
            samples["samples"],
            out_path=out_path, fps=float(fps), scale=float(scale),
            quality=quality, strength=float(strength), multiplier=int(interpolate),
            mode=interpolate_mode, shot_detect=bool(shot_change_detection),
            frames_per_chunk=int(frames_per_chunk),
            latent_overlap=int(latent_overlap), codec=codec, cq=int(cq))

        video = _video_from_path(path)
        ui = {"text": [report]}
        return {"ui": ui, "result": (path, report, video)}


NODE_CLASS_MAPPINGS = {"StreamFinishVideo": StreamFinishVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"StreamFinishVideo": "Stream Finish Video (upscale + interpolate)"}
