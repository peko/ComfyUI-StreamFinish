# ComfyUI-StreamFinish

Upscale and frame-interpolate a video latent **in one streaming pass**, without
ever materialising the frame batch.

One node: `Stream Finish Video (upscale + interpolate)`. It takes the `LATENT`
straight from the sampler, decodes it in chunks, runs NVIDIA RTX Video Super
Resolution and Video Frame Generation on each frame, and writes it to ffmpeg
immediately. No upscale model, no RIFE checkpoint — **no weights at all**; both
effects live in the NVIDIA Video Effects SDK.

## The problem it solves

ComfyUI's `IMAGE` is one materialised `[N, H, W, 3]` float32 tensor in system
RAM, and every edge of the graph carries the whole batch. Chain the two official
nodes — `RTXVideoSuperResolution` → `RTXVideoFrameGeneration` → `CreateVideo` —
and the intermediates coexist, because the first node's output is the second's
input:

| clip | 2x | 4x |
|---|---|---|
| 81 frames, 832x448 | 0.6 → 1.2 GB | 2.2 → 4.3 GB |
| 321 frames (SVI, 4 clips) | 5.3 → **10.7 GB** | 21.4 → 42.7 GB |
| 801 frames (SVI, 10 clips) | 13.3 → **26.7 GB** | 53.4 → **106.7 GB** |

On a 32 GB machine only the small cases survive. This node holds **four frames**
at a time regardless of length: one decoded chunk, one upscaled frame, the
previous upscaled frame, one generated frame.

It takes the `LATENT` rather than an `IMAGE` deliberately. By the time a graph
has an `IMAGE`, every frame already exists at full resolution and the memory has
already been spent — decoding has to happen inside the node for the saving to be
real.

## Measured

Wan 2.2 I2V, 272x416, scale 2x, interpolate 2x, `frames_per_chunk` 16:

| input | frames | time | peak VRAM | min MemAvailable |
|---|---|---|---|---|
| 81-frame latent, in a real graph | 81 -> **161** | 5.5 s | 10867 MiB* | 12.1 GB |
| **801-frame latent** | 801 -> **1601** | **39.1 s** | **4566 MiB** | 20.8 GB |
| 801-frame mp4, via `finish_video.py` | 801 -> 1601 | 9.4 s | **89 MiB** | 27.3 GB |

\* that figure is the *sampling* peak — it includes the Wan experts, not this node.

The 801-frame row is the point of the project. The same job built from the
official IMAGE-batch nodes needs **26.7 GB of system RAM at 2x** (and 107 GB at
4x); here it ran to completion with 20.8 GB still free, and the VRAM it did use
is the VAE decode, not the streaming. The last row is the same chain without a
VAE in it at all, which is why its VRAM is two orders of magnitude lower.

## Requirements

* NVIDIA RTX **40-series or newer** (Frame Generation); Super Resolution works
  from Turing
* PyTorch built against **cu130 or newer**
* `nvidia-vfx >= 0.2` — **install it from NVIDIA's own index**:

```bash
pip install --extra-index-url https://pypi.nvidia.com nvidia-vfx
```

> The `nvidia-vfx` on PyPI proper is a stub: every release there is a ~0 MB
> sdist containing `PKG-INFO` and `pyproject.toml` and no code. Installed from
> the default index, nothing fails until the first render, inside the node.
> Check with `python -c "from nvvfx import VideoSuperRes"`.

* `ffmpeg` with `h264_nvenc` / `hevc_nvenc`, or `imageio-ffmpeg` installed

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/peko/ComfyUI-StreamFinish
pip install -r ComfyUI-StreamFinish/requirements.txt \
    --extra-index-url https://pypi.nvidia.com
```

A new pack needs a **fresh server** — reloading the browser is not enough.

## Inputs

| | |
|---|---|
| `samples`, `vae` | the video latent and the VAE to decode it with |
| `fps` | **source** rate; the node multiplies it by `interpolate` for the output |
| `scale` | 1.0–4.0. Use `1.0` with a `DENOISE_*`/`DEBLUR_*` quality to clean without resizing |
| `quality` | 19 modes: plain upscale, `HIGHBITRATE_*`, `STREAMING_*` (compressed sources), `DENOISE_*`, `DEBLUR_*` |
| `interpolate` | frame multiplier, `1` disables |
| `shot_change_detection` | do not interpolate across a cut — matters for clips joined end to end, as SVI and multipart output are |
| `frames_per_chunk` | latent frames decoded at a time; lower it if the VAE decode is what runs out |
| `codec`, `cq` | h264 is capped at 4096 px per axis, the node refuses up front rather than letting NVENC die mid-clip |

Returns `video_path`, a one-line `report`, and a native `VIDEO` wrapping the
finished file without re-reading it.

## Things that cost time to rediscover

**Frame count is `(N-1) × multiplier + 1`, not `N × multiplier`.** The last frame
has no successor to interpolate towards: 81 frames at 2x is 161, not 162. Output
fps is multiplied to match, so real duration is unchanged.

**The DLPack capsule must be cloned.** The SDK hands back a pointer into memory
it reuses on the next call. Without the clone, frame N arrives holding frame
N+1's pixels — which looks like a one-frame timing bug, not a memory bug.

**Chunking a video latent does NOT give clean boundaries, and the loss is
silent.** A temporal VAE turns `T` latent frames into `(T-1) * r + 1` pixel
frames, not `T * r` — the first latent frame of whatever you hand it decodes to
a single frame, and only the rest expand. Decoding `[0..15]` and `[16..20]`
independently yields 61 + 17 = **78** frames where the whole latent yields 81.
Three frames vanish with no error, and it gets worse with more chunks. This cost
a measured run to find: the node reported `78 decoded` for an 81-frame latent.

The node therefore overlaps chunks by `latent_overlap` latent frames and drops
`r * (overlap - 1) + 1` already-emitted frames at the head of each later chunk.
`r` is **derived from the first chunk**, not assumed — 4:1 is Wan's ratio, other
video VAEs differ. Overlap of 1 is the minimum for a correct count; raise it if
the seam is visible, which is the same reason `VAEDecodeTiled` has
`temporal_overlap`.

## License

MIT.
