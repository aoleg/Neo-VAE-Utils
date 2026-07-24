# Neo VAE Utils

Decode with *upscaling* VAEs — decoders that emit more than 3 channels and
pixel-shuffle them into an image at higher resolution — in **Forge Neo** and
**SwarmUI**.

The reference model is [spacepxl/Wan2.1-VAE-upscale2x](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x),
a decoder-only finetune of the Wan 2.1 VAE that outputs 12 channels instead of 3 and
therefore decodes straight to 2x resolution. Because **Krea 2**, Qwen-Image and Wan 2.1
all share the same 16-channel latent space, one decoder covers all of them.

Compared to decoding normally and then running an upscale model, the 2x decoder is
essentially free — the compute cost of decoding is virtually unchanged — and it removes
the speckle/polka-dot grain the stock Wan decoder is known for.

This one repo contains two independent, self-contained extensions — a Forge Neo one at
the repo root, and a SwarmUI one alongside it — install whichever webui(s) you use.

## Model

Both extensions need the same file. Download
[`Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors`](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x/blob/main/Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors)
and put it in your webui's `models/VAE` folder.

---

## Forge Neo

### Install

Extensions → Install from URL:

```text
https://github.com/aoleg/Neo-VAE-Utils
```

Or clone into your Forge Neo `extensions` directory and restart:

```bash
git clone https://github.com/aoleg/Neo-VAE-Utils
```

No extra dependencies.

> [!IMPORTANT]
> Select the VAE in the **VAE Utils** accordion, **not** in the *VAE / Text Encoder*
> selector at the top of the page. That selector loads a VAE into the checkpoint at load
> time and builds the decoder with 3 output channels, so an upscaling VAE fails there
> with a tensor size mismatch and the checkpoint will not load at all.

### Usage

Open the **VAE Utils** accordion, enable it, and pick the VAE.

| Control | Default | Description |
| --- | --- | --- |
| Decoder VAE | `None` | Which VAE from `models/VAE` to decode with. 🔄 rescans the folder. |
| Output | `Downscale to normal size` | See below. |
| Downscale filter | `gaussian` | Resample filter, only used when downscaling. See the table below. |
| Gaussian blur | `0.5` | Kernel width for the `gaussian` filter. Higher = smoother, less grain. |
| Downscale in linear light | off | Resample in linear light instead of sRGB. |

#### Output modes

**Downscale to normal size** — decode at 2x, then resample back down. The image comes out
at exactly the resolution you asked for, but noticeably cleaner than the stock decoder can
produce: the downsample averages away the decoder's residual noise. This is the safe
default and works everywhere, including inpainting.

**Upscale** — keep the decoder's native 2x output. A 1024x1024 generation is saved as
2048x2048. Nothing else in the pipeline changes; the extra resolution is real detail
reconstructed by the decoder, not an interpolation.

#### Choosing a downscale filter

The two things that make a downscaled image look wrong are *aliasing* (high-frequency
decoder grain folding back down as mottled noise) and *ringing* (bright/dark halos at
edges, which read as oversharpening). They come from different filter properties, so
they need to be traded off deliberately:

| filter | taps | detail kept | grain surviving | ringing |
| --- | ---: | ---: | ---: | ---: |
| `gaussian` σ=0.5 | 6 | 0.458 | **9.1%** | 0.0% |
| `bilinear` | 4 | 0.542 | 11.6% | 0.0% |
| `hamming` | 4 | 0.593 | 15.6% | 0.0% |
| `bicubic` | 8 | 0.814 | 18.5% | 4.7% |
| `lanczos` | 12 | **1.000** | 19.6% | **8.2%** |
| `area` | 2 | 0.815 | **36.2%** | 0.0% |

Measured on synthetic test signals for a 2x downscale — "grain surviving" is the share of
pure above-Nyquist energy that survives (lower is cleaner), "ringing" is edge overshoot as
a percentage of edge height, "detail kept" is retention at 0.20 cycles/px.

- **`gaussian` (default)** is the best all-rounder for this decoder: strong grain
  rejection with mathematically zero ringing, and the **Gaussian blur** slider lets you
  dial the exact softness you want (0.3 sharper → 0.8 much smoother). This is also
  literally what the model author recommends — *"a slight blur and downsample"*.
- **`bilinear`** is nearly as clean with slightly more bite, and needs no tuning.
- **`lanczos`** retains the most detail but has by far the worst ringing. It is the
  "keep every last detail" option, **not** the clean one — if the image looks
  oversharpened, this will make it worse.
- **`area`** is only a 2-tap box average and barely low-passes at all, leaving 3-4x more
  grain than anything else here. It was the old default; it is kept for compatibility but
  is not recommended.

On a real encode→decode→downscale round-trip against ground truth, the ordering holds:
`gaussian` σ=0.5 scored best (36.23 dB) and `lanczos` worst (35.52 dB). Note that test
uses *encoded* latents, which is the decoder's best case — spacepxl's notes on latent
degradation mean **generated** latents carry more artifacts, so the gap in real use is
wider than those numbers suggest.

#### Downscale in linear light

Resamples with the sRGB transfer curve undone, so pixels average as light rather than as
code values. Mainly changes how bright speckles blend into their surroundings.

Measured, it is a very slight *negative* on reconstruction accuracy (36.23 → 35.97 dB with
`gaussian` σ=0.5), so it is **off by default** and offered as a look preference rather
than a quality setting.

#### Notes

- **Encoding is untouched.** Only the decoder is replaced; `encode()` is delegated back to
  the checkpoint's own VAE, so img2img and any re-encode behave exactly as they would
  without this extension.
- **Inpainting forces the downscale mode.** The result has to be composited back into the
  original image at its original size, so the extra resolution cannot survive that path.
  The extension detects this and downscales, with a warning in the console.
- **The generation's `Size:` in the infotext is the latent size**, not the file size, when
  running in Upscale mode. `vae_utils_scale` records the actual factor.
- Tiled decoding (Forge's OOM fallback, and `--vae-always-tiled`) works — the shuffle is
  applied once, after the tiles are reassembled.
- Any Wan-family VAE can be selected, including ordinary 3-channel ones; a 1x VAE simply
  decodes normally.

### How it works

Forge Neo's `backend.nn.wan_vae.WanVAE` already exposes `conv_out_channels` separately
from `image_channels`, so the model class needs no changes — it is just never constructed
with anything but 3. This extension:

1. reads the channel counts back out of the checkpoint (`decoder.head.2.weight` is
   `[12, 96, 3, 3, 3]` for the 2x model) and builds `WanVAE` to match;
2. wraps it in a `backend.patcher.vae.VAE` subclass that overrides `process_output()` —
   the one point every decode path funnels through — to pixel-shuffle 12 channels into
   RGB at 2x, and optionally resample back down;
3. swaps that object into `p.sd_model.forge_objects.vae` in `process_before_every_sampling`.

Forge re-copies `forge_objects` from `forge_objects_after_applying_lora` before every
sampling pass, so the swap is scoped to the pass that follows and never has to be undone.
The raw decoder is still 8x spatial, so tiling ratios stay correct with no other changes.

`area`, `bilinear` and `bicubic` use PyTorch's own antialiased `F.interpolate`. PyTorch has
no Lanczos, Hamming or Gaussian resampler, so those are built as explicit kernels and
applied as a separable strided convolution — in float, on GPU, with no 8-bit round trip
(which is why Pillow isn't used). Because the downscale factor is always a whole number,
a single fixed kernel per axis is exactly correct; the only subtlety is that an output
pixel's centre lands on a half-integer input coordinate for even factors and a whole one
for odd, so the tap count has to carry the same parity or the image shifts half a pixel.

Nothing in `sd-webui-forge-classic` is modified.

### Files

`scripts/neo_vae_utils.py` (the Forge Script) and `lib_vae_utils/` (the decode-side VAE
wrapper). Everything else in the repo belongs to the SwarmUI side below.

---

## SwarmUI

Unlike Forge Neo, SwarmUI's backend *is* an actual ComfyUI instance, so this side of the
repo doesn't reimplement any VAE math — it installs spacepxl's original
[ComfyUI-VAE-Utils](https://github.com/spacepxl/ComfyUI-VAE-Utils) node pack unmodified
and adds a small C# extension that wires it into the normal generation graph.

### Install

Clone into your SwarmUI `src/Extensions` directory and restart (or run the `update`
script):

```bash
cd src/Extensions
git clone https://github.com/aoleg/Neo-VAE-Utils VAE-Utils
```

Open **Advanced Options → VAE Utils**, enable it, and click **Install VAE Utils** — this
one-click-installs spacepxl's `ComfyUI-VAE-Utils` node pack (MIT License) onto your comfy
backend. Restart the backend afterward.

### Usage

| Parameter | Default | Description |
| --- | --- | --- |
| Decoder VAE | `None` | Which VAE file to decode with. The upscale factor is auto-detected from the checkpoint. |
| Output | `Downscale to normal size` | Same two modes as the Forge Neo side, see above. |
| Downscale Filter | `area` | `area` / `bicubic` / `bilinear`, only used when downscaling. |

Same recommendation as Forge Neo: `area` for the cleanest/most denoised result, `bicubic`
or `bilinear` for a touch more apparent sharpness.

#### Notes

- **Encoding is untouched.** Only the final decode node is replaced; the normal `VAE`
  parameter (or the checkpoint's own baked-in VAE) still handles img2img encoding.
- Do **not** put the upscaling VAE in the stock **VAE** parameter — the same tensor-size
  mismatch as Forge Neo applies, since it also goes through stock ComfyUI `VAELoader`.
- **A masked-crop composite (eg "Inpaint Only Masked") forces the downscale mode** — the
  decoded region gets pasted back into the original-size canvas afterward, so the extra
  resolution can't survive that path.
- Set alongside a **Pixel Decoder Model** (PiD), VAE Utils is skipped with a console
  warning — the two are mutually exclusive decode paths.
- VAE tiling settings (`VAE Tile Size` etc.) are respected if set.
- Any Wan-family VAE can be selected, including ordinary 3-channel ones; a 1x VAE simply
  decodes normally.

### How it works

`VAEUtilsExtension.cs` registers a `VAE Utils` parameter group and, at workflow-generation
time, takes over node ID `8` — the reserved "Final VAEDecode" slot — building
`VAEUtils_CustomVAELoader` + `VAEUtils_VAEDecodeTiled` nodes instead of the stock
`VAELoader`/`VAEDecode`. The upscale factor is read directly from the checkpoint's
`decoder.head.2.weight` shape (via `T2IModel.GetMetadataHeaderFrom`), the same detection
Forge Neo's side does. For downscale mode, a stock `ImageScaleBy` node resamples the
result back down — no custom node needed, since ComfyUI's built-in scaler already
supports `area`/`bicubic`/`bilinear`.

Compiles and loads cleanly against SwarmUI (verified with `dotnet build` against the real
`SwarmUI.dll`, and a live server startup showing the extension prepped alongside the
built-ins with no errors). End-to-end generation through a live comfy backend has not been
tested — please report back if you hit issues.

### Files

`VAEUtilsExtension.cs`, `VAEUtilsExtension.csproj`, and `Assets/vae_utils_install.js`.
Everything else in the repo belongs to the Forge Neo side above.

## Credits

- [spacepxl](https://huggingface.co/spacepxl) — the Wan2.1-VAE-upscale2x model and the
  original [ComfyUI-VAE-Utils](https://github.com/spacepxl/ComfyUI-VAE-Utils) nodes.
- [Haoming02](https://github.com/Haoming02/sd-webui-forge-classic) — Forge Neo.
- [mcmonkey](https://github.com/mcmonkeyprojects/SwarmUI) — SwarmUI.
