# Neo VAE Utils

A **Forge Neo** extension for decoding with *upscaling* VAEs — decoders that emit more
than 3 channels and pixel-shuffle them into an image at higher resolution.

The reference model is [spacepxl/Wan2.1-VAE-upscale2x](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x),
a decoder-only finetune of the Wan 2.1 VAE that outputs 12 channels instead of 3 and
therefore decodes straight to 2x resolution. Because **Krea 2**, Qwen-Image and Wan 2.1
all share the same 16-channel latent space, one decoder covers all of them.

Compared to decoding normally and then running an upscale model, the 2x decoder is
essentially free — the compute cost of decoding is virtually unchanged — and it removes
the speckle/polka-dot grain the stock Wan decoder is known for.

## Install

Extensions → Install from URL:

```text
https://github.com/aoleg/Neo-VAE-Utils
```

Or clone into your Forge Neo `extensions` directory and restart:

```bash
git clone https://github.com/aoleg/Neo-VAE-Utils
```

No extra dependencies.

## Model

Download [`Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors`](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x/blob/main/Wan2.1_VAE_upscale2x_imageonly_real_v1.safetensors)
and put it in `models/VAE`.

> [!IMPORTANT]
> Select it in the **VAE Utils** accordion, **not** in the *VAE / Text Encoder* selector
> at the top of the page. That selector loads a VAE into the checkpoint at load time and
> builds the decoder with 3 output channels, so an upscaling VAE fails there with a
> tensor size mismatch and the checkpoint will not load at all.

## Usage

Open the **VAE Utils** accordion, enable it, and pick the VAE.

| Control | Default | Description |
| --- | --- | --- |
| Decoder VAE | `None` | Which VAE from `models/VAE` to decode with. 🔄 rescans the folder. |
| Output | `Downscale to normal size` | See below. |
| Downscale filter | `area` | `area` / `bicubic` / `bilinear`, only used when downscaling. |

### Output modes

**Downscale to normal size** — decode at 2x, then resample back down. The image comes out
at exactly the resolution you asked for, but noticeably cleaner than the stock decoder can
produce: the downsample averages away the decoder's residual noise. This is the safe
default and works everywhere, including inpainting.

**Upscale** — keep the decoder's native 2x output. A 1024x1024 generation is saved as
2048x2048. Nothing else in the pipeline changes; the extra resolution is real detail
reconstructed by the decoder, not an interpolation.

`area` is a plain box filter and is the closest match to "slight blur, then downsample";
`bicubic` and `bilinear` are antialiased and keep a little more apparent sharpness.

### Notes

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

## How it works

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

Nothing in `sd-webui-forge-classic` is modified.

## Credits

- [spacepxl](https://huggingface.co/spacepxl) — the Wan2.1-VAE-upscale2x model and the
  original [ComfyUI-VAE-Utils](https://github.com/spacepxl/ComfyUI-VAE-Utils) nodes.
- [Haoming02](https://github.com/Haoming02/sd-webui-forge-classic) — Forge Neo.
