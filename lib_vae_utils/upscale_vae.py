"""Loading and decode-side wrapping for pixel-shuffle upscaling Wan-family VAEs.

The reference model is spacepxl's `Wan2.1_VAE_upscale2x_imageonly_real_v1`: a
decoder-only finetune of the Wan 2.1 VAE whose final `Conv3d` emits 12 channels
instead of 3.  Rearranging those 12 channels with a 2x pixel shuffle produces an
image at twice the usual resolution, at essentially the cost of a normal decode.

Krea 2, Qwen-Image and Wan 2.1 all share the same 16-channel latent space, so the
same decoder works for any of them.

Forge Neo's `WanVAE` already exposes `conv_out_channels` separately from
`image_channels`, so no changes to the webui itself are needed: this module just
builds the model with the channel counts read back out of the checkpoint, and
inserts the pixel shuffle into the decode path.
"""

import math
import os

import torch
import torch.nn.functional as F
from transformers.modeling_utils import no_init_weights

from backend import memory_management
from backend.nn.wan_vae import WanVAE
from backend.operations import using_forge_operations
from backend.patcher.vae import VAE
from backend.utils import load_torch_file

#   Wan 2.1 / Qwen-Image / Krea 2 all share this topology; only the channel
#   counts are read back out of the checkpoint.
TOPOLOGY = {
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_scales": [],
    "temporal_downsample": [False, True, True],
    "dropout": 0.0,
}

#   Ordered worst-to-best at rejecting the decoder's high-frequency grain, which is
#   what reads as "noise" in the downscaled image.  Measured share of pure
#   above-Nyquist grain surviving a 2x downscale, and edge overshoot ("ringing",
#   which reads as an oversharpened halo):
#
#       area      36.2% grain   0.0% ring   <- only a 2-tap box, barely low-passes
#       lanczos   19.6% grain   8.2% ring   <- sharpest, but rings the most
#       bicubic   18.5% grain   4.7% ring
#       hamming   15.6% grain   0.0% ring
#       bilinear  11.6% grain   0.0% ring
#       gaussian   9.1% grain   0.0% ring   <- at the default sigma of 0.5
#
#   Ringing only comes from kernels with negative lobes (bicubic, lanczos), so the
#   "clean and smooth" options are the ones with neither problem: gaussian, then
#   bilinear.  lanczos is the "keep every last detail" choice, not the clean one.
DOWNSCALE_FILTERS = ("gaussian", "bilinear", "hamming", "area", "bicubic", "lanczos")

#   torch resamples these three natively, with correct antialiasing on the way down
_NATIVE_FILTERS = frozenset({"area", "bilinear", "bicubic"})

GAUSSIAN_SIGMA = 0.5

#   the extra output channels only widen the very last conv, but the shuffled
#   result briefly coexists with the unshuffled one
MEMORY_HEADROOM = 1.25


def _pixel_shuffle(x: torch.Tensor, factor: int) -> torch.Tensor:
    """[B, C*f*f, (T,) H, W] -> [B, C, (T,) H*f, W*f]"""
    if factor <= 1:
        return x
    if x.ndim == 5:  # pixel_shuffle wants the channels at dim -3
        return F.pixel_shuffle(x.movedim(1, 2), factor).movedim(2, 1)
    return F.pixel_shuffle(x, factor)


def _filter_taps(name: str, factor: int, sigma: float) -> torch.Tensor:
    """Build a normalised 1D resampling kernel, in input-pixel spacing.

    `support` is the filter's radius in *output* pixels, so it scales by `factor` to
    reach input-pixel space - that scaling is what makes the kernel a low-pass at the
    new sample rate rather than at the old one, ie what actually antialiases.
    """
    if name == "gaussian":
        support = max(1.0, 3.0 * sigma)
    elif name == "lanczos":
        support = 3.0
    elif name == "hamming":
        support = 1.0
    else:
        raise ValueError(f"no kernel for filter '{name}'")

    radius = support * factor

    #   Output pixel j covers input pixels [j*f, j*f + f), centred at j*f + (f-1)/2 -
    #   a half-integer for even factors, a whole one for odd.  The tap count has to
    #   carry that same parity or every output pixel lands half a pixel off, and it is
    #   also what makes the 'same'-style padding (K - factor) / 2 come out whole.
    if factor % 2 == 0:
        taps = 2 * math.ceil(radius)
    else:
        taps = 2 * math.ceil(radius - 0.5) + 1

    pos = (torch.arange(taps, dtype=torch.float64) - (taps - 1) / 2.0) / factor

    if name == "gaussian":
        weights = torch.exp(-pos.pow(2) / (2.0 * sigma * sigma))
    elif name == "hamming":
        weights = torch.where(pos.abs() < 1.0, 0.54 + 0.46 * torch.cos(math.pi * pos.clamp(-1.0, 1.0)), torch.zeros_like(pos))
    else:  # lanczos, 3 lobes
        weights = torch.where(pos.abs() < 3.0, torch.sinc(pos) * torch.sinc(pos / 3.0), torch.zeros_like(pos))

    return weights / weights.sum()


def _conv_downscale(x: torch.Tensor, factor: int, taps: torch.Tensor) -> torch.Tensor:
    """Separable strided convolution: [B, C, H, W] -> [B, C, H/f, W/f]."""
    channels = x.shape[1]
    count = taps.numel()
    pad = (count - factor) // 2

    taps = taps.to(dtype=x.dtype, device=x.device)
    #   reflect needs pad < the dimension it mirrors; replicate has no such limit
    mode = "reflect" if pad < min(x.shape[-2:]) else "replicate"
    x = F.pad(x, (pad, pad, pad, pad), mode=mode)

    vertical = taps.view(1, 1, -1, 1).expand(channels, 1, -1, 1).contiguous()
    horizontal = taps.view(1, 1, 1, -1).expand(channels, 1, 1, -1).contiguous()
    x = F.conv2d(x, vertical, stride=(factor, 1), groups=channels)
    return F.conv2d(x, horizontal, stride=(1, factor), groups=channels)


def _downscale(x: torch.Tensor, factor: int, filter_name: str, sigma: float = GAUSSIAN_SIGMA) -> torch.Tensor:
    """Resample [B, C, (T,) H, W] down by `factor`, folding frames into the batch."""
    if factor <= 1:
        return x

    is_video = x.ndim == 5
    if is_video:
        b, c, t, h, w = x.shape
        x = x.movedim(1, 2).reshape(b * t, c, h, w)

    if filter_name in _NATIVE_FILTERS:
        h, w = x.shape[-2:]
        size = (max(1, round(h / factor)), max(1, round(w / factor)))
        if filter_name == "area":
            x = F.interpolate(x, size=size, mode="area")
        else:
            x = F.interpolate(x, size=size, mode=filter_name, antialias=True, align_corners=False)
    else:
        x = _conv_downscale(x, factor, _filter_taps(filter_name, factor, sigma))

    if is_video:
        x = x.reshape(b, t, x.shape[-3], *x.shape[-2:]).movedim(2, 1)
    return x


#   sRGB transfer function. Resampling in linear light averages photons rather than
#   code values, which mainly changes how bright speckles blend into their neighbours;
#   sRGB-space averaging under-weights them. Both are defensible, so it's a toggle.
def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0.0)  # the power term is undefined below zero
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055).pow(2.4))


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(min=0.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.pow(1.0 / 2.4) - 0.055)


class UpscaleDecodeVAE(VAE):
    """Wraps an upscaling Wan VAE so that `decode()` hands back ordinary RGB.

    Only the decode side belongs to us: `encode()` is delegated straight back to
    the checkpoint's own VAE, so img2img and any re-encode keep using the exact
    encoder the model was trained against.

    The pixel shuffle is installed by overriding `process_output()`, which is the
    single point every decode path in `backend.patcher.vae.VAE` funnels through
    (plain, tiled-2d and tiled-3d alike).  Tiling therefore keeps working
    untouched: the raw decoder is still 8x spatial, so `upscale_ratio` stays
    correct and the shuffle happens once, after the tiles are reassembled.
    """

    def __init__(self, model: WanVAE, factor: int):
        super().__init__(model=model, is_wan=True)

        self.upscale_factor = int(factor)
        self.output_channels = 3 * self.upscale_factor**2  # what the decoder really emits
        self.downscale = 1
        self.downscale_filter = DOWNSCALE_FILTERS[0]
        self.gaussian_sigma = GAUSSIAN_SIGMA
        self.linear_light = False
        self.source: VAE = None

        base_estimate = self.memory_used_decode
        self.memory_used_decode = lambda shape, dtype: base_estimate(shape, dtype) * MEMORY_HEADROOM

    def configure(
        self,
        source: VAE,
        downscale: int,
        downscale_filter: str,
        latent_format,
        gaussian_sigma: float = GAUSSIAN_SIGMA,
        linear_light: bool = False,
    ) -> "UpscaleDecodeVAE":
        """Rebind to the currently loaded checkpoint and the current UI settings."""
        self.source = source
        self.downscale = int(downscale)
        self.downscale_filter = downscale_filter if downscale_filter in DOWNSCALE_FILTERS else DOWNSCALE_FILTERS[0]
        self.gaussian_sigma = float(gaussian_sigma)
        self.linear_light = bool(linear_light)
        self.first_stage_model.latent_format = latent_format
        return self

    @property
    def output_scale(self) -> float:
        """Spatial scale of the returned image relative to a normal 1x decode."""
        return self.upscale_factor / max(1, self.downscale)

    def process_output(self, image: torch.Tensor) -> torch.Tensor:
        #   Clamp before the downscale, not after.  About 3.4% of decoder outputs land
        #   outside [0, 1], but those are reconstruction artifacts rather than real
        #   highlight headroom - the target image was in range to begin with - so
        #   letting them into the average just drags neighbouring pixels off.  Measured
        #   both orders on a real round-trip: clamping first was very slightly better
        #   (36.235 vs 36.231 dB), so the theoretically-tidier "clamp last" is not worth
        #   the behaviour change.  The trailing clamp stays, to catch the overshoot that
        #   ringing kernels (bicubic, lanczos) introduce during the downscale itself.
        image = image.add(1.0).div_(2.0).clamp_(0.0, 1.0)
        image = _pixel_shuffle(image, self.upscale_factor)

        if self.downscale > 1 and self.linear_light:
            image = _linear_to_srgb(_downscale(_srgb_to_linear(image), self.downscale, self.downscale_filter, self.gaussian_sigma))
        else:
            image = _downscale(image, self.downscale, self.downscale_filter, self.gaussian_sigma)

        return image.clamp_(0.0, 1.0)

    #   the upscale VAE's encoder is a verbatim copy of the one it was finetuned
    #   from, but the loaded checkpoint may ship a different Wan-family encoder;
    #   always defer so encoding is bit-identical to not having this extension
    def encode(self, pixel_samples: torch.Tensor):
        return self.source.encode(pixel_samples)

    def encode_tiled(self, pixel_samples: torch.Tensor, *args, **kwargs):
        return self.source.encode_tiled(pixel_samples, *args, **kwargs)


def upscale_factor_of(out_channels: int) -> int:
    """12 -> 2, 27 -> 3, 3 -> 1."""
    if out_channels % 3:
        raise ValueError(f"decoder emits {out_channels} channels, which is not a multiple of 3")

    factor = round((out_channels // 3) ** 0.5)
    if 3 * factor * factor != out_channels:
        raise ValueError(f"decoder emits {out_channels} channels, which is not 3 x (integer scale)^2")

    return factor


def infer_config(state_dict: dict) -> dict:
    """Read the channel counts of a native-format Wan VAE checkpoint."""
    try:
        conv_in = state_dict["encoder.conv1.weight"]
        head = state_dict["decoder.head.2.weight"]
        z_dim = state_dict["conv2.weight"].shape[0]
    except KeyError as key:
        raise ValueError(f"not a Wan-family VAE: missing {key}") from None

    config = dict(TOPOLOGY)
    config["base_dim"] = int(conv_in.shape[0])
    config["image_channels"] = int(conv_in.shape[1])
    config["z_dim"] = int(z_dim)
    config["conv_out_channels"] = int(head.shape[0])
    return config


def _build(path: os.PathLike) -> UpscaleDecodeVAE:
    state_dict = load_torch_file(path)
    config = infer_config(state_dict)
    factor = upscale_factor_of(config["conv_out_channels"])

    with no_init_weights():
        with using_forge_operations(device=memory_management.cpu, dtype=memory_management.vae_dtype(), bnb_dtype="vae"):
            model = WanVAE.from_config(config)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    del state_dict

    if missing:
        raise ValueError(f"{len(missing)} weight(s) not present in the file, eg. {missing[:3]}")
    if unexpected:
        memory_management.logger.warning(f"VAE Utils: ignoring {len(unexpected)} unused key(s) in {os.path.basename(path)}")

    return UpscaleDecodeVAE(model, factor)


#   keyed by path; a single entry, so switching files frees the previous one
_cache: dict[str, UpscaleDecodeVAE] = {}


def get(path: os.PathLike) -> UpscaleDecodeVAE:
    """Load (or return the already loaded) decoder for `path`."""
    path = str(path)

    if path not in _cache:
        vae = _build(path)

        for stale in _cache.values():  # only ever keep one decoder resident
            memory_management.unload_model(stale.patcher)
        _cache.clear()
        memory_management.soft_empty_cache()

        _cache[path] = vae
        memory_management.logger.info(f"VAE Utils: loaded {os.path.basename(path)} ({vae.upscale_factor}x decoder)")

    return _cache[path]
