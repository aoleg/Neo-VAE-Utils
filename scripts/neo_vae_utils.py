import gradio as gr
import torch

from lib_vae_utils import upscale_vae

from modules import scripts, sd_vae
from modules.processing import StableDiffusionProcessing, logger
from modules.ui import refresh_symbol
from modules.ui_components import InputAccordion, ToolButton

NONE = "None"

MODE_UPSCALE = "Upscale"
MODE_NORMAL = "Downscale to normal size"


def restore(sd_model):
    """Undo our swap wherever it may have ended up.

    `forge_objects.vae` is normally transient — Forge re-copies it from
    `forge_objects_after_applying_lora` before every pass — but applying a LoRA
    rebuilds that copy from the *live* objects and only resets `unet` and `clip`
    (`sd_forge_lora/networks.py`), so a swapped VAE can get baked in for good.
    Cheap to just check for ourselves and undo it.
    """
    original = getattr(sd_model, "forge_objects_original", None)
    if original is None:
        return

    for objects in (sd_model.forge_objects, sd_model.forge_objects_after_applying_lora):
        if objects is not None and isinstance(objects.vae, upscale_vae.UpscaleDecodeVAE):
            objects.vae = objects.vae.source or original.vae


class NeoVAEUtils(scripts.Script):
    sorting_priority = 15

    def title(self):
        return "VAE Utils"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    @staticmethod
    def vae_choices() -> list[str]:
        return [NONE, *sorted(sd_vae.vae_dict.keys())]

    def ui(self, *args, **kwargs):
        with InputAccordion(False, label=self.title()) as enabled:
            with gr.Row():
                vae_name = gr.Dropdown(value=NONE, choices=self.vae_choices(), label="Decoder VAE", info="an upscaling Wan VAE, eg. Wan2.1_VAE_upscale2x")
                refresh = ToolButton(value=refresh_symbol, tooltip="Refresh the list of VAE files")

            with gr.Row():
                mode = gr.Radio(value=MODE_NORMAL, choices=(MODE_NORMAL, MODE_UPSCALE), label="Output", info="whether to keep the decoder's extra resolution")
                downscale_filter = gr.Dropdown(
                    value=upscale_vae.DOWNSCALE_FILTERS[0],
                    choices=upscale_vae.DOWNSCALE_FILTERS,
                    label="Downscale filter",
                    info="gaussian/bilinear are cleanest; lanczos is sharpest but rings",
                )

            with gr.Row():
                gaussian_sigma = gr.Slider(
                    value=upscale_vae.GAUSSIAN_SIGMA,
                    minimum=0.3,
                    maximum=0.8,
                    step=0.05,
                    label="Gaussian blur",
                    info="higher = smoother, less grain (gaussian filter only)",
                )
                linear_light = gr.Checkbox(value=False, label="Downscale in linear light", info="average photons rather than sRGB values")

            gr.Markdown(
                "Replaces the decoder for models that use the Wan 2.1 latent space "
                "(**Krea 2**, Qwen-Image, Wan). The checkpoint's own VAE still does all encoding. "
                "Do **not** put the upscaling VAE in *VAE / Text Encoder* — it will fail to load there."
            )

        def do_refresh():
            sd_vae.refresh_vae_list()
            return gr.update(choices=self.vae_choices())

        refresh.click(fn=do_refresh, outputs=[vae_name], show_progress=False)

        self.infotext_fields = [
            (vae_name, "vae_utils_vae"),
            (mode, "vae_utils_mode"),
            (downscale_filter, "vae_utils_filter"),
            (gaussian_sigma, "vae_utils_sigma"),
            (linear_light, "vae_utils_linear"),
        ]

        return enabled, vae_name, mode, downscale_filter, gaussian_sigma, linear_light

    @torch.inference_mode()
    def process_before_every_sampling(
        self,
        p: StableDiffusionProcessing,
        enabled: bool,
        vae_name: str,
        mode: str,
        downscale_filter: str,
        gaussian_sigma: float,
        linear_light: bool,
        **kwargs,
    ):
        """Swap in the upscaling decoder.

        Forge resets `forge_objects` from `forge_objects_after_applying_lora` right
        before calling this hook, for every pass, so the swap is scoped to the pass
        that follows and never needs to be undone.
        """
        sd_model = p.sd_model
        restore(sd_model)  # also cleans up after a previous run that was interrupted

        if not enabled or vae_name in (None, NONE):
            return

        if not getattr(sd_model, "is_wan", False):
            logger.error("VAE Utils: the loaded checkpoint does not use the Wan latent space; skipping")
            return

        path = sd_vae.vae_dict.get(vae_name)
        if path is None:
            sd_vae.refresh_vae_list()
            path = sd_vae.vae_dict.get(vae_name)
        if path is None:
            logger.error(f'VAE Utils: could not find a VAE named "{vae_name}"; skipping')
            return

        try:
            vae = upscale_vae.get(path)
        except Exception as e:
            logger.error(f"VAE Utils: failed to load {vae_name}\n{e}")
            return

        source = sd_model.forge_objects.vae

        if source.latent_channels != vae.latent_channels:
            logger.error(f"VAE Utils: {vae_name} has {vae.latent_channels} latent channels, but the checkpoint uses {source.latent_channels}; skipping")
            return

        downscale = 1 if mode == MODE_UPSCALE else vae.upscale_factor

        #   inpainting composites the result back into the original image at its
        #   original size, so the extra resolution cannot survive that path
        if downscale == 1 and getattr(p, "overlay_images", None):
            logger.warning("VAE Utils: inpainting requires a normally sized image; downscaling instead")
            downscale = vae.upscale_factor

        vae.configure(source, downscale, downscale_filter, sd_model.model_config.latent_format, gaussian_sigma, linear_light)
        sd_model.forge_objects.vae = vae

        downscaling = downscale > 1
        p.extra_generation_params.update(
            {
                "vae_utils_vae": vae_name,
                "vae_utils_mode": MODE_UPSCALE if downscale == 1 else MODE_NORMAL,
                "vae_utils_filter": vae.downscale_filter if downscaling else None,
                "vae_utils_sigma": f"{vae.gaussian_sigma:g}" if downscaling and vae.downscale_filter == "gaussian" else None,
                "vae_utils_linear": True if downscaling and vae.linear_light else None,
                "vae_utils_scale": f"{vae.output_scale:g}x" if vae.output_scale != 1 else None,
            }
        )

    def postprocess(self, p: StableDiffusionProcessing, processed, *args):
        restore(p.sd_model)
