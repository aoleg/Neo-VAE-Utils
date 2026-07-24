using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using SwarmUI.Accounts;
using SwarmUI.Builtin_ComfyUIBackend;
using SwarmUI.Core;
using SwarmUI.Text2Image;
using SwarmUI.Utils;

namespace Spacepxl.VAEUtils;

/// <summary>SwarmUI extension for spacepxl's ComfyUI-VAE-Utils (https://github.com/spacepxl/ComfyUI-VAE-Utils),
/// specifically its upscaling-decoder VAE nodes (eg Wan2.1-VAE-upscale2x). Any model that shares the Wan 2.1
/// latent space - Wan, Qwen-Image, Krea 2 - can decode through it, either at the decoder's native higher
/// resolution or downscaled back to normal size for a cleaner 1x image.
/// Only the decoder is replaced: encoding (img2img) still goes through the checkpoint's own VAE, via the
/// stock 'VAE' parameter.</summary>
public class VAEUtilsExtension : Extension
{
    public const string FeatureId = "vae_utils";

    public static T2IRegisteredParam<T2IModel> DecoderVAE;

    public static T2IRegisteredParam<string> Mode, DownscaleFilter;

    public static T2IParamGroup VAEUtilsGroup;

    /// <summary>Pixel-shuffle upscale factor per VAE file path, read from the checkpoint's own decoder head shape (eg 12-channel head -&gt; 2x). Cheap to read but no reason to reread it every generation.</summary>
    static readonly ConcurrentDictionary<string, int> UpscaleFactorCache = new();

    public override void OnInit()
    {
        InstallableFeatures.RegisterInstallableFeature(new("VAE Utils", FeatureId, "https://github.com/spacepxl/ComfyUI-VAE-Utils", "spacepxl", "This will install the ComfyUI-VAE-Utils node pack by spacepxl (MIT License).\nDo you wish to install?"));
        ScriptFiles.Add("Assets/vae_utils_install.js");
        ComfyUIBackendExtension.NodeToFeatureMap["VAEUtils_VAEDecodeTiled"] = FeatureId;

        VAEUtilsGroup = new("VAE Utils", Toggles: true, Open: false, IsAdvanced: true,
            Description: "Decode with an upscaling Wan-family VAE (eg spacepxl's Wan2.1-VAE-upscale2x), for models that share the Wan 2.1 latent space: Krea 2, Qwen-Image, Wan.\nOnly the decoder is replaced - encoding (img2img) still uses whatever the 'VAE' parameter or checkpoint default resolves to.");

        DecoderVAE = T2IParamTypes.Register<T2IModel>(new("[VAE Utils] Decoder VAE", "[VAE Utils]\nThe upscaling VAE to decode with, eg 'Wan2.1_VAE_upscale2x_imageonly_real_v1'.\nMust be a Wan-family VAE whose decoder head outputs a multiple of 3 channels above 3 - the pixel-shuffle upscale factor is auto-detected from the file, no need to specify it.",
            "None", Permission: Permissions.ModelParams, Group: VAEUtilsGroup, FeatureFlag: FeatureId, OrderPriority: 1, GetValues: ListVAEs, Subtype: "VAE"
            ));
        Mode = T2IParamTypes.Register<string>(new("[VAE Utils] Output", "[VAE Utils]\n'Downscale to normal size' resamples the decoder's output back down to the resolution you asked for - noticeably cleaner than the stock decoder, since downsampling averages away residual decoder noise.\n'Upscale' keeps the decoder's native higher resolution instead (eg a 1024x1024 request comes out 2048x2048).",
            "downscale", Group: VAEUtilsGroup, FeatureFlag: FeatureId, OrderPriority: 2,
            GetValues: (_) => ["downscale///Downscale to normal size", "upscale///Upscale"]
            ));
        DownscaleFilter = T2IParamTypes.Register<string>(new("[VAE Utils] Downscale Filter", "[VAE Utils]\nResample filter used when downscaling. 'area' is plain box-filter averaging and gives the most denoising - recommended. 'bicubic' and 'bilinear' keep a bit more apparent sharpness at the cost of some of that denoising.\nOnly used when Output is 'Downscale to normal size'.",
            "area", Group: VAEUtilsGroup, FeatureFlag: FeatureId, OrderPriority: 3, DependNonDefault: Mode.Type.ID,
            GetValues: (_) => ["area", "bicubic", "bilinear"]
            ));

        WorkflowGenerator.AddStep(ApplyVAEUtils, 0.5);
        // See WorkflowGeneratorSteps for the reserved-node-ID map: "8" is "Final VAEDecode",
        // created by the core "VAEDecode" region step at priority 1. Running at 0.5 (after
        // the refiner region ends at -4, before that core step) lets us build node "8"
        // ourselves - WGNodeData.DecodeLatents() is a no-op once the data is already
        // raw-image-typed, so the core step just passes our result through untouched.
    }

    static List<string> ListVAEs(Session s)
    {
        return T2IParamTypes.CleanModelList(Program.T2IModelSets["VAE"].ListModelsFor(s).Select(m => m.Name)).Prepend("None").ToList();
    }

    /// <summary>12 -&gt; 2, 27 -&gt; 3, throws if the file isn't a Wan-family upscale decoder.</summary>
    static int GetUpscaleFactor(T2IModel vae)
    {
        return UpscaleFactorCache.GetOrAdd(vae.RawFilePath, path =>
        {
            JObject header = T2IModel.GetMetadataHeaderFrom(path);
            if (header["decoder.head.2.weight"] is not JObject headTensor || headTensor["shape"] is not JArray shape || shape.Count == 0)
            {
                throw new SwarmUserErrorException($"VAE Utils: '{vae.Name}' doesn't look like a Wan-family VAE (missing 'decoder.head.2.weight'). Is this actually an upscaling decoder checkpoint?");
            }
            int outChannels = (int)shape[0];
            int factor = outChannels > 0 ? (int)Math.Round(Math.Sqrt(outChannels / 3.0)) : 0;
            if (outChannels % 3 != 0 || factor < 1 || 3 * factor * factor != outChannels)
            {
                throw new SwarmUserErrorException($"VAE Utils: '{vae.Name}' decoder emits {outChannels} channels, which isn't 3x(whole number)^2. Can't determine an upscale factor.");
            }
            return factor;
        });
    }

    public static void ApplyVAEUtils(WorkflowGenerator g)
    {
        // Any one group-param being present indicates the group toggle is enabled (all params in the group toggle as one).
        if (!g.UserInput.TryGet(DecoderVAE, out T2IModel vaeModel) || vaeModel is null)
        {
            return;
        }
        // This shouldn't be possible outside of corrupt API calls, but check just to be safe.
        if (!g.Features.Contains(FeatureId))
        {
            throw new SwarmUserErrorException("VAE Utils parameters specified, but the ComfyUI-VAE-Utils node pack isn't installed on the backend.");
        }
        if (g.UserInput.TryGet(ComfyUIBackendExtension.PixelDecoderModel, out _))
        {
            Logs.Warning("[VAE Utils] a Pixel Decoder Model is also set; VAE Utils skipped (the two are mutually exclusive decode paths).");
            return;
        }
        if (!g.CurrentMedia.IsLatentData)
        {
            Logs.Warning("[VAE Utils] media is already decoded (eg by another extension); nothing to do, skipped.");
            return;
        }

        int factor = GetUpscaleFactor(vaeModel);
        string mode = g.UserInput.Get(Mode, "downscale");

        // A masked-crop composite (eg "Inpaint Only Masked") pastes the decoded region back
        // into the original-size canvas afterward, so the extra resolution can't survive
        // that path - force it back down instead of letting the composite break.
        if (mode == "upscale" && g.MaskShrunkInfo?.BoundsNode is not null)
        {
            Logs.Warning("[VAE Utils] a masked-crop composite is active, which requires a normally sized image; downscaling instead.");
            mode = "downscale";
        }

        string vaeLoader = g.CreateNode("VAEUtils_CustomVAELoader", new JObject()
        {
            ["vae_name"] = vaeModel.ToString(g.ModelFolderFormat),
            ["disable_offload"] = true
        });

        bool tiled = g.UserInput.TryGet(T2IParamTypes.VAETileSize, out _) || g.UserInput.TryGet(T2IParamTypes.VAETemporalTileSize, out _);
        string decodeNode = g.CreateNode("VAEUtils_VAEDecodeTiled", new JObject()
        {
            ["samples"] = g.CurrentMedia.Path,
            ["vae"] = new JArray() { vaeLoader, 0 },
            ["upscale"] = -1, // auto-detect from the decoder's output channel count
            ["tile"] = tiled,
            ["tile_size"] = g.UserInput.Get(T2IParamTypes.VAETileSize, 256),
            ["overlap"] = g.UserInput.Get(T2IParamTypes.VAETileOverlap, 64),
            ["temporal_size"] = g.UserInput.Get(T2IParamTypes.VAETemporalTileSize, g.IsAnyWanModel() ? 9999 : 32),
            ["temporal_overlap"] = g.UserInput.Get(T2IParamTypes.VAETemporalTileOverlap, 4)
        }, "8");
        g.CurrentMedia = g.CurrentMedia.WithPath([decodeNode, 0], WGNodeData.DT_IMAGE, g.CurrentVae.Compat);

        if (mode == "downscale")
        {
            string filter = g.UserInput.Get(DownscaleFilter, "area");
            string scaleNode = g.CreateNode("ImageScaleBy", new JObject()
            {
                ["image"] = new JArray() { decodeNode, 0 },
                ["upscale_method"] = filter,
                ["scale_by"] = 1.0 / factor
            });
            g.CurrentMedia = g.CurrentMedia.WithPath([scaleNode, 0], WGNodeData.DT_IMAGE, g.CurrentVae.Compat);
            g.UserInput.ExtraMeta["vae_utils_filter"] = filter;
        }

        g.UserInput.ExtraMeta["vae_utils_vae"] = vaeModel.Name;
        g.UserInput.ExtraMeta["vae_utils_mode"] = mode;
    }
}
