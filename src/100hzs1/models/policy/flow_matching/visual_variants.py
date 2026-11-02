"""The four supported B0 visual architectures; training settings are independent."""
from copy import deepcopy

VISUAL_VARIANTS = ("baseline", "stage4", "rms", "stage4_rms")
STAGE4_VARIANTS = ("stage4", "stage4_rms")
RMS_VARIANTS = ("rms", "stage4_rms")
RETIRED_BACKBONE_OPTIONS = ("stage4_repeats", "stage4_pool_shape", "stage4_repeat_position",
                            "mbconv_downsample_blocks", "mbconv_downsample_init")
RETIRED_VISION_OPTIONS = ("compress_before_fusion", "output_projection_type")


def validate_visual_variant(value):
    if value not in VISUAL_VARIANTS:
        raise ValueError(f"visual_variant must be one of {VISUAL_VARIANTS}, got {value!r}")
    return value


def validate_policy_visual_variant(kwargs):
    obs = kwargs.get("obs_encoder") or {}
    vision = obs.get("emvis_config") or {}
    backbone = vision.get("backbone_config") or {}
    is_b0 = (vision.get("backbone_target", "").lower() == "efficientvit"
             and backbone.get("model_name", "").startswith("efficientvit_b0."))
    variant = kwargs.get("visual_variant")
    if not is_b0:
        if variant not in (None, "baseline"):
            raise ValueError("B0 visual variants require the EfficientViT-B0 backbone")
        return kwargs
    variant = validate_visual_variant(variant or "baseline")
    kwargs = dict(kwargs)
    kwargs["visual_variant"] = variant
    pooling = kwargs.get("visual_pooling", True)
    if type(pooling) is not bool:
        raise ValueError("visual_pooling must be a bool")
    kwargs["visual_pooling"] = pooling
    width = 1024 if variant in RMS_VARIANTS else 2048
    expected = {"hidden_size": width, "num_heads": width // 64,
                "scene_adaptor": "residual_mlp2x_gelu" if variant in RMS_VARIANTS else "mlp2x_gelu",
                "state_adaptor": "mlp3x_gelu"}
    # Explicit visual-normalization ablation; no change to the four default presets.
    if (variant in RMS_VARIANTS
            and kwargs.get("scene_adaptor") == "residual_mlp2x_gelu_no_norm"):
        expected["scene_adaptor"] = "residual_mlp2x_gelu_no_norm"
    for key, value in expected.items():
        if kwargs.get(key, value) != value:
            raise ValueError(f"visual_variant={variant!r} fixes {key}={value!r}")
        kwargs[key] = value
    vision = deepcopy(vision)
    for name in RETIRED_VISION_OPTIONS:
        if name in vision:
            raise ValueError(f"{name} was removed; select visual_variant instead")
    for name in RETIRED_BACKBONE_OPTIONS:
        if name in backbone:
            raise ValueError(f"{name} was removed; select visual_variant instead")
    if vision.get("visual_variant", variant) != variant:
        raise ValueError("policy and visual encoder visual_variant must agree")
    vision["visual_variant"] = variant
    if vision.get("efficientvit_fusion_mode", "native") != "native":
        raise ValueError("B0 visual variants require native 128-dimensional fusion")
    if vision.get("backbone_trainable", False):
        raise ValueError("B0 visual variants keep the pretrained backbone frozen")
    adapter = vision.get("model_adapter_config", {})
    expected_adapter = {"dim_out": 1024, "conv_num": None if variant in STAGE4_VARIANTS else 2}
    for key, value in expected_adapter.items():
        if adapter.get(key) != value:
            raise ValueError(f"visual_variant={variant!r} fixes model_adapter_config.{key}={value!r}")
    if adapter.get("use_pooling", True) != pooling:
        raise ValueError("visual_pooling and model_adapter_config.use_pooling must agree")
    grid = [2, 2] if pooling else [2, 3]
    if list(adapter.get("shape_out", [])) != grid:
        raise ValueError(f"visual_pooling={pooling} requires shape_out={grid}")
    if adapter.get("target", "ResNetAdapter") != "ResNetAdapter":
        raise ValueError("B0 visual variants retain the original pooling/Linear adapter")
    if any(adapter.get(key) for key in ("stage_blocks", "dim_ratio", "use_max_pool", "use_pixelshuffle")):
        raise ValueError("B0 spatial compression is fixed by visual_variant")
    kwargs["obs_encoder"] = dict(obs, emvis_config=vision)
    return kwargs
