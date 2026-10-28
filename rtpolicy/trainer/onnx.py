import argparse
import importlib
import inspect
import json
import logging
import os
import shutil
from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_model
from transformers import AutoConfig

from rtpolicy.models.base_runner import load_policy
from rtpolicy.common.checkpoint import load_checkpoint_config, normalize_checkpoint_metadata


_ORIGINAL_INTERPOLATE = F.interpolate
_ORIGINAL_TORCHVISION_RESIZE = None
_ORIGINAL_TORCH_ONNX_GET_MODULE_ATTRIBUTES = None
logger = logging.getLogger(__name__)
_FALSEY_EXPORT_VALUES = {"", "0", "false", "no", "none", "null", "off", "disable", "disabled"}
_TRUTHY_EXPORT_VALUES = {"1", "true", "yes", "on", "all"}


def _patch_mha_fastpath_for_onnx() -> None:
    mha_backend = getattr(torch.backends, "mha", None)
    if mha_backend is None or not hasattr(mha_backend, "set_fastpath_enabled"):
        return

    if hasattr(mha_backend, "get_fastpath_enabled") and mha_backend.get_fastpath_enabled():
        logger.info("Disabling torch.backends.mha fastpath for ONNX export.")
    mha_backend.set_fastpath_enabled(False)


def _patch_interpolate_for_onnx() -> None:
    def _interpolate_patched(
        input,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=None,
        antialias=False,
    ):
        return _ORIGINAL_INTERPOLATE(
            input,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=False,
        )

    F.interpolate = _interpolate_patched


def _patch_torchvision_resize_for_onnx() -> None:
    global _ORIGINAL_TORCHVISION_RESIZE
    try:
        import torchvision.transforms.functional as tv_functional
    except ImportError:
        return

    if _ORIGINAL_TORCHVISION_RESIZE is None:
        _ORIGINAL_TORCHVISION_RESIZE = tv_functional.resize

    def _resize_patched(img, size, interpolation, max_size=None, antialias=True):
        return _ORIGINAL_TORCHVISION_RESIZE(
            img,
            size,
            interpolation=interpolation,
            max_size=max_size,
            antialias=False,
        )

    tv_functional.resize = _resize_patched


def _patch_position_getters_for_onnx() -> None:
    module_names = [
        "rtpolicy.models.policy.rtpolicy.emvis.encoder.dinov3cnn_encoder",
        "rtpolicy.models.policy.rtpolicy.emvis.encoder.dinov3_encoder",
        "rtpolicy.models.policy.rtpolicy.emvis.encoder.da3_encoder",
    ]

    def _position_getter_call(self, batch_size: int, height: int, width: int, device: torch.device):
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            positions = torch.stack((grid_y, grid_x), dim=-1).reshape(-1, 2)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        position_getter_cls = getattr(module, "PositionGetter", None)
        if position_getter_cls is not None:
            position_getter_cls.__call__ = _position_getter_call


def _is_supported_onnx_scope_attr(value) -> bool:
    if value is None or isinstance(value, (bool, int, float, str, torch.Tensor)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_supported_onnx_scope_attr(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_supported_onnx_scope_attr(item)
            for key, item in value.items()
        )
    return False


def _patch_torch_onnx_scope_attributes() -> None:
    global _ORIGINAL_TORCH_ONNX_GET_MODULE_ATTRIBUTES
    try:
        import torch.onnx.utils as onnx_utils
    except ImportError:
        return

    if getattr(onnx_utils, "_rtpolicy_scope_attr_filter_patched", False):
        return

    _ORIGINAL_TORCH_ONNX_GET_MODULE_ATTRIBUTES = onnx_utils._get_module_attributes

    def _get_module_attributes_onnx_safe(module):
        attrs = _ORIGINAL_TORCH_ONNX_GET_MODULE_ATTRIBUTES(module)
        return {
            key: value
            for key, value in attrs.items()
            if _is_supported_onnx_scope_attr(value)
        }

    onnx_utils._get_module_attributes = _get_module_attributes_onnx_safe
    onnx_utils._rtpolicy_scope_attr_filter_patched = True


def _patch_dinov3_encoders_for_onnx() -> None:
    try:
        dinov3cnn_module = importlib.import_module(
            "rtpolicy.models.policy.rtpolicy.emvis.encoder.dinov3cnn_encoder"
        )
    except ImportError:
        dinov3cnn_module = None

    if dinov3cnn_module is not None:
        dinov3cnn_cls = getattr(dinov3cnn_module, "DINOv3Encoder", None)
        if dinov3cnn_cls is not None and not getattr(dinov3cnn_cls, "_onnx_forward_patched", False):

            def _dinov3cnn_forward(self, images: torch.Tensor, V_fake: int = None):
                if len(images.shape) == 4:
                    images = images.unsqueeze(0)
                batch_size, views, channels, height, width = images.shape
                images = images.reshape(-1, channels, height, width)
                feats = super(dinov3cnn_cls, self).forward(images, return_dict=False)[0]
                patch_pos = self.position_getter(
                    batch_size * views,
                    height // self.patch_size_log,
                    width // self.patch_size_log,
                    device=images.device,
                )
                patch_pos = patch_pos.reshape((batch_size, views, *patch_pos.shape[-2:]))
                return {
                    "image_tokens": feats[:, 1:, :].view(batch_size, views, -1, feats.shape[-1]),
                    "image_tokens_pos": patch_pos,
                    "camera_tokens_list": None,
                    "spatial_tokens_pos": patch_pos,
                    "spatial_tokens_list": None,
                }

            dinov3cnn_cls.forward = _dinov3cnn_forward
            dinov3cnn_cls._onnx_forward_patched = True

    try:
        dinov3_module = importlib.import_module(
            "rtpolicy.models.policy.rtpolicy.emvis.encoder.dinov3_encoder"
        )
    except ImportError:
        dinov3_module = None

    if dinov3_module is not None:
        dinov3_cls = getattr(dinov3_module, "DINOv3Encoder", None)
        if dinov3_cls is not None and not getattr(dinov3_cls, "_onnx_forward_patched", False):

            def _dinov3_forward(self, images: torch.Tensor, V_fake: int = None):
                if len(images.shape) == 4:
                    images = images.unsqueeze(0)
                batch_size, views, channels, height, width = images.shape
                images = images.reshape(-1, channels, height, width)
                feats = super(dinov3_cls, self).forward(images, return_dict=False)[0]
                patch_pos = self.position_getter(
                    batch_size * views,
                    height // self.patch_size_log,
                    width // self.patch_size_log,
                    device=images.device,
                )
                sp_len = self.embeddings.cls_token.shape[1] + self.embeddings.register_tokens.shape[1]
                patch_pos = patch_pos.reshape((batch_size, views, *patch_pos.shape[-2:]))
                return {
                    "image_tokens": feats[:, sp_len:, :],
                    "image_tokens_pos": patch_pos,
                    "camera_tokens_list": None,
                    "spatial_tokens_pos": patch_pos,
                    "spatial_tokens_list": None,
                }

            dinov3_cls.forward = _dinov3_forward
            dinov3_cls._onnx_forward_patched = True


def _patch_adaptive_pool2d_modules_for_onnx(module: torch.nn.Module) -> None:
    def _pool_forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.interpolate(input, size=self.output_size, mode="area")

    for child in module.modules():
        if isinstance(child, (torch.nn.AdaptiveAvgPool2d, torch.nn.AdaptiveMaxPool2d)):
            child.forward = _pool_forward.__get__(child, child.__class__)


def _patch_adapter_for_onnx() -> None:
    try:
        from rtpolicy.models.policy.rtpolicy.emvis import adapter as adapter_module
    except ImportError:
        return

    def _adaptive_pool_ranges(input_size: int, output_size: int) -> List[Tuple[int, int]]:
        import math

        return [
            (
                math.floor(idx * input_size / output_size),
                math.ceil((idx + 1) * input_size / output_size),
            )
            for idx in range(output_size)
        ]

    def _adaptive_pool2d_onnx_friendly(
        x: torch.Tensor,
        shape_out: Tuple[int, int],
        use_max_pool: bool = False,
    ) -> torch.Tensor:
        out_h, out_w = tuple(int(dim) for dim in shape_out)
        in_h, in_w = int(x.shape[-2]), int(x.shape[-1])
        row_ranges = _adaptive_pool_ranges(in_h, out_h)
        col_ranges = _adaptive_pool_ranges(in_w, out_w)

        pooled_rows = []
        for h_start, h_end in row_ranges:
            pooled_cols = []
            for w_start, w_end in col_ranges:
                patch = x[..., h_start:h_end, w_start:w_end]
                if use_max_pool:
                    pooled = patch.max(dim=-1, keepdim=True).values.max(
                        dim=-2, keepdim=True
                    ).values
                else:
                    pooled = patch.mean(dim=-1, keepdim=True).mean(dim=-2, keepdim=True)
                pooled_cols.append(pooled)
            pooled_rows.append(torch.cat(pooled_cols, dim=-1))
        return torch.cat(pooled_rows, dim=-2)

    def _adaptive_patch_merging_forward(self, x: torch.Tensor, shape_out: Tuple[int, int]):
        shape_out = tuple(int(dim) for dim in shape_out)
        if torch.onnx.is_in_onnx_export() and self.adapter in (
            torch.nn.AdaptiveAvgPool2d,
            torch.nn.AdaptiveMaxPool2d,
        ):
            x = _adaptive_pool2d_onnx_friendly(
                x,
                shape_out,
                use_max_pool=self.adapter is torch.nn.AdaptiveMaxPool2d,
            )
        else:
            x = self.adapter(shape_out)(x)
        x = x.permute(0, 2, 3, 1)
        x = self.proj(x)
        x = self.drop(x)
        return x

    adaptive_patch_merging_cls = getattr(adapter_module, "AdaptivePatchMerging", None)
    if adaptive_patch_merging_cls is not None:
        adaptive_patch_merging_cls.forward = _adaptive_patch_merging_forward


def _patch_rms_norm_for_onnx() -> None:
    try:
        from timm.layers.norm import RmsNorm
    except ImportError:
        return

    def _rms_norm_forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized_shape = getattr(self, "normalized_shape", x.shape[-1:])
        if isinstance(normalized_shape, int):
            normalized_rank = 1
        else:
            normalized_rank = len(tuple(normalized_shape))
        dims = tuple(range(-normalized_rank, 0))
        variance = x.pow(2).mean(dim=dims, keepdim=True)
        x = x / torch.sqrt(variance + self.eps)
        weight = getattr(self, "weight", None)
        if weight is not None:
            x = x * weight
        return x

    RmsNorm.forward = _rms_norm_forward



def patch_for_onnx_export() -> None:
    _patch_mha_fastpath_for_onnx()
    _patch_torch_onnx_scope_attributes()
    _patch_interpolate_for_onnx()
    _patch_torchvision_resize_for_onnx()
    _patch_position_getters_for_onnx()
    _patch_dinov3_encoders_for_onnx()
    _patch_adapter_for_onnx()
    _patch_rms_norm_for_onnx()


def _maybe_get(mapping, key, default=None):
    if mapping is None:
        return default
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _get_runtime_emvis_override(args) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}

    explicit_model_path = _maybe_get(args, "backbone_path")
    if explicit_model_path:
        overrides["backbone_path"] = explicit_model_path

    return overrides


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def _load_checkpoint_config_dict(checkpoint_dir: str) -> Dict[str, Any]:
    return load_checkpoint_config(checkpoint_dir)


def _get_checkpoint_emvis_config(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    obs_encoder_cfg = config_dict.get("obs_encoder")
    if not isinstance(obs_encoder_cfg, dict):
        raise ValueError("Checkpoint config.json is missing obs_encoder config required for ONNX export.")

    emvis_cfg = obs_encoder_cfg.get("emvis_config")
    if not isinstance(emvis_cfg, dict):
        raise ValueError(
            "Checkpoint config.json is missing obs_encoder.emvis_config required for ONNX export."
        )
    return emvis_cfg


def _resolve_backbone_path(backbone_path: str, backbone_target: str) -> str:
    if not backbone_path:
        raise ValueError(f"Missing backbone_path for backbone target: {backbone_target}")

    candidates = []
    if os.path.isabs(backbone_path):
        candidates.append(backbone_path)
    else:
        repo_root = _repo_root()
        basename = os.path.basename(os.path.normpath(backbone_path))
        candidates.extend(
            [
                backbone_path,
                os.path.abspath(backbone_path),
                os.path.join(repo_root, backbone_path),
            ]
        )
        if basename:
            candidates.append(os.path.join(repo_root, basename))

    checked = set()
    for candidate in candidates:
        if not candidate or candidate in checked:
            continue
        checked.add(candidate)
        if os.path.exists(candidate):
            return candidate

    basename = os.path.basename(os.path.normpath(backbone_path))
    if basename and "dinov3" in str(backbone_target).lower():
        return f"facebook/{basename}"
    return backbone_path


def _infer_checkpoint_adapter_input_dim(checkpoint_dir: str) -> Optional[int]:
    safetensor_path = os.path.join(checkpoint_dir, "model.safetensors")
    tensor_keys = [
        "obs_encoder.scene_encoder.model_adapter.adapter.patch_merging.proj.weight",
        "obs_encoder.scene_encoder.model_adapter.adapter.downsample_blocks.0_0.mainstream.conv1.weight",
    ]

    try:
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            available_keys = set(f.keys())
            for key in tensor_keys:
                if key not in available_keys:
                    continue
                shape = tuple(f.get_tensor(key).shape)
                if key.endswith("patch_merging.proj.weight") and len(shape) >= 2:
                    return int(shape[1])
                if key.endswith("conv1.weight") and len(shape) >= 2:
                    return int(shape[1])
    except Exception as exc:
        logger.warning("Failed to inspect checkpoint adapter width from %s: %s", safetensor_path, exc)

    return None


def _infer_backbone_output_dim(backbone_target: str, backbone_path: str) -> Tuple[int, str]:
    resolved_model_path = _resolve_backbone_path(backbone_path, backbone_target)
    config = AutoConfig.from_pretrained(resolved_model_path, trust_remote_code=True)
    if hasattr(config, "hidden_sizes"):
        return int(config.hidden_sizes[-1]), resolved_model_path
    return int(config.hidden_size), resolved_model_path


def _validate_emvis_runtime_config(
    checkpoint_dir: str,
    checkpoint_emvis_cfg: Dict[str, Any],
    effective_emvis_cfg: Dict[str, Any],
) -> None:
    checkpoint_target = checkpoint_emvis_cfg.get("backbone_target")
    checkpoint_model_path = checkpoint_emvis_cfg.get("backbone_path")
    effective_target = effective_emvis_cfg.get("backbone_target")
    effective_model_path = effective_emvis_cfg.get("backbone_path")
    expected_adapter_dim = _infer_checkpoint_adapter_input_dim(checkpoint_dir)

    logger.info(
        "ONNX EmVis backbone: checkpoint target=%s, checkpoint model_path=%s, "
        "effective target=%s, effective model_path=%s, checkpoint adapter input dim=%s",
        checkpoint_target,
        checkpoint_model_path,
        effective_target,
        effective_model_path,
        expected_adapter_dim,
    )

    # The width cross-check below assumes a plain dinov3 backbone resolved via
    # AutoConfig; da3_vodpp encoders carry their own backbone config and would
    # produce false mismatches.
    if (
        not effective_target
        or not effective_model_path
        or "dinov3" not in str(effective_target).lower()
        or "da3" in str(effective_target).lower()
    ):
        return

    try:
        runtime_dim, resolved_model_path = _infer_backbone_output_dim(
            str(effective_target),
            str(effective_model_path),
        )
    except Exception:
        resolved_model_path = _resolve_backbone_path(str(effective_model_path), str(effective_target))
        raise RuntimeError(
            "ONNX export cannot load the required EmVis backbone. "
            f"Checkpoint expects backbone_target={checkpoint_target!r}, "
            f"backbone_path={checkpoint_model_path!r}. "
            f"Effective export config uses backbone_target={effective_target!r}, "
            f"backbone_path={effective_model_path!r}. "
            f"Resolved lookup target: {resolved_model_path!r}. "
            "Provide the matching local backbone directory via "
            "`VGGT_MODEL_PATH=/abs/path/to/dinov3-convnext-large-pretrain-lvd1689m` "
            "or place that directory under the repo root."
        ) from None

    if expected_adapter_dim is not None and runtime_dim != expected_adapter_dim:
        raise ValueError(
            "Incompatible ONNX EmVis backbone configuration. "
            f"Checkpoint adapter expects backbone width {expected_adapter_dim}, "
            f"but {effective_target!r} from {resolved_model_path!r} resolves to width {runtime_dim}. "
            f"Checkpoint config records backbone_target={checkpoint_target!r}, "
            f"backbone_path={checkpoint_model_path!r}. "
            "Remove `VGGT_MODEL_PATH` or point it to the matching checkpoint backbone."
        )


def _materialize_emvis_runtime_config(emvis_cfg: Dict[str, Any]) -> Dict[str, Any]:
    resolved_cfg = dict(emvis_cfg)
    backbone_target = resolved_cfg.get("backbone_target")
    backbone_path = resolved_cfg.get("backbone_path")

    if backbone_target and backbone_path and "dinov3" in str(backbone_target).lower():
        resolved_cfg["backbone_path"] = _resolve_backbone_path(
            str(backbone_path), str(backbone_target)
        )

    return resolved_cfg


def _get_rgb_keys(policy, override_keys: Optional[List[str]] = None) -> List[str]:
    if override_keys is not None:
        return list(override_keys)

    obs_encoder = getattr(policy, "obs_encoder", None)
    runtime_rgb_keys = getattr(obs_encoder, "rgb_key", None)
    if runtime_rgb_keys:
        return list(runtime_rgb_keys)

    obs_meta = policy.config.shape_meta["obs"]
    obs_encoder_cfg = getattr(policy.config, "obs_encoder", None)
    rgb_keys = _maybe_get(obs_encoder_cfg, "rgb_key")
    if rgb_keys:
        return list(rgb_keys)
    return [
        key
        for key, value in obs_meta.items()
        if value is not None and value.get("type") == "rgb"
    ]


def _resolve_dinov3cnn_rgb_keys(args) -> Optional[List[str]]:
    if args is None:
        return None
    value = _maybe_get(args, "dinov3cnn_rgb_keys", None)
    if value is None:
        value = _maybe_get(args, "dinov3cnn_rgb_key", None)
    if value is None:
        return None
    if isinstance(value, str):
        keys = [item.strip() for item in value.split(",") if item.strip()]
    else:
        keys = [str(item) for item in list(value)]
    if not keys:
        raise ValueError("dinov3cnn_rgb_keys is set but empty.")
    return keys


def _resolve_visual_encoder_rgb_keys(args) -> Optional[List[str]]:
    if args is None:
        return None
    value = _maybe_get(args, "visual_encoder_rgb_keys", None)
    if value is None:
        value = _maybe_get(args, "visual_encoder_rgb_key", None)
    if value is None:
        return _resolve_dinov3cnn_rgb_keys(args)
    if isinstance(value, str):
        keys = [item.strip() for item in value.split(",") if item.strip()]
    else:
        keys = [str(item) for item in list(value)]
    if not keys:
        raise ValueError("visual_encoder_rgb_keys is set but empty.")
    return keys


def _get_state_key(policy) -> str:
    obs_meta = policy.config.shape_meta["obs"]
    candidates = [
        getattr(policy.config, "state_traj_key", None),
        getattr(policy.config, "action_key", None),
        "state",
        "agent_pos",
    ]
    for key in candidates:
        if key and key in obs_meta and obs_meta[key] is not None:
            return str(key)

    for key, value in obs_meta.items():
        if value is not None and value.get("type") == "low_dim":
            return str(key)
    return "agent_pos"


def _ordered_input_names_for_export(
    policy,
    input_tensors: Dict[str, torch.Tensor],
) -> List[str]:
    ordered_names: List[str] = []
    signature = inspect.signature(policy.forward)
    for name in signature.parameters.keys():
        if name == "self":
            continue
        if name in input_tensors:
            ordered_names.append(name)

    # Keep backward compatibility if model forward adds **kwargs-only inputs.
    for name in input_tensors.keys():
        if name not in ordered_names:
            ordered_names.append(name)

    return ordered_names


def _normalize_export_modules_token(token: str) -> str:
    return token.strip().lower().replace("-", "_")


def _dedupe_module_types(module_types: List[type]) -> Tuple[type, ...]:
    unique_types: List[type] = []
    seen = set()
    for module_type in module_types:
        if not inspect.isclass(module_type):
            continue
        if not issubclass(module_type, torch.nn.Module):
            continue
        if module_type in seen:
            continue
        seen.add(module_type)
        unique_types.append(module_type)
    return tuple(unique_types)


def _build_export_module_aliases(policy) -> Dict[str, type]:
    aliases: Dict[str, type] = {}

    def _register(alias: str, module, allow_torch_nn: bool = False) -> None:
        if module is None or not isinstance(module, torch.nn.Module):
            return
        module_type = type(module)
        if (
            not allow_torch_nn
            and getattr(module_type, "__module__", "").startswith("torch.nn.modules")
        ):
            return
        aliases.setdefault(_normalize_export_modules_token(alias), module_type)
        aliases.setdefault(module_type.__name__.lower(), module_type)

    _register("policy", policy)
    _register("obs_encoder", getattr(policy, "obs_encoder", None))
    obs_encoder = getattr(policy, "obs_encoder", None)
    _register("scene_encoder", getattr(obs_encoder, "scene_encoder", None))
    _register("scene_adaptor", getattr(policy, "scene_adaptor", None))
    _register("state_adaptor", getattr(policy, "state_adaptor", None))
    _register("diffusion", getattr(policy, "model", None))
    _register("diffusion_model", getattr(policy, "model", None))
    return aliases


def _resolve_export_modules_as_functions(policy, raw_value: Any) -> Any:
    if raw_value is None:
        return False
    if isinstance(raw_value, bool):
        return raw_value
    if inspect.isclass(raw_value) and issubclass(raw_value, torch.nn.Module):
        return {raw_value}
    if isinstance(raw_value, (list, tuple, set)):
        if all(inspect.isclass(item) and issubclass(item, torch.nn.Module) for item in raw_value):
            return set(_dedupe_module_types(list(raw_value)))
        joined_tokens = ",".join(str(item) for item in raw_value)
        return _resolve_export_modules_as_functions(policy, joined_tokens)
    if not isinstance(raw_value, str):
        return bool(raw_value)

    normalized_value = _normalize_export_modules_token(raw_value)
    if normalized_value in _FALSEY_EXPORT_VALUES:
        return False
    if normalized_value in _TRUTHY_EXPORT_VALUES:
        return True

    aliases = _build_export_module_aliases(policy)
    resolved_types: List[type] = []
    for token in [_normalize_export_modules_token(item) for item in raw_value.split(",") if item.strip()]:
        if token in _FALSEY_EXPORT_VALUES:
            continue
        if token in _TRUTHY_EXPORT_VALUES:
            return True
        if token == "minimal":
            module_type = aliases.get("diffusion")
            if module_type is not None:
                resolved_types.append(module_type)
            continue
        if token in {"safe", "netron"}:
            module_type = aliases.get("diffusion")
            if module_type is not None:
                resolved_types.append(module_type)
            continue
        if token in aliases:
            resolved_types.append(aliases[token])
            continue
        supported_tokens = sorted(
            {"false", "true", "safe", "netron", "minimal", *aliases.keys()}
        )
        raise ValueError(
            "Unsupported export_modules_as_functions value "
            f"{raw_value!r}. Supported presets/tokens: {supported_tokens}"
        )

    resolved_types = list(_dedupe_module_types(resolved_types))
    if not resolved_types:
        return False
    return set(resolved_types)


def _export_external_data_artifact(
    onnx_path: str,
    external_data_filename: Optional[str] = None,
    external_data_size_threshold: int = 1024,
) -> str:
    try:
        import onnx
        from onnx import external_data_helper
    except ImportError as exc:
        raise RuntimeError(
            "use_external_data=true requires the `onnx` package, but it is unavailable in the current environment."
        ) from exc

    onnx_abs_path = os.path.abspath(onnx_path)
    onnx_dir = os.path.dirname(onnx_abs_path)
    data_filename = (
        os.path.basename(external_data_filename)
        if external_data_filename
        else f"{os.path.basename(onnx_abs_path)}.data"
    )
    data_path = os.path.join(onnx_dir, data_filename)

    existing_model = onnx.load_model(onnx_abs_path, load_external_data=False)
    stale_external_paths = set()
    for tensor in existing_model.graph.initializer:
        location = None
        for entry in tensor.external_data:
            if entry.key == "location":
                location = entry.value
                break
        if not location:
            continue
        candidate_path = os.path.abspath(os.path.join(onnx_dir, location))
        if os.path.commonpath([onnx_dir, candidate_path]) != onnx_dir:
            continue
        stale_external_paths.add(candidate_path)

    onnx_model = onnx.load_model(onnx_abs_path, load_external_data=True)
    external_data_helper.convert_model_from_external_data(onnx_model)

    if os.path.exists(data_path):
        os.remove(data_path)

    onnx.save_model(
        onnx_model,
        onnx_abs_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=max(int(external_data_size_threshold), 0),
        convert_attribute=True,
    )

    if not os.path.exists(data_path):
        raise RuntimeError(
            f"ONNX external data export failed: expected external data file at {data_path!r}."
        )

    for artifact_path in (onnx_abs_path, data_path):
        try:
            os.chmod(artifact_path, 0o644)
        except OSError:
            pass


    for stale_path in stale_external_paths:
        if stale_path == data_path:
            continue
        if os.path.exists(stale_path):
            os.remove(stale_path)

    return data_path


def _summarize_onnx_graph(onnx_path: str) -> Optional[Dict[str, Any]]:
    try:
        import onnx
    except ImportError:
        return None

    model = onnx.load_model(onnx_path, load_external_data=False)
    return {
        "ir_version": int(model.ir_version),
        "opset_imports": {entry.domain or "ai.onnx": int(entry.version) for entry in model.opset_import},
        "node_count": len(model.graph.node),
        "initializer_count": len(model.graph.initializer),
        "function_count": len(model.functions),
        "input_names": [value.name for value in model.graph.input],
        "output_names": [value.name for value in model.graph.output],
    }


def _default_netron_model_path(onnx_path: str) -> str:
    root, ext = os.path.splitext(onnx_path)
    return f"{root}.netron{ext or '.onnx'}"


def _clear_tensor_payload_for_netron(tensor) -> None:
    for field_name in (
        "raw_data",
        "float_data",
        "int32_data",
        "string_data",
        "int64_data",
        "double_data",
        "uint64_data",
        "external_data",
    ):
        tensor.ClearField(field_name)
    tensor.data_location = 0


def _clear_node_attribute_payloads_for_netron(nodes) -> None:
    for node in nodes:
        for attr in node.attribute:
            if attr.HasField("t"):
                _clear_tensor_payload_for_netron(attr.t)
            if attr.HasField("g"):
                _clear_graph_payloads_for_netron(attr.g)
            for tensor in attr.tensors:
                _clear_tensor_payload_for_netron(tensor)
            for graph in attr.graphs:
                _clear_graph_payloads_for_netron(graph)


def _clear_graph_payloads_for_netron(graph) -> None:
    for tensor in graph.initializer:
        _clear_tensor_payload_for_netron(tensor)
    for sparse_tensor in graph.sparse_initializer:
        _clear_tensor_payload_for_netron(sparse_tensor.values)
        _clear_tensor_payload_for_netron(sparse_tensor.indices)
    _clear_node_attribute_payloads_for_netron(graph.node)


def _write_netron_view_model(
    onnx_path: str,
    netron_model_path: Optional[str] = None,
) -> Optional[str]:
    try:
        import onnx
    except ImportError:
        logger.warning("Skipping Netron view model because the `onnx` package is unavailable.")
        return None

    source_path = os.path.abspath(onnx_path)
    view_path = os.path.abspath(netron_model_path or _default_netron_model_path(onnx_path))
    os.makedirs(os.path.dirname(view_path), exist_ok=True)

    model = onnx.load_model(source_path, load_external_data=False)
    _clear_graph_payloads_for_netron(model.graph)
    for function in model.functions:
        _clear_node_attribute_payloads_for_netron(function.node)

    marker = model.metadata_props.add()
    marker.key = "netron_visualization_only"
    marker.value = "true; initializer payloads were stripped and this file is not runnable."

    onnx.save_model(model, view_path)
    try:
        os.chmod(view_path, 0o644)
    except OSError:
        pass
    return view_path


def _disable_export_only_features(policy, export_dtype: torch.dtype = torch.float32) -> None:
    obs_encoder = getattr(policy, "obs_encoder", None)
    scene_encoder = getattr(obs_encoder, "scene_encoder", None)
    if scene_encoder is not None:
        if hasattr(scene_encoder, "visualize"):
            scene_encoder.visualize = False

    for module in policy.modules():
        if hasattr(module, "dtype"):
            try:
                module.dtype = export_dtype
            except Exception:
                pass
    _patch_adaptive_pool2d_modules_for_onnx(policy)


def _load_policy_from_config_dict(
    config_dict: Dict[str, Any],
    checkpoint_dir: str,
    device: str,
):
    config_dict = normalize_checkpoint_metadata(config_dict)
    config_cls = hydra.utils.get_class(config_dict["pkg_map"]["AutoConfig"])
    policy_cls = hydra.utils.get_class(config_dict["pkg_map"]["AutoModel"])

    if hasattr(config_cls, "from_customed_dict"):
        policy_config = config_cls.from_customed_dict(config_dict)
    else:
        policy_config = config_cls(**config_dict)

    policy = policy_cls(policy_config)
    policy = policy.to(device)
    load_model(policy, os.path.join(checkpoint_dir, "model.safetensors"), strict=False, device=device)
    return policy


def _load_policy_for_onnx(
    checkpoint_dir: str,
    args=None,
    device: str = "cpu",
):
    emvis_overrides = _get_runtime_emvis_override(args)
    config_dict = _load_checkpoint_config_dict(checkpoint_dir)
    checkpoint_emvis_cfg = dict(_get_checkpoint_emvis_config(config_dict))
    effective_emvis_cfg = dict(checkpoint_emvis_cfg)
    effective_emvis_cfg.update(emvis_overrides)
    # from IPython import embed; embed()
    _validate_emvis_runtime_config(checkpoint_dir, checkpoint_emvis_cfg, effective_emvis_cfg)
    materialized_emvis_cfg = _materialize_emvis_runtime_config(effective_emvis_cfg)

    if not emvis_overrides and materialized_emvis_cfg == checkpoint_emvis_cfg:
        return load_policy(checkpoint_dir, use_ckp_code=False, device=device)

    obs_encoder_cfg = config_dict["obs_encoder"]
    obs_encoder_cfg["emvis_config"] = materialized_emvis_cfg
    return _load_policy_from_config_dict(config_dict, checkpoint_dir, device)


def _randn(shape: Tuple[int, ...], device: str, seed: Optional[int] = None) -> torch.Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
    tensor = torch.randn(shape, generator=generator, dtype=torch.float32)
    return tensor.to(device=device)


def _rand_image(shape: Tuple[int, ...], device: str, seed: Optional[int] = None) -> torch.Tensor:
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
    tensor = torch.rand(shape, generator=generator, dtype=torch.float32)
    return tensor.to(device=device)


def _to_tensor(value: Any, device: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        tensor = torch.as_tensor(value)
    return tensor.to(device=device, dtype=torch.float32)


def _resolve_export_target(value: Any) -> str:
    normalized = str(value or "policy").strip().lower()
    if not normalized or normalized in {"policy", "full", "full_policy"}:
        return "policy"
    if normalized in {"dinov3cnn", "dinov3cnn_only", "encoder", "encoder_only"}:
        return "dinov3cnn"
    if normalized in {
        "visual_encoder",
        "visualencoder",
        "vision_encoder",
        "visionencoder",
        "scene_encoder",
        "sceneencoder",
        "obs_encoder",
        "obsencoder",
        "emvis",
        "emvisrm",
    }:
        return "visual_encoder"
    raise ValueError(
        f"Unsupported export_target={value!r}. Supported values: 'policy', 'dinov3cnn', 'visual_encoder'."
    )


def _resolve_export_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        if value in {torch.float32, torch.float16}:
            return value
        raise ValueError("Unsupported export_dtype. Supported values: 'fp32', 'fp16'.")

    normalized = str(value or "fp32").strip().lower()
    if normalized in {"fp32", "float32", "float", "32"}:
        return torch.float32
    if normalized in {"fp16", "float16", "half", "16"}:
        return torch.float16
    raise ValueError(
        f"Unsupported export_dtype={value!r}. Supported values: 'fp32', 'fp16'."
    )


def _export_dtype_name(export_dtype: torch.dtype) -> str:
    export_dtype = _resolve_export_dtype(export_dtype)
    return "fp16" if export_dtype == torch.float16 else "fp32"


def _cast_floating_tensors(
    tensor_dict: Dict[str, torch.Tensor],
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    dtype = _resolve_export_dtype(dtype)
    casted: Dict[str, torch.Tensor] = {}
    for key, value in tensor_dict.items():
        if not isinstance(value, torch.Tensor):
            casted[key] = value
            continue
        if value.is_floating_point():
            casted[key] = value.to(device=device, dtype=dtype)
        elif device is not None:
            casted[key] = value.to(device=device)
        else:
            casted[key] = value
    return casted


def _device_type(device: str) -> str:
    try:
        return torch.device(device).type
    except (TypeError, RuntimeError):
        return str(device).split(":", 1)[0]


def _should_do_constant_folding(device: str) -> bool:
    if _device_type(device) == "cuda":
        logger.warning(
            "Disabling ONNX constant folding for device=%s because PyTorch legacy exporter can "
            "hit mixed CUDA/CPU tensor errors during this pass.",
            device,
        )
        return False
    return True


def _onnxruntime_providers_for_device(device: str) -> List[str]:
    providers = ["CPUExecutionProvider"]
    device_type = _device_type(device)
    if device_type != "cuda":
        return providers

    try:
        import onnxruntime as ort
    except ImportError:
        return providers

    available_providers = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    logger.warning(
        "device=%s was used for export, but onnxruntime CUDAExecutionProvider is unavailable; "
        "verification will fall back to CPUExecutionProvider.",
        device,
    )
    return providers


def _create_onnxruntime_session_for_verify(
    onnx_path: str,
    device: str,
    export_dtype: torch.dtype = torch.float32,
):
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    disabled_optimizers = None
    if _resolve_export_dtype(export_dtype) == torch.float16:
        logger.warning(
            "Disabling ONNXRuntime graph optimizations during fp16 verification to avoid "
            "optimizer-side fusion errors on large local-function graphs."
        )
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        disabled_optimizers = ["SimplifiedLayerNormFusion"]

    return ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=_onnxruntime_providers_for_device(device),
        disabled_optimizers=disabled_optimizers,
    )


def _serialize_config_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_config_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialize_config_value(item) for key, item in value.items()}
    return str(value)


def _resolve_interpolate_for_shape(interpolate: Any, width: int) -> Optional[Tuple[str, int]]:
    if interpolate is None or interpolate is False:
        return None
    if isinstance(interpolate, str):
        if interpolate.lower() in _FALSEY_EXPORT_VALUES:
            return None
        return interpolate, width
    if isinstance(interpolate, int):
        return "bilinear", int(interpolate)
    if isinstance(interpolate, (list, tuple)):
        if not interpolate:
            return None
        mode = interpolate[0]
        target_size = interpolate[1] if len(interpolate) > 1 else width
        if mode is None or str(mode).lower() in _FALSEY_EXPORT_VALUES:
            return None
        return str(mode), width if target_size is None else int(target_size)
    raise ValueError(f"Unsupported image interpolate config: {interpolate!r}")


def _compute_preprocessed_hw(
    height: int,
    width: int,
    mode: str = "crop",
    interpolate: Any = "bilinear",
    patch_size: int = 14,
) -> Tuple[int, int]:
    if mode not in {"crop", "pad"}:
        raise ValueError("Mode must be either 'crop' or 'pad'")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")

    resolved_interpolate = _resolve_interpolate_for_shape(interpolate, width)
    if resolved_interpolate is not None:
        _, target_size = resolved_interpolate
        target_width = round(target_size / patch_size) * patch_size
        height = round(height * (target_width / width) / patch_size) * patch_size
        width = target_width

    if mode == "pad":
        max_size = round(max(width, height) // patch_size) * patch_size
        return int(max_size), int(max_size)

    new_width = (width // patch_size) * patch_size
    new_height = min((height // patch_size) * patch_size, new_width)
    return int(new_height), int(new_width)


def _get_policy_emvis_config(policy) -> Dict[str, Any]:
    obs_encoder_cfg = getattr(policy.config, "obs_encoder", None)
    emvis_cfg = _maybe_get(obs_encoder_cfg, "emvis_config", {})
    if emvis_cfg is None:
        return {}
    if isinstance(emvis_cfg, dict):
        return dict(emvis_cfg)
    return dict(OmegaConf.to_container(emvis_cfg, resolve=True))


def _get_scene_encoder(policy):
    obs_encoder = getattr(policy, "obs_encoder", None)
    return getattr(obs_encoder, "scene_encoder", None)


def _get_scene_patch_size(policy, emvis_cfg: Optional[Dict[str, Any]] = None) -> int:
    scene_encoder = _get_scene_encoder(policy)
    backbone = getattr(scene_encoder, "backbone", None)
    patch_size = getattr(backbone, "patch_size_log", None)
    if patch_size is None:
        patch_size = getattr(backbone, "patch_size", None)
    if patch_size is None:
        target = str((emvis_cfg or {}).get("backbone_target", "")).lower()
        patch_size = 32 if "dinov3" in target and "cnn" in target else 14
    return int(patch_size)


def _build_image_preprocess_metadata(policy) -> Dict[str, Any]:
    rgb_keys = _get_rgb_keys(policy)
    if not rgb_keys:
        raise ValueError("ONNX export requires at least one rgb input.")

    obs_meta = policy.config.shape_meta["obs"]
    first_rgb_key = rgb_keys[0]
    raw_image_shape = tuple(obs_meta[first_rgb_key]["shape"])
    if len(raw_image_shape) != 3:
        raise ValueError(f"Expected rgb shape_meta for {first_rgb_key} to be [3,H,W], got {raw_image_shape}")
    channels, height, width = [int(dim) for dim in raw_image_shape]
    if channels != 3:
        raise ValueError(f"Expected RGB channel dimension 3 for {first_rgb_key}, got {channels}")

    emvis_cfg = _get_policy_emvis_config(policy)
    processor_config = dict(emvis_cfg.get("processor_config") or {})
    backbone_target = str(emvis_cfg.get("backbone_target", "")).lower()
    patch_size = _get_scene_patch_size(policy, emvis_cfg)
    mode = str(emvis_cfg.get("preprocess_mode", emvis_cfg.get("mode", "crop")))
    interpolate = emvis_cfg.get("interpolate", "bilinear")
    resize_height, resize_width = height, width
    resolved_interpolate = _resolve_interpolate_for_shape(interpolate, width)
    if resolved_interpolate is not None:
        _, target_size = resolved_interpolate
        resize_width = round(target_size / patch_size) * patch_size
        resize_height = round(height * (resize_width / width) / patch_size) * patch_size
    processed_height, processed_width = _compute_preprocessed_hw(
        height=height,
        width=width,
        mode=mode,
        interpolate=interpolate,
        patch_size=patch_size,
    )

    do_normalize = bool("dinov3" in backbone_target and processor_config.get("do_normalize", True))
    image_mean = processor_config.get("image_mean", [0.485, 0.456, 0.406])
    image_std = processor_config.get("image_std", [0.229, 0.224, 0.225])

    return {
        "mode": mode,
        "interpolate": _serialize_config_value(interpolate),
        "patch_size": patch_size,
        "input_range": "0_1",
        "do_normalize": do_normalize,
        "image_mean": _serialize_config_value(image_mean),
        "image_std": _serialize_config_value(image_std),
        "backbone_target": backbone_target,
        "raw_image_shape": [channels, height, width],
        "resize_shape": [channels, int(resize_height), int(resize_width)],
        "image_shape": [channels, processed_height, processed_width],
    }


def _build_onnx_runtime_metadata(
    policy,
    input_names: List[str],
    output_names: List[str],
    export_target: str,
    export_dtype: torch.dtype = torch.float32,
    image_preprocess_in_onnx: bool = False,
    rgb_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    export_dtype_name = _export_dtype_name(export_dtype)
    raw_input_format = "raw_images_state_noise" if export_target == "policy" else "raw_images"
    preprocessed_input_format = (
        "preprocessed_images_state_noise" if export_target == "policy" else "preprocessed_images"
    )
    n_obs_steps = int(policy.n_obs_steps)
    n_image_obs_steps = int(getattr(policy, "n_image_obs_steps", None) or n_obs_steps)
    num_inference_timesteps = int(getattr(policy, "num_inference_timesteps", 0))
    input_time_steps = max(n_obs_steps, n_image_obs_steps)
    obs_meta = policy.config.shape_meta["obs"]
    state_key = _get_state_key(policy) if export_target == "policy" else None
    return {
        "export_target": export_target,
        "export_dtype": export_dtype_name,
        "onnx_input_dtype": export_dtype_name,
        "input_names": list(input_names),
        "output_names": list(output_names),
        "rgb_keys": _get_rgb_keys(policy, override_keys=rgb_keys),
        "state_key": state_key,
        "infer_input_format": raw_input_format,
        "onnx_graph_input_format": raw_input_format if image_preprocess_in_onnx else preprocessed_input_format,
        "image_preprocess_in_onnx": bool(image_preprocess_in_onnx),
        "n_obs_steps": n_obs_steps,
        "n_image_obs_steps": n_image_obs_steps,
        # input tensors' time dimension: state reads the FIRST n_obs_steps
        # frames, the image branch reads the LAST n_image_obs_steps frames.
        "input_time_steps": input_time_steps,
        "n_action_steps": int(getattr(policy, "n_action_steps", 0)),
        "horizon": int(getattr(policy, "horizon", 0)),
        "action_dim": int(getattr(policy, "action_dim", 0)),
        "action_key": str(getattr(policy, "action_key", "")),
        "supervision_key": str(getattr(policy, "supervision_key", None) or getattr(policy, "action_key", "")),
        "state_dim": int(tuple(obs_meta[state_key]["shape"])[0]) if state_key and state_key in obs_meta else None,
        "num_inference_timesteps": num_inference_timesteps,
        "num_timestep_buckets": int(getattr(policy, "num_timestep_buckets", 0)),
        "noise_trajectory_shape": [1, int(getattr(policy, "horizon", 0)), int(getattr(policy, "action_dim", 0))],
        "static_joint_mask": bool(getattr(policy, "static_joint_mask", torch.zeros(1)).any().item())
        if torch.is_tensor(getattr(policy, "static_joint_mask", None)) else False,
        "preprocess": _build_image_preprocess_metadata(policy),
    }


def _write_onnx_runtime_metadata(onnx_path: str, metadata: Dict[str, Any]) -> str:
    metadata_path = f"{onnx_path}.metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata_path


def _identity_image_processing(images: torch.Tensor) -> torch.Tensor:
    return images


def _disable_scene_encoder_preprocess(policy) -> None:
    scene_encoder = _get_scene_encoder(policy)
    if scene_encoder is not None and hasattr(scene_encoder, "img_processing"):
        scene_encoder.img_processing = _identity_image_processing


def _preprocess_image_batch_for_onnx(
    images: torch.Tensor,
    preprocess: Dict[str, Any],
    device: str,
) -> torch.Tensor:
    images = images.to(device=device, dtype=torch.float32)
    if images.ndim == 3:
        images = images.permute(2, 0, 1) if images.shape[-1] == 3 else images
        images = images.unsqueeze(0)
    elif images.ndim == 4 and images.shape[-1] == 3:
        images = images.permute(0, 3, 1, 2)

    input_range = str(preprocess.get("input_range", "0_1")).lower()
    if input_range in {"0_255", "255", "uint8"}:
        images = images / 255.0
    elif input_range == "auto" and images.max() > 2.0:
        images = images / 255.0
    images = images.clamp(0.0, 1.0)

    patch_size = int(preprocess["patch_size"])
    mode = str(preprocess.get("mode", "crop"))
    _, height, width = [int(dim) for dim in preprocess["raw_image_shape"]]

    interpolate = _resolve_interpolate_for_shape(preprocess.get("interpolate", "bilinear"), width)
    if interpolate is not None:
        interpolate_mode, _ = interpolate
        _, target_height, target_width = [int(dim) for dim in preprocess["resize_shape"]]
        kwargs = (
            {"align_corners": False, "antialias": False}
            if interpolate_mode in ("bilinear", "bicubic")
            else {}
        )
        images = F.interpolate(
            images,
            size=(int(target_height), int(target_width)),
            mode=interpolate_mode,
            **kwargs,
        )
        height, width = target_height, target_width

    if mode == "pad":
        size = round(max(width, height) // patch_size) * patch_size
        h_pad = size - height
        w_pad = size - width
        images = F.pad(
            images,
            (w_pad // 2, w_pad - w_pad // 2, h_pad // 2, h_pad - h_pad // 2),
            mode="constant",
            value=1.0,
        )
    else:
        new_width = (width // patch_size) * patch_size
        new_height = min((height // patch_size) * patch_size, new_width)
        y0 = (height - new_height) // 2
        x0 = (width - new_width) // 2
        images = images[:, :, y0 : y0 + new_height, x0 : x0 + new_width]

    if preprocess.get("do_normalize", False):
        mean = torch.as_tensor(preprocess["image_mean"], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.as_tensor(preprocess["image_std"], device=device, dtype=torch.float32).view(1, 3, 1, 1)
        images = (images - mean) / std

    return images.contiguous()


def _preprocess_rgb_for_onnx_input(
    tensor: torch.Tensor,
    preprocess: Dict[str, Any],
    device: str,
) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.ndim <= 4:
        return _preprocess_image_batch_for_onnx(tensor, preprocess, device=device)

    batch, steps = tensor.shape[:2]
    flat = tensor.reshape(-1, *tensor.shape[2:])
    processed = _preprocess_image_batch_for_onnx(flat, preprocess, device=device)
    return processed.reshape(batch, steps, *processed.shape[1:])


def _get_input_time_steps(policy) -> int:
    # images are read from the LAST n_image_obs_steps frames, state from the
    # FIRST n_obs_steps frames; inputs must cover both.
    n_image_obs_steps = int(getattr(policy, "n_image_obs_steps", None) or policy.n_obs_steps)
    return max(int(policy.n_obs_steps), n_image_obs_steps)


def build_dummy_input(
    policy,
    batch_size: int = 1,
    device: str = "cpu",
    seed: Optional[int] = None,
    image_preprocess_in_onnx: bool = False,
) -> Dict[str, torch.Tensor]:
    obs_meta = policy.config.shape_meta["obs"]
    n_obs_steps = _get_input_time_steps(policy)
    horizon = int(policy.horizon)
    action_dim = int(policy.action_dim)
    state_key = _get_state_key(policy)

    if state_key not in obs_meta or obs_meta[state_key] is None:
        raise ValueError(f"Missing shape_meta for state input: {state_key}")

    dummy_input: Dict[str, torch.Tensor] = {}
    rgb_keys = _get_rgb_keys(policy)
    if "head_cam" not in rgb_keys:
        raise ValueError("RTPolicy ONNX export requires head_cam in obs_encoder rgb inputs.")

    local_seed = seed
    for key in rgb_keys:
        meta = obs_meta.get(key)
        if meta is None:
            raise ValueError(f"Missing shape_meta for rgb input: {key}")
        shape_key = "raw_image_shape" if image_preprocess_in_onnx else "image_shape"
        shape = tuple(_build_image_preprocess_metadata(policy)[shape_key])
        image_builder = _rand_image if image_preprocess_in_onnx else _randn
        dummy_input[key] = image_builder((batch_size, n_obs_steps, *shape), device=device, seed=local_seed)
        if local_seed is not None:
            local_seed += 1

    state_shape = tuple(obs_meta[state_key]["shape"])
    dummy_input[state_key] = _randn(
        (batch_size, n_obs_steps, *state_shape), device=device, seed=local_seed
    )
    if local_seed is not None:
        local_seed += 1
    dummy_input["noise_trajectory"] = _randn(
        (batch_size, horizon, action_dim), device=device, seed=local_seed
    )
    return dummy_input


def build_raw_dummy_input(
    policy,
    batch_size: int = 1,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    obs_meta = policy.config.shape_meta["obs"]
    n_obs_steps = _get_input_time_steps(policy)
    horizon = int(policy.horizon)
    action_dim = int(policy.action_dim)
    state_key = _get_state_key(policy)

    dummy_input: Dict[str, torch.Tensor] = {}
    local_seed = seed
    for key in _get_rgb_keys(policy):
        shape = tuple(obs_meta[key]["shape"])
        dummy_input[key] = _rand_image((batch_size, n_obs_steps, *shape), device=device, seed=local_seed)
        if local_seed is not None:
            local_seed += 1

    state_shape = tuple(obs_meta[state_key]["shape"])
    dummy_input[state_key] = _randn(
        (batch_size, n_obs_steps, *state_shape), device=device, seed=local_seed
    )
    if local_seed is not None:
        local_seed += 1
    dummy_input["noise_trajectory"] = _randn(
        (batch_size, horizon, action_dim), device=device, seed=local_seed
    )
    return dummy_input


def build_dinov3cnn_dummy_input(
    policy,
    batch_size: int = 1,
    device: str = "cpu",
    seed: Optional[int] = None,
    image_preprocess_in_onnx: bool = False,
    rgb_keys: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    obs_meta = policy.config.shape_meta["obs"]
    n_obs_steps = _get_input_time_steps(policy)

    dummy_input: Dict[str, torch.Tensor] = {}
    rgb_keys = _get_rgb_keys(policy, override_keys=rgb_keys)

    local_seed = seed
    shape_key = "raw_image_shape" if image_preprocess_in_onnx else "image_shape"
    image_shape = tuple(_build_image_preprocess_metadata(policy)[shape_key])
    image_builder = _rand_image if image_preprocess_in_onnx else _randn
    for key in rgb_keys:
        meta = obs_meta.get(key)
        if meta is None:
            raise ValueError(f"Missing shape_meta for rgb input: {key}")
        dummy_input[key] = image_builder((batch_size, n_obs_steps, *image_shape), device=device, seed=local_seed)
        if local_seed is not None:
            local_seed += 1

    return dummy_input


def build_dinov3cnn_raw_dummy_input(
    policy,
    batch_size: int = 1,
    device: str = "cpu",
    seed: Optional[int] = None,
    rgb_keys: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    obs_meta = policy.config.shape_meta["obs"]
    n_obs_steps = int(policy.n_obs_steps)

    dummy_input: Dict[str, torch.Tensor] = {}
    local_seed = seed
    for key in _get_rgb_keys(policy, override_keys=rgb_keys):
        shape = tuple(obs_meta[key]["shape"])
        dummy_input[key] = _rand_image((batch_size, n_obs_steps, *shape), device=device, seed=local_seed)
        if local_seed is not None:
            local_seed += 1
    return dummy_input


def _prepare_sample_for_onnx(
    policy,
    sample_input: Dict[str, torch.Tensor],
    device: str,
    export_dtype: torch.dtype = torch.float32,
    image_preprocess_in_onnx: bool = False,
    rgb_keys: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    if image_preprocess_in_onnx:
        return _cast_floating_tensors(sample_input, device=device, dtype=export_dtype)

    preprocess = _build_image_preprocess_metadata(policy)
    onnx_input = dict(sample_input)
    for key in _get_rgb_keys(policy, override_keys=rgb_keys):
        onnx_input[key] = _preprocess_rgb_for_onnx_input(sample_input[key], preprocess, device=device)
    return _cast_floating_tensors(onnx_input, device=device, dtype=export_dtype)


def _build_calibration_input_from_sample(
    policy,
    sample: Dict[str, Any],
    device: str = "cpu",
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    obs = sample["obs"]
    n_obs_steps = _get_input_time_steps(policy)
    horizon = int(policy.horizon)
    action_dim = int(policy.action_dim)
    state_key = _get_state_key(policy)
    rgb_keys = _get_rgb_keys(policy)

    calibration_input: Dict[str, torch.Tensor] = {}
    for key in rgb_keys:
        if key not in obs:
            raise ValueError(f"Calibration sample is missing rgb input: {key}")
        tensor = _to_tensor(obs[key], device=device)
        if tensor.shape[0] < n_obs_steps:
            raise ValueError(
                f"Calibration sample for {key} has only {tensor.shape[0]} steps, expected >= {n_obs_steps}"
            )
        calibration_input[key] = tensor[:n_obs_steps].unsqueeze(0)

    if state_key not in obs:
        raise ValueError(f"Calibration sample is missing state input: {state_key}")
    state_tensor = _to_tensor(obs[state_key], device=device)
    if state_tensor.shape[0] < n_obs_steps:
        raise ValueError(
            f"Calibration sample for {state_key} has only {state_tensor.shape[0]} steps, expected >= {n_obs_steps}"
        )
    calibration_input[state_key] = state_tensor[:n_obs_steps].unsqueeze(0)
    calibration_input["noise_trajectory"] = _randn((1, horizon, action_dim), device=device, seed=seed)
    return calibration_input


def _build_dinov3cnn_calibration_input_from_sample(
    policy,
    sample: Dict[str, Any],
    device: str = "cpu",
    rgb_keys: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    obs = sample["obs"]
    n_obs_steps = _get_input_time_steps(policy)
    rgb_keys = _get_rgb_keys(policy, override_keys=rgb_keys)

    calibration_input: Dict[str, torch.Tensor] = {}
    for key in rgb_keys:
        if key not in obs:
            raise ValueError(f"Calibration sample is missing rgb input: {key}")
        tensor = _to_tensor(obs[key], device=device)
        if tensor.shape[0] < n_obs_steps:
            raise ValueError(
                f"Calibration sample for {key} has only {tensor.shape[0]} steps, expected >= {n_obs_steps}"
            )
        calibration_input[key] = tensor[:n_obs_steps].unsqueeze(0)

    return calibration_input


def _build_calibration_dataset(policy, args):
    train_dataset_cfg = _maybe_get(args, "train_dataset")
    if train_dataset_cfg is None:
        return None, None
    dataset_path = _maybe_get(train_dataset_cfg, "path")
    dataset_task = _maybe_get(train_dataset_cfg, "task")
    if not dataset_path or not dataset_task:
        return None, None

    dataset_cfg = OmegaConf.create(OmegaConf.to_container(train_dataset_cfg, resolve=True))
    if _maybe_get(dataset_cfg, "horizon") is None:
        dataset_cfg.horizon = int(policy.horizon)
    if _maybe_get(dataset_cfg, "pad_before") is None:
        dataset_cfg.pad_before = max(int(policy.n_obs_steps) - 1, 0)
    if _maybe_get(dataset_cfg, "pad_after") is None:
        dataset_cfg.pad_after = max(int(policy.horizon) - int(policy.n_obs_steps), 0)
    if _maybe_get(dataset_cfg, "action_key") is None:
        dataset_cfg.action_key = getattr(policy.config, "action_key", "action")

    dataset = hydra.utils.instantiate(dataset_cfg)
    prefer_validation_dataset = bool(_maybe_get(args, "prefer_validation_dataset", True))
    if prefer_validation_dataset and hasattr(dataset, "get_validation_dataset"):
        val_dataset = dataset.get_validation_dataset()
        if val_dataset is not None and len(val_dataset) > 0:
            return val_dataset, "validation"
    return dataset, "train"


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _resolve_local_lerobot_root(args) -> Optional[str]:
    train_dataset_cfg = _maybe_get(args, "train_dataset")
    if train_dataset_cfg is None:
        return None

    candidates = []
    root = _maybe_get(train_dataset_cfg, "root")
    if root:
        candidates.append(str(root))

    dataset_path = _maybe_get(train_dataset_cfg, "path")
    dataset_task = _maybe_get(train_dataset_cfg, "task")
    if dataset_path and dataset_task:
        candidates.append(os.path.join(str(dataset_path), str(dataset_task)))
    if dataset_path:
        candidates.append(str(dataset_path))

    for candidate in candidates:
        if candidate and os.path.isfile(os.path.join(candidate, "meta", "info.json")):
            return candidate
    return None


def _get_optional_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _load_lerobot_metadata(root: str) -> Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
    info_path = os.path.join(root, "meta", "info.json")
    episodes_path = os.path.join(root, "meta", "episodes.jsonl")
    if not os.path.isfile(info_path) or not os.path.isfile(episodes_path):
        return None

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)
    episodes = _read_jsonl(episodes_path)
    if not episodes:
        return None
    return info, episodes


def _select_calibration_episodes(
    episodes: List[Dict[str, Any]],
    args,
) -> Tuple[List[Dict[str, Any]], str]:
    train_dataset_cfg = _maybe_get(args, "train_dataset")
    seed = int(_maybe_get(args, "seed", 42))
    prefer_validation_dataset = bool(_maybe_get(args, "prefer_validation_dataset", True))
    val_ratio = float(_maybe_get(train_dataset_cfg, "val_ratio", 0.0) or 0.0)

    if prefer_validation_dataset and val_ratio > 0.0 and len(episodes) > 1:
        import numpy as np

        n_val = min(max(1, round(len(episodes) * val_ratio)), len(episodes) - 1)
        rng = np.random.default_rng(seed=seed)
        val_positions = set(rng.choice(len(episodes), size=n_val, replace=False).tolist())
        selected = [episode for idx, episode in enumerate(episodes) if idx in val_positions]
        if selected:
            return selected, "validation"

    max_train_episodes = _get_optional_positive_int(_maybe_get(train_dataset_cfg, "max_train_episodes"))
    if max_train_episodes is not None:
        episodes = episodes[:max_train_episodes]
    return episodes, "train"


def _build_lerobot_episode_path(root: str, info: Dict[str, Any], episode_index: int) -> str:
    data_path = info.get("data_path") or "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    chunks_size = int(info.get("chunks_size") or 1000)
    episode_chunk = episode_index // chunks_size
    rel_path = data_path.format(episode_chunk=episode_chunk, episode_index=episode_index)
    return os.path.join(root, rel_path)


def _decode_lerobot_image(value: Any, root: str, device: str) -> torch.Tensor:
    from io import BytesIO

    import numpy as np
    from PIL import Image

    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes is not None:
            image = Image.open(BytesIO(image_bytes))
        elif image_path:
            resolved_path = image_path if os.path.isabs(image_path) else os.path.join(root, image_path)
            image = Image.open(resolved_path)
        else:
            raise ValueError("Image dict must contain either `bytes` or `path`.")
    else:
        image = value

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB")
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.to(device=device)


def _read_parquet_rows(
    path: str,
    columns: List[str],
    row_indices: List[int],
) -> Dict[int, Dict[str, Any]]:
    import pyarrow.parquet as pq

    wanted = set(int(idx) for idx in row_indices)
    if not wanted:
        return {}

    result: Dict[int, Dict[str, Any]] = {}
    row_offset = 0
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=64, columns=columns):
        arrays = {
            name: batch.column(batch.schema.get_field_index(name))
            for name in columns
        }
        for local_idx in range(batch.num_rows):
            row_idx = row_offset + local_idx
            if row_idx not in wanted:
                continue
            result[row_idx] = {
                name: arrays[name][local_idx].as_py()
                for name in columns
            }
            if len(result) == len(wanted):
                return result
        row_offset += batch.num_rows
    return result


def _make_calibration_specs(
    episodes: List[Dict[str, Any]],
    sample_count: int,
    n_obs_steps: int,
) -> List[Tuple[Dict[str, Any], int]]:
    usable_episodes = [
        episode
        for episode in episodes
        if int(episode.get("length", 0)) >= max(n_obs_steps, 1)
    ]
    if not usable_episodes:
        return []

    specs: List[Tuple[Dict[str, Any], int]] = []
    offsets = {int(episode["episode_index"]): 0 for episode in usable_episodes}
    while len(specs) < sample_count:
        made_progress = False
        for episode in usable_episodes:
            episode_index = int(episode["episode_index"])
            offset = offsets[episode_index]
            if offset + n_obs_steps <= int(episode["length"]):
                specs.append((episode, offset))
                offsets[episode_index] = offset + 1
                made_progress = True
                if len(specs) >= sample_count:
                    break
        if not made_progress:
            break
    return specs


def _build_state_sequence_from_rows(
    rows: Dict[int, Dict[str, Any]],
    start_row: int,
    n_obs_steps: int,
    state_key: str,
    device: str,
) -> torch.Tensor:
    source_key = state_key
    if state_key.startswith("delta_"):
        source_key = state_key[len("delta_"):]

    values = [
        rows[start_row + step][source_key]
        for step in range(n_obs_steps)
    ]
    tensor = _to_tensor(values, device=device)
    if state_key.startswith("delta_") and n_obs_steps > 1:
        delta = tensor - tensor[0:1]
        delta[0:1] = tensor[0:1]
        tensor = delta
    return tensor


def _collect_local_parquet_calibration_inputs(
    policy,
    args,
    device: str,
    include_state_noise: bool,
    rgb_keys: Optional[List[str]] = None,
) -> Optional[Tuple[List[Dict[str, torch.Tensor]], Dict[str, Any]]]:
    root = _resolve_local_lerobot_root(args)
    if root is None:
        return None

    metadata = _load_lerobot_metadata(root)
    if metadata is None:
        return None
    info, episodes = metadata

    features = info.get("features") or {}
    rgb_keys = _get_rgb_keys(policy, override_keys=rgb_keys)
    state_key = _get_state_key(policy)
    source_state_key = state_key[len("delta_"):] if state_key.startswith("delta_") else state_key
    required_columns = list(rgb_keys)
    if include_state_noise:
        required_columns.append(source_state_key)

    missing_columns = [key for key in required_columns if key not in features]
    if missing_columns:
        return None

    if any((features.get(key) or {}).get("dtype") != "image" for key in rgb_keys):
        return None

    num_calibration_samples = max(int(_maybe_get(args, "num_calibration_samples", 1)), 1)
    n_obs_steps = _get_input_time_steps(policy)
    horizon = int(policy.horizon)
    action_dim = int(policy.action_dim)
    base_seed = int(_maybe_get(args, "seed", 42))

    selected_episodes, dataset_split = _select_calibration_episodes(episodes, args)
    specs = _make_calibration_specs(selected_episodes, num_calibration_samples, n_obs_steps)
    if not specs:
        return None

    samples: List[Dict[str, torch.Tensor]] = []
    specs_by_episode: Dict[int, List[Tuple[int, int]]] = {}
    for sample_idx, (episode, start_row) in enumerate(specs):
        episode_index = int(episode["episode_index"])
        specs_by_episode.setdefault(episode_index, []).append((sample_idx, start_row))

    samples_by_idx: Dict[int, Dict[str, torch.Tensor]] = {}
    for episode_index, episode_specs in specs_by_episode.items():
        parquet_path = _build_lerobot_episode_path(root, info, episode_index)
        if not os.path.isfile(parquet_path):
            raise FileNotFoundError(f"Missing calibration parquet file: {parquet_path}")

        wanted_rows = sorted({
            start_row + step
            for _, start_row in episode_specs
            for step in range(n_obs_steps)
        })
        rows = _read_parquet_rows(parquet_path, required_columns, wanted_rows)
        missing_rows = [idx for idx in wanted_rows if idx not in rows]
        if missing_rows:
            raise ValueError(f"Missing rows {missing_rows[:8]} in calibration parquet: {parquet_path}")

        for sample_idx, start_row in episode_specs:
            sample: Dict[str, torch.Tensor] = {}
            for key in rgb_keys:
                frames = [
                    _decode_lerobot_image(rows[start_row + step][key], root=root, device=device)
                    for step in range(n_obs_steps)
                ]
                sample[key] = torch.stack(frames, dim=0).unsqueeze(0)

            if include_state_noise:
                sample[state_key] = _build_state_sequence_from_rows(
                    rows=rows,
                    start_row=start_row,
                    n_obs_steps=n_obs_steps,
                    state_key=state_key,
                    device=device,
                ).unsqueeze(0)
                sample["noise_trajectory"] = _randn(
                    (1, horizon, action_dim),
                    device=device,
                    seed=base_seed + sample_idx,
                )
            samples_by_idx[sample_idx] = sample

    samples = [samples_by_idx[idx] for idx in sorted(samples_by_idx)]
    dataset_size = int(sum(int(episode.get("length", 0)) for episode in selected_episodes))
    logger.info(
        "Collected %s ONNX calibration samples from local parquet dataset split=%s root=%s.",
        len(samples),
        dataset_split,
        root,
    )
    return samples, {
        "source": "local_parquet_dataset",
        "split": dataset_split,
        "dataset_size": dataset_size,
        "num_samples": len(samples),
    }


def collect_calibration_inputs(
    policy,
    args,
    device: str = "cpu",
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, Any]]:
    num_calibration_samples = max(int(_maybe_get(args, "num_calibration_samples", 1)), 1)
    base_seed = int(_maybe_get(args, "seed", 42))

    local_parquet_result = _collect_local_parquet_calibration_inputs(
        policy,
        args,
        device=device,
        include_state_noise=True,
    )
    if local_parquet_result is not None:
        return local_parquet_result

    dataset, dataset_split = _build_calibration_dataset(policy, args)
    if dataset is None or len(dataset) == 0:
        samples = [
            build_raw_dummy_input(policy, batch_size=1, device=device, seed=base_seed + idx)
            for idx in range(num_calibration_samples)
        ]
        return samples, {
            "source": "dummy_input",
            "split": None,
            "dataset_size": 0,
            "num_samples": len(samples),
        }

    sample_count = min(num_calibration_samples, len(dataset))
    samples = [
        _build_calibration_input_from_sample(
            policy,
            dataset[idx],
            device=device,
            seed=base_seed + idx,
        )
        for idx in range(sample_count)
    ]
    return samples, {
        "source": "dataset",
        "split": dataset_split,
        "dataset_size": len(dataset),
        "num_samples": len(samples),
    }


def collect_dinov3cnn_calibration_inputs(
    policy,
    args,
    device: str = "cpu",
    rgb_keys: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, Any]]:
    num_calibration_samples = max(int(_maybe_get(args, "num_calibration_samples", 1)), 1)
    base_seed = int(_maybe_get(args, "seed", 42))

    local_parquet_result = _collect_local_parquet_calibration_inputs(
        policy,
        args,
        device=device,
        include_state_noise=False,
        rgb_keys=rgb_keys,
    )
    if local_parquet_result is not None:
        return local_parquet_result

    dataset, dataset_split = _build_calibration_dataset(policy, args)
    if dataset is None or len(dataset) == 0:
        samples = [
            build_dinov3cnn_raw_dummy_input(
                policy,
                batch_size=1,
                device=device,
                seed=base_seed + idx,
                rgb_keys=rgb_keys,
            )
            for idx in range(num_calibration_samples)
        ]
        return samples, {
            "source": "dummy_input",
            "split": None,
            "dataset_size": 0,
            "num_samples": len(samples),
        }

    sample_count = min(num_calibration_samples, len(dataset))
    samples = [
        _build_dinov3cnn_calibration_input_from_sample(
            policy,
            dataset[idx],
            device=device,
            rgb_keys=rgb_keys,
        )
        for idx in range(sample_count)
    ]
    return samples, {
        "source": "dataset",
        "split": dataset_split,
        "dataset_size": len(dataset),
        "num_samples": len(samples),
    }


def _tensor_dict_to_numpy(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    return {key: value.detach().cpu().numpy() for key, value in tensor_dict.items()}


def _resolve_calibration_data_dir(
    checkpoint_dir: str,
    calibration_data_dir: Optional[str] = None,
) -> str:
    if not calibration_data_dir:
        return os.path.join(checkpoint_dir, "onnx_calibration")
    if os.path.isabs(calibration_data_dir):
        return calibration_data_dir
    return os.path.join(checkpoint_dir, calibration_data_dir)


def _save_numpy_dir(output_dir: str, arrays: Dict[str, Any]) -> None:
    import numpy as np

    os.makedirs(output_dir, exist_ok=True)
    for key, value in arrays.items():
        np.save(os.path.join(output_dir, f"{key}.npy"), value)


def save_calibration_data(
    calibration_inputs: List[Dict[str, torch.Tensor]],
    checkpoint_dir: str,
    calibration_data_dir: Optional[str] = None,
    verification_outputs: Optional[Dict[int, Dict[str, Dict[str, Any]]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    save_dir = _resolve_calibration_data_dir(checkpoint_dir, calibration_data_dir)
    os.makedirs(save_dir, exist_ok=True)

    for entry in os.listdir(save_dir):
        entry_path = os.path.join(save_dir, entry)
        if entry.startswith('sample_') and os.path.isdir(entry_path):
            shutil.rmtree(entry_path)

    for sample_idx, sample in enumerate(calibration_inputs):
        sample_dir = os.path.join(save_dir, f"sample_{sample_idx:03d}")
        _save_numpy_dir(os.path.join(sample_dir, "inputs"), _tensor_dict_to_numpy(sample))

        if verification_outputs and sample_idx in verification_outputs:
            result = verification_outputs[sample_idx]
            _save_numpy_dir(os.path.join(sample_dir, "torch_outputs"), result["torch"])
            _save_numpy_dir(os.path.join(sample_dir, "onnx_outputs"), result["onnx"])
            if "metrics" in result:
                metrics_path = os.path.join(sample_dir, "verification_metrics.json")
                with open(metrics_path, "w", encoding="utf-8") as f:
                    json.dump(result["metrics"], f, indent=2, ensure_ascii=False)

    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata or {}, f, indent=2, ensure_ascii=False)
    return save_dir


class DINOv3CNNExportWrapper(torch.nn.Module):
    def __init__(
        self,
        policy,
        image_preprocess_in_onnx: bool = False,
        rgb_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        obs_encoder = getattr(policy, "obs_encoder", None)
        scene_encoder = getattr(obs_encoder, "scene_encoder", None)
        if scene_encoder is None or not isinstance(scene_encoder, torch.nn.Module):
            raise ValueError("Missing policy.obs_encoder.scene_encoder for export_target='dinov3cnn'.")
        emvis_cfg = _maybe_get(getattr(policy.config, "obs_encoder", None), "emvis_config", None)
        backbone_target = str(_maybe_get(emvis_cfg, "backbone_target", "")).lower()
        if "dinov3" not in backbone_target or "cnn" not in backbone_target:
            raise ValueError(
                "export_target='dinov3cnn' requires a dinov3cnn checkpoint/config, "
                f"but got backbone_target={backbone_target!r}."
            )

        self.scene_encoder = scene_encoder
        available_rgb_keys = list(_get_rgb_keys(policy))
        self.rgb_keys = _get_rgb_keys(policy, override_keys=rgb_keys)
        missing_rgb_keys = [key for key in self.rgb_keys if key not in available_rgb_keys]
        if missing_rgb_keys:
            raise ValueError(
                "DINOv3CNN rgb key override contains keys that are not in the checkpoint "
                f"rgb inputs: {missing_rgb_keys}. Available rgb inputs: {available_rgb_keys}."
            )
        if not self.rgb_keys:
            raise ValueError("DINOv3CNN export requires at least one rgb input.")
        self.n_obs_steps = int(
            getattr(policy, "n_image_obs_steps", None) or policy.n_obs_steps
        )
        self.image_preprocess_in_onnx = bool(image_preprocess_in_onnx)
        self.image_preprocess = (
            _build_image_preprocess_metadata(policy) if self.image_preprocess_in_onnx else None
        )

    def _maybe_preprocess_images(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.image_preprocess_in_onnx:
            return obs_dict

        dtype = next(self.scene_encoder.parameters()).dtype
        processed = dict(obs_dict)
        for key in self.rgb_keys:
            processed[key] = _preprocess_rgb_for_onnx_input(
                obs_dict[key],
                self.image_preprocess,
                device=str(obs_dict[key].device),
            ).to(dtype=dtype)
        return processed

    def forward(
        self,
        head_cam: torch.Tensor,
        front_cam: Optional[torch.Tensor] = None,
        left_cam: Optional[torch.Tensor] = None,
        right_cam: Optional[torch.Tensor] = None,
        twist_cam: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_dict = {"head_cam": head_cam}
        if front_cam is not None:
            obs_dict["front_cam"] = front_cam
        if left_cam is not None:
            obs_dict["left_cam"] = left_cam
        if right_cam is not None:
            obs_dict["right_cam"] = right_cam
        if twist_cam is not None:
            obs_dict["twist_cam"] = twist_cam

        missing_rgb = [key for key in self.rgb_keys if key not in obs_dict]
        if missing_rgb:
            raise ValueError(
                f"Missing required rgb inputs for export_target='dinov3cnn': {missing_rgb}"
            )

        obs_dict = self._maybe_preprocess_images(obs_dict)
        sample = obs_dict[self.rgb_keys[0]][:, -self.n_obs_steps :, ...]
        batch_size, time_steps = sample.shape[:2]
        rgb_image = torch.cat(
            [
                obs_dict[key][:, :time_steps, ...].reshape(-1, *obs_dict[key].shape[2:]).unsqueeze(1)
                for key in self.rgb_keys
            ],
            dim=1,
        )

        token_dict = self.scene_encoder.backbone(rgb_image, V_fake=None)
        image_tokens = token_dict["image_tokens"].view(
            batch_size, time_steps, *token_dict["image_tokens"].shape[1:]
        )
        image_tokens_pos = token_dict["image_tokens_pos"].view(
            batch_size, time_steps, *token_dict["image_tokens_pos"].shape[1:]
        )
        return image_tokens, image_tokens_pos


class VisualEncoderExportWrapper(torch.nn.Module):
    def __init__(
        self,
        policy,
        image_preprocess_in_onnx: bool = False,
        rgb_keys: Optional[List[str]] = None,
    ):
        super().__init__()
        obs_encoder = getattr(policy, "obs_encoder", None)
        scene_encoder = getattr(obs_encoder, "scene_encoder", None)
        if scene_encoder is None or not isinstance(scene_encoder, torch.nn.Module):
            raise ValueError("Missing policy.obs_encoder.scene_encoder for export_target='visual_encoder'.")

        self.scene_encoder = scene_encoder
        available_rgb_keys = list(_get_rgb_keys(policy))
        self.rgb_keys = _get_rgb_keys(policy, override_keys=rgb_keys)
        missing_rgb_keys = [key for key in self.rgb_keys if key not in available_rgb_keys]
        if missing_rgb_keys:
            raise ValueError(
                "Visual encoder rgb key override contains keys that are not in the checkpoint "
                f"rgb inputs: {missing_rgb_keys}. Available rgb inputs: {available_rgb_keys}."
            )
        if not self.rgb_keys:
            raise ValueError("Visual encoder export requires at least one rgb input.")
        self.n_obs_steps = int(
            getattr(policy, "n_image_obs_steps", None) or policy.n_obs_steps
        )
        self.image_preprocess_in_onnx = bool(image_preprocess_in_onnx)
        self.image_preprocess = (
            _build_image_preprocess_metadata(policy) if self.image_preprocess_in_onnx else None
        )

    def _maybe_preprocess_images(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not self.image_preprocess_in_onnx:
            return obs_dict

        dtype = next(self.scene_encoder.parameters()).dtype
        processed = dict(obs_dict)
        for key in self.rgb_keys:
            processed[key] = _preprocess_rgb_for_onnx_input(
                obs_dict[key],
                self.image_preprocess,
                device=str(obs_dict[key].device),
            ).to(dtype=dtype)
        return processed

    def forward(
        self,
        head_cam: torch.Tensor,
        front_cam: Optional[torch.Tensor] = None,
        left_cam: Optional[torch.Tensor] = None,
        right_cam: Optional[torch.Tensor] = None,
        twist_cam: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        obs_dict = {"head_cam": head_cam}
        if front_cam is not None:
            obs_dict["front_cam"] = front_cam
        if left_cam is not None:
            obs_dict["left_cam"] = left_cam
        if right_cam is not None:
            obs_dict["right_cam"] = right_cam
        if twist_cam is not None:
            obs_dict["twist_cam"] = twist_cam

        missing_rgb = [key for key in self.rgb_keys if key not in obs_dict]
        if missing_rgb:
            raise ValueError(
                f"Missing required rgb inputs for export_target='visual_encoder': {missing_rgb}"
            )

        obs_dict = self._maybe_preprocess_images(obs_dict)
        sample = obs_dict[self.rgb_keys[0]][:, -self.n_obs_steps :, ...]
        batch_size, time_steps = sample.shape[:2]
        rgb_image = torch.cat(
            [
                obs_dict[key][:, :time_steps, ...].reshape(-1, *obs_dict[key].shape[2:]).unsqueeze(1)
                for key in self.rgb_keys
            ],
            dim=1,
        )

        scene_features = self.scene_encoder(rgb_image, batch_size=batch_size)
        return scene_features.reshape(batch_size, time_steps, -1, scene_features.shape[-1])


class PolicyExportWrapper(torch.nn.Module):
    def __init__(self, policy, image_preprocess_in_onnx: bool = False):
        super().__init__()
        self.policy = policy
        self.rgb_keys = list(_get_rgb_keys(policy))
        self.state_key = _get_state_key(policy)
        # temp-policy semantics: images use the LAST n_image_obs_steps frames,
        # state uses the FIRST n_obs_steps frames.
        self.n_image_obs_steps = int(
            getattr(policy, "n_image_obs_steps", None) or policy.n_obs_steps
        )
        self.supervision_key = getattr(policy, "supervision_key", None) or policy.action_key
        self.image_preprocess_in_onnx = bool(image_preprocess_in_onnx)
        self.image_preprocess = (
            _build_image_preprocess_metadata(policy) if self.image_preprocess_in_onnx else None
        )

    def _build_obs_dict(
        self,
        head_cam: Optional[torch.Tensor] = None,
        front_cam: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        agent_pos: Optional[torch.Tensor] = None,
        left_cam: Optional[torch.Tensor] = None,
        right_cam: Optional[torch.Tensor] = None,
        twist_cam: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        values = {
            "head_cam": head_cam,
            "front_cam": front_cam,
            "left_cam": left_cam,
            "right_cam": right_cam,
            "twist_cam": twist_cam,
            "state": state,
            "agent_pos": agent_pos,
        }
        obs_dict = {
            key: values[key]
            for key in self.rgb_keys
            if key in values and values[key] is not None
        }
        state_value = values.get(self.state_key)
        if state_value is None:
            raise ValueError(f"Missing required state input for state_key={self.state_key}")
        obs_dict[self.state_key] = state_value

        missing_rgb = [key for key in self.rgb_keys if key not in obs_dict]
        if missing_rgb:
            raise ValueError(f"Missing required rgb inputs for policy ONNX export: {missing_rgb}")
        return obs_dict

    def _maybe_preprocess_images(
        self,
        obs_dict: Dict[str, torch.Tensor],
        device: str,
    ) -> Dict[str, torch.Tensor]:
        if not self.image_preprocess_in_onnx:
            return obs_dict

        processed = dict(obs_dict)
        for key in self.rgb_keys:
            processed[key] = _preprocess_rgb_for_onnx_input(
                obs_dict[key],
                self.image_preprocess,
                device=device,
            )
        return processed

    def forward(
        self,
        head_cam: torch.Tensor,
        noise_trajectory: torch.Tensor,
        front_cam: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        agent_pos: Optional[torch.Tensor] = None,
        left_cam: Optional[torch.Tensor] = None,
        right_cam: Optional[torch.Tensor] = None,
        twist_cam: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        policy = self.policy
        obs_dict = self._build_obs_dict(
            head_cam=head_cam,
            front_cam=front_cam,
            state=state,
            agent_pos=agent_pos,
            left_cam=left_cam,
            right_cam=right_cam,
            twist_cam=twist_cam,
        )

        device = policy.device
        dtype = policy.dtype
        obs_dict = self._maybe_preprocess_images(obs_dict, device=device)
        nobs = {key: value.to(device=device, dtype=dtype) for key, value in obs_dict.items()}
        filtered_obs_dict = {
            key: value
            for key, value in obs_dict.items()
            if key in getattr(policy, "action_keys", ())
        }
        nobs.update(policy.normalizer.normalize(filtered_obs_dict))
        if policy.action_key in nobs:
            nobs[policy.action_key] = policy._zero_static_joints(nobs[policy.action_key])

        value = next(iter(nobs.values()))
        batch_size = value.shape[0]
        n_obs_steps = int(policy.n_obs_steps)
        horizon = int(policy.horizon)
        action_dim = int(policy.action_dim)

        scene_cond = policy.encode_image_conditions(nobs, batch_size)
        state_traj = torch.zeros(
            size=(batch_size, horizon, action_dim),
            device=device,
            dtype=dtype,
        )
        state_traj[:, :n_obs_steps, :] = nobs[policy.action_key][:, :n_obs_steps].view(
            batch_size, n_obs_steps, -1
        )
        action_mask = policy.action_mask.expand(batch_size, -1, -1).to(device=device)

        noisy_action = noise_trajectory.to(device=device, dtype=dtype)
        noisy_action = policy._zero_static_joints(noisy_action)
        state_mask = ~action_mask
        num_steps = int(policy.num_inference_timesteps)
        dt = 1.0 / num_steps
        for t in range(num_steps):
            t_discretized = int(t / float(num_steps) * policy.num_timestep_buckets)
            timesteps = torch.full(
                size=(batch_size,),
                fill_value=t_discretized,
                device=device,
            )
            noisy_action[state_mask] = state_traj[state_mask]
            noisy_action = policy._zero_static_joints(noisy_action)
            state_action_traj = policy.state_adaptor(noisy_action)
            pred_velocity = policy.model(state_action_traj, timesteps, scene_cond)
            pred_velocity = policy._zero_static_joints(pred_velocity)
            noisy_action = noisy_action + dt * pred_velocity
            noisy_action = policy._zero_static_joints(noisy_action)

        noisy_action[state_mask] = state_traj[state_mask]
        action_pred = policy.normalizer[self.supervision_key].unnormalize(noisy_action[..., :action_dim])
        start = n_obs_steps
        end = start + int(policy.n_action_steps)
        action = action_pred[:, start:end]
        if "delta" in policy.action_key:
            key = policy.action_key.replace("delta_", "")
            action = policy._zero_static_joints(action)
            action = action + obs_dict[key][:, 0:1, ...]
        else:
            action = policy._hold_static_joints(action, obs_dict[policy.action_key])
        return action, action_pred


def verify_module_onnx_export(
    onnx_path: str,
    module,
    sample_input: Dict[str, torch.Tensor],
    output_names: List[str],
    rtol: float = 1e-4,
    atol: float = 1e-4,
    ort_session=None,
    fail_on_mismatch: bool = True,
) -> Dict[str, Dict[str, Any]]:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("verify requested but onnxruntime/numpy is unavailable") from exc

    with torch.no_grad():
        torch_output_values = module(**sample_input)

    if isinstance(torch_output_values, torch.Tensor):
        torch_output_values = (torch_output_values,)
    elif not isinstance(torch_output_values, (tuple, list)):
        raise TypeError(
            f"Expected module outputs to be a tensor or tuple/list of tensors, got {type(torch_output_values)!r}."
        )

    if len(torch_output_values) != len(output_names):
        raise ValueError(
            f"Output count mismatch for verification: expected {len(output_names)} values for "
            f"{output_names}, got {len(torch_output_values)}."
        )

    torch_outputs = {
        name: value.detach().cpu().numpy()
        for name, value in zip(output_names, torch_output_values)
    }

    if ort_session is None:
        ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {}
    missing_inputs = []
    for model_input in ort_session.get_inputs():
        name = model_input.name
        if name not in sample_input:
            missing_inputs.append(name)
            continue
        ort_inputs[name] = sample_input[name].detach().cpu().numpy()
    if missing_inputs:
        raise ValueError(
            f"sample_input is missing ONNX model inputs: {missing_inputs}. "
            f"Provided keys: {sorted(sample_input.keys())}"
        )

    ort_output_values = ort_session.run(output_names, ort_inputs)
    ort_outputs = {
        name: value
        for name, value in zip(output_names, ort_output_values)
    }

    metrics = {}
    for name in output_names:
        torch_value = torch_outputs[name].astype(np.float32, copy=False)
        ort_value = ort_outputs[name].astype(np.float32, copy=False)
        abs_diff = np.abs(ort_value - torch_value)
        rel_diff = abs_diff / np.maximum(np.abs(torch_value), 1e-12)
        is_close = np.isclose(ort_value, torch_value, rtol=rtol, atol=atol)
        output_metrics = {
            "shape": list(torch_outputs[name].shape),
            "torch_dtype": str(torch_outputs[name].dtype),
            "onnx_dtype": str(ort_outputs[name].dtype),
            "rtol": float(rtol),
            "atol": float(atol),
            "allclose": bool(np.all(is_close)),
            "max_abs_diff": float(abs_diff.max()) if abs_diff.size else 0.0,
            "mean_abs_diff": float(abs_diff.mean()) if abs_diff.size else 0.0,
            "max_rel_diff": float(rel_diff.max()) if rel_diff.size else 0.0,
            "mean_rel_diff": float(rel_diff.mean()) if rel_diff.size else 0.0,
            "mismatch_count": int(np.count_nonzero(~is_close)),
            "numel": int(abs_diff.size),
        }
        metrics[name] = output_metrics
        logger.info(
            "ONNX verification output=%s allclose=%s max_abs_diff=%.6g mean_abs_diff=%.6g "
            "max_rel_diff=%.6g mismatch_count=%s/%s rtol=%s atol=%s",
            name,
            output_metrics["allclose"],
            output_metrics["max_abs_diff"],
            output_metrics["mean_abs_diff"],
            output_metrics["max_rel_diff"],
            output_metrics["mismatch_count"],
            output_metrics["numel"],
            rtol,
            atol,
        )
        if fail_on_mismatch:
            np.testing.assert_allclose(ort_outputs[name], torch_outputs[name], rtol=rtol, atol=atol)
    return {"torch": torch_outputs, "onnx": ort_outputs, "metrics": metrics}


def verify_onnx_export(
    onnx_path: str,
    module,
    sample_input: Dict[str, torch.Tensor],
    rtol: float = 1e-4,
    atol: float = 1e-4,
    ort_session=None,
    fail_on_mismatch: bool = True,
) -> Dict[str, Dict[str, Any]]:
    return verify_module_onnx_export(
        onnx_path=onnx_path,
        module=module,
        sample_input=sample_input,
        output_names=["action", "action_pred"],
        rtol=rtol,
        atol=atol,
        ort_session=ort_session,
        fail_on_mismatch=fail_on_mismatch,
    )


def export_policy_to_onnx(
    checkpoint_dir: str,
    onnx_path: str,
    args=None,
    device: str = "cpu",
    batch_size: int = 1,
    opset_version: int = 19,
    verify: bool = False,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    save_calibration_data_flag: bool = True,
    calibration_data_dir: Optional[str] = None,
    use_external_data: bool = True,
    external_data_filename: Optional[str] = None,
    external_data_size_threshold: int = 1024,
    export_modules_as_functions: Any = "diffusion",
    dynamic_batch: bool = True,
    optimize: bool = False,
    report: bool = False,
    save_netron_model: bool = True,
    netron_model_path: Optional[str] = None,
    export_dtype: Any = torch.float32,
    fail_on_verify_mismatch: bool = True,
    image_preprocess_in_onnx: bool = False,
):
    patch_for_onnx_export()
    export_dtype = _resolve_export_dtype(export_dtype)
    export_dtype_name = _export_dtype_name(export_dtype)
    if export_dtype == torch.float16 and torch.device(device).type == "cpu":
        logger.warning(
            "export_dtype=fp16 with device=cpu may fail because some PyTorch/ONNXRuntime CPU ops "
            "do not support float16. Prefer device=cuda for fp16 export."
        )

    policy = _load_policy_for_onnx(checkpoint_dir, args=args, device=device)
    policy = policy.to(device=device, dtype=export_dtype)

    # Optional export-time override of the FM inference step count. The diffusion
    # loop is unrolled into the ONNX graph for `policy.num_inference_timesteps`
    # iterations, so this bakes the chosen step count into the exported model.
    override_inference_steps = _get_optional_positive_int(
        getattr(args, "num_inference_timesteps", None)
    )
    if override_inference_steps is not None:
        original_steps = getattr(policy, "num_inference_timesteps", None)
        logger.info(
            "Overriding policy.num_inference_timesteps for ONNX export: %s -> %s.",
            original_steps,
            override_inference_steps,
        )
        policy.num_inference_timesteps = override_inference_steps

    _disable_export_only_features(policy, export_dtype=export_dtype)
    _disable_scene_encoder_preprocess(policy)
    policy.eval()
    export_module = PolicyExportWrapper(
        policy,
        image_preprocess_in_onnx=image_preprocess_in_onnx,
    ).to(device=device, dtype=export_dtype)
    export_module.eval()
    resolved_export_modules_as_functions = _resolve_export_modules_as_functions(
        policy, export_modules_as_functions
    )

    dummy_input = build_dummy_input(
        policy,
        batch_size=batch_size,
        device=device,
        image_preprocess_in_onnx=image_preprocess_in_onnx,
    )
    input_names = _ordered_input_names_for_export(export_module, dummy_input)
    ordered_dummy_input = _cast_floating_tensors(
        {name: dummy_input[name] for name in input_names},
        device=device,
        dtype=export_dtype,
    )
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {name: {0: "batch_size"} for name in input_names}
        dynamic_axes["action"] = {0: "batch_size"}
        dynamic_axes["action_pred"] = {0: "batch_size"}

    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)

    logger.info(
        "Using legacy torch.onnx exporter with dynamo=False, export_dtype=%s, image_preprocess_in_onnx=%s, and export_modules_as_functions=%s.",
        export_dtype_name,
        bool(image_preprocess_in_onnx),
        True
        if resolved_export_modules_as_functions is True
        else [
            f"{module_type.__module__}.{module_type.__name__}"
            for module_type in resolved_export_modules_as_functions
        ]
        if resolved_export_modules_as_functions
        else False,
    )
    if optimize or report:
        logger.warning(
            "Ignoring optimize=%s and report=%s because RTPolicy ONNX export is forced to the legacy exporter (dynamo=False) to avoid onnxscript custom domains.",
            bool(optimize),
            bool(report),
        )

    export_kwargs = dict(
        opset_version=opset_version,
        input_names=input_names,
        output_names=["action", "action_pred"],
        do_constant_folding=_should_do_constant_folding(device),
        export_params=True,
        keep_initializers_as_inputs=False,
        dynamo=False,
        verbose=False,
        export_modules_as_functions=resolved_export_modules_as_functions,
    )
    if dynamic_axes is not None:
        export_kwargs["dynamic_axes"] = dynamic_axes
    else:
        logger.info("Exporting ONNX with static batch dimension fixed to %s.", batch_size)

    with torch.no_grad():
        torch.onnx.export(
            export_module,
            kwargs=ordered_dummy_input,
            f=onnx_path,
            **export_kwargs,
        )

    onnx_data_path = None
    if use_external_data:
        onnx_data_path = _export_external_data_artifact(
            onnx_path=onnx_path,
            external_data_filename=external_data_filename,
            external_data_size_threshold=external_data_size_threshold,
        )
        logger.info("ONNX external data saved: %s", onnx_data_path)

    graph_summary = _summarize_onnx_graph(onnx_path)
    if graph_summary is not None:
        logger.info("ONNX graph summary: %s", graph_summary)
    saved_netron_model_path = None
    if save_netron_model:
        saved_netron_model_path = _write_netron_view_model(
            onnx_path=onnx_path,
            netron_model_path=netron_model_path,
        )
        if saved_netron_model_path is not None:
            logger.info("Netron view ONNX saved: %s", saved_netron_model_path)
    metadata_path = _write_onnx_runtime_metadata(
        onnx_path,
        _build_onnx_runtime_metadata(
            policy=policy,
            input_names=input_names,
            output_names=["action", "action_pred"],
            export_target="policy",
            export_dtype=export_dtype,
            image_preprocess_in_onnx=image_preprocess_in_onnx,
        ),
    )

    calibration_inputs: List[Dict[str, torch.Tensor]] = []
    calibration_info = {
        "source": "disabled",
        "split": None,
        "dataset_size": 0,
        "num_samples": 0,
    }
    if args is not None and save_calibration_data_flag:
        calibration_inputs, calibration_info = collect_calibration_inputs(policy, args, device=device)

    verification_outputs = {}
    if verify:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("verify requested but onnxruntime is unavailable") from exc

        samples_to_verify = calibration_inputs or [
            build_raw_dummy_input(policy, batch_size=1, device=device)
        ]
        ort_session = _create_onnxruntime_session_for_verify(
            onnx_path=onnx_path,
            device=device,
            export_dtype=export_dtype,
        )
        for idx, sample_input in enumerate(samples_to_verify):
            onnx_sample_input = _prepare_sample_for_onnx(
                policy,
                sample_input,
                device=device,
                export_dtype=export_dtype,
                image_preprocess_in_onnx=image_preprocess_in_onnx,
            )
            verification_outputs[idx] = verify_onnx_export(
                onnx_path=onnx_path,
                module=export_module,
                sample_input=onnx_sample_input,
                rtol=rtol,
                atol=atol,
                ort_session=ort_session,
                fail_on_mismatch=fail_on_verify_mismatch,
            )

    saved_calibration_dir = None
    if save_calibration_data_flag and calibration_inputs:
        metadata = {
            "checkpoint_dir": checkpoint_dir,
            "onnx_path": os.path.abspath(onnx_path),
            "onnx_data_path": onnx_data_path,
            "onnx_metadata_path": metadata_path,
            "netron_model_path": saved_netron_model_path,
            "export_dtype": export_dtype_name,
            "onnx_input_dtype": export_dtype_name,
            "input_names": input_names,
            "state_key": _get_state_key(policy),
            "rgb_keys": _get_rgb_keys(policy),
            "preprocess": _build_image_preprocess_metadata(policy),
            "image_preprocess_in_onnx": bool(image_preprocess_in_onnx),
            "calibration_input_format": "raw_images_state_noise",
            "onnx_graph_input_format": (
                "raw_images_state_noise"
                if image_preprocess_in_onnx
                else "preprocessed_images_state_noise"
            ),
            "num_calibration_samples": len(calibration_inputs),
            "verification_enabled": verify,
            "fail_on_verify_mismatch": bool(fail_on_verify_mismatch),
            "calibration_source": calibration_info["source"],
            "calibration_split": calibration_info["split"],
            "dataset_size": calibration_info["dataset_size"],
        }
        saved_calibration_dir = save_calibration_data(
            calibration_inputs=calibration_inputs,
            checkpoint_dir=checkpoint_dir,
            calibration_data_dir=calibration_data_dir,
            verification_outputs=verification_outputs if verify else None,
            metadata=metadata,
        )

    return {
        "checkpoint_dir": checkpoint_dir,
        "onnx_path": onnx_path,
        "onnx_data_path": onnx_data_path,
        "onnx_metadata_path": metadata_path,
        "netron_model_path": saved_netron_model_path,
        "export_dtype": export_dtype_name,
        "graph_summary": graph_summary,
        "input_names": input_names,
        "state_key": _get_state_key(policy),
        "rgb_keys": _get_rgb_keys(policy),
        "image_preprocess_in_onnx": bool(image_preprocess_in_onnx),
        "calibration_data_dir": saved_calibration_dir,
        "num_calibration_samples": calibration_info["num_samples"],
        "verification_samples": len(verification_outputs),
    }


def export_dinov3cnn_to_onnx(
    checkpoint_dir: str,
    onnx_path: str,
    args=None,
    device: str = "cpu",
    batch_size: int = 1,
    opset_version: int = 19,
    verify: bool = False,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    save_calibration_data_flag: bool = True,
    calibration_data_dir: Optional[str] = None,
    use_external_data: bool = True,
    external_data_filename: Optional[str] = None,
    external_data_size_threshold: int = 1024,
    export_modules_as_functions: Any = False,
    dynamic_batch: bool = True,
    optimize: bool = False,
    report: bool = False,
    save_netron_model: bool = True,
    netron_model_path: Optional[str] = None,
    export_dtype: Any = torch.float32,
    fail_on_verify_mismatch: bool = True,
    image_preprocess_in_onnx: bool = False,
):
    patch_for_onnx_export()
    export_dtype = _resolve_export_dtype(export_dtype)
    export_dtype_name = _export_dtype_name(export_dtype)
    if export_dtype == torch.float16 and torch.device(device).type == "cpu":
        logger.warning(
            "export_dtype=fp16 with device=cpu may fail because some PyTorch/ONNXRuntime CPU ops "
            "do not support float16. Prefer device=cuda for fp16 export."
        )

    policy = _load_policy_for_onnx(checkpoint_dir, args=args, device=device)
    policy = policy.to(device=device, dtype=export_dtype)
    _disable_export_only_features(policy, export_dtype=export_dtype)
    _disable_scene_encoder_preprocess(policy)
    policy.eval()
    dinov3cnn_rgb_keys = _resolve_dinov3cnn_rgb_keys(args)
    if dinov3cnn_rgb_keys is not None:
        logger.info(
            "Exporting DINOv3CNN with rgb key override: %s",
            dinov3cnn_rgb_keys,
        )

    export_module = DINOv3CNNExportWrapper(
        policy,
        image_preprocess_in_onnx=image_preprocess_in_onnx,
        rgb_keys=dinov3cnn_rgb_keys,
    )
    export_module = export_module.to(device=device, dtype=export_dtype)
    export_module.eval()

    dummy_input = build_dinov3cnn_dummy_input(
        policy,
        batch_size=batch_size,
        device=device,
        image_preprocess_in_onnx=image_preprocess_in_onnx,
        rgb_keys=dinov3cnn_rgb_keys,
    )
    input_names = _ordered_input_names_for_export(export_module, dummy_input)
    ordered_dummy_input = _cast_floating_tensors(
        {name: dummy_input[name] for name in input_names},
        device=device,
        dtype=export_dtype,
    )
    output_names = ["image_tokens", "image_tokens_pos"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {name: {0: "batch_size"} for name in input_names}
        dynamic_axes["image_tokens"] = {0: "batch_size"}
        dynamic_axes["image_tokens_pos"] = {0: "batch_size"}

    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)

    logger.info(
        "Using legacy torch.onnx exporter with dynamo=False, export_dtype=%s, and image_preprocess_in_onnx=%s for export_target=dinov3cnn.",
        export_dtype_name,
        bool(image_preprocess_in_onnx),
    )
    if _resolve_export_modules_as_functions(policy, export_modules_as_functions):
        logger.warning(
            "Ignoring export_modules_as_functions=%s for export_target=dinov3cnn.",
            export_modules_as_functions,
        )
    if optimize or report:
        logger.warning(
            "Ignoring optimize=%s and report=%s because RTPolicy ONNX export is forced to the legacy exporter (dynamo=False) to avoid onnxscript custom domains.",
            bool(optimize),
            bool(report),
        )

    export_kwargs = dict(
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=_should_do_constant_folding(device),
        export_params=True,
        keep_initializers_as_inputs=False,
        dynamo=False,
        verbose=False,
    )
    if dynamic_axes is not None:
        export_kwargs["dynamic_axes"] = dynamic_axes
    else:
        logger.info("Exporting ONNX with static batch dimension fixed to %s.", batch_size)

    with torch.no_grad():
        torch.onnx.export(
            export_module,
            kwargs=ordered_dummy_input,
            f=onnx_path,
            **export_kwargs,
        )

    onnx_data_path = None
    if use_external_data:
        onnx_data_path = _export_external_data_artifact(
            onnx_path=onnx_path,
            external_data_filename=external_data_filename,
            external_data_size_threshold=external_data_size_threshold,
        )
        logger.info("ONNX external data saved: %s", onnx_data_path)

    graph_summary = _summarize_onnx_graph(onnx_path)
    if graph_summary is not None:
        logger.info("ONNX graph summary: %s", graph_summary)
    saved_netron_model_path = None
    if save_netron_model:
        saved_netron_model_path = _write_netron_view_model(
            onnx_path=onnx_path,
            netron_model_path=netron_model_path,
        )
        if saved_netron_model_path is not None:
            logger.info("Netron view ONNX saved: %s", saved_netron_model_path)
    metadata_path = _write_onnx_runtime_metadata(
        onnx_path,
        _build_onnx_runtime_metadata(
            policy=policy,
            input_names=input_names,
            output_names=output_names,
            export_target="dinov3cnn",
            export_dtype=export_dtype,
            image_preprocess_in_onnx=image_preprocess_in_onnx,
            rgb_keys=export_module.rgb_keys,
        ),
    )

    calibration_inputs: List[Dict[str, torch.Tensor]] = []
    calibration_info = {
        "source": "disabled",
        "split": None,
        "dataset_size": 0,
        "num_samples": 0,
    }
    if args is not None and save_calibration_data_flag:
        calibration_inputs, calibration_info = collect_dinov3cnn_calibration_inputs(
            policy,
            args,
            device=device,
            rgb_keys=export_module.rgb_keys,
        )

    verification_outputs = {}
    if verify:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("verify requested but onnxruntime is unavailable") from exc

        samples_to_verify = calibration_inputs or [
            build_dinov3cnn_raw_dummy_input(
                policy,
                batch_size=1,
                device=device,
                rgb_keys=export_module.rgb_keys,
            )
        ]
        ort_session = _create_onnxruntime_session_for_verify(
            onnx_path=onnx_path,
            device=device,
            export_dtype=export_dtype,
        )
        for idx, sample_input in enumerate(samples_to_verify):
            onnx_sample_input = _prepare_sample_for_onnx(
                policy,
                sample_input,
                device=device,
                export_dtype=export_dtype,
                image_preprocess_in_onnx=image_preprocess_in_onnx,
                rgb_keys=export_module.rgb_keys,
            )
            verification_outputs[idx] = verify_module_onnx_export(
                onnx_path=onnx_path,
                module=export_module,
                sample_input=onnx_sample_input,
                output_names=output_names,
                rtol=rtol,
                atol=atol,
                ort_session=ort_session,
                fail_on_mismatch=fail_on_verify_mismatch,
            )

    saved_calibration_dir = None
    if save_calibration_data_flag and calibration_inputs:
        metadata = {
            "checkpoint_dir": checkpoint_dir,
            "onnx_path": os.path.abspath(onnx_path),
            "onnx_data_path": onnx_data_path,
            "onnx_metadata_path": metadata_path,
            "netron_model_path": saved_netron_model_path,
            "export_dtype": export_dtype_name,
            "onnx_input_dtype": export_dtype_name,
            "input_names": input_names,
            "output_names": output_names,
            "rgb_keys": list(export_module.rgb_keys),
            "preprocess": _build_image_preprocess_metadata(policy),
            "image_preprocess_in_onnx": bool(image_preprocess_in_onnx),
            "calibration_input_format": "raw_images",
            "onnx_graph_input_format": "raw_images" if image_preprocess_in_onnx else "preprocessed_images",
            "num_calibration_samples": len(calibration_inputs),
            "verification_enabled": verify,
            "fail_on_verify_mismatch": bool(fail_on_verify_mismatch),
            "calibration_source": calibration_info["source"],
            "calibration_split": calibration_info["split"],
            "dataset_size": calibration_info["dataset_size"],
            "export_target": "dinov3cnn",
        }
        saved_calibration_dir = save_calibration_data(
            calibration_inputs=calibration_inputs,
            checkpoint_dir=checkpoint_dir,
            calibration_data_dir=calibration_data_dir,
            verification_outputs=verification_outputs if verify else None,
            metadata=metadata,
        )

    return {
        "checkpoint_dir": checkpoint_dir,
        "onnx_path": onnx_path,
        "onnx_data_path": onnx_data_path,
        "onnx_metadata_path": metadata_path,
        "netron_model_path": saved_netron_model_path,
        "export_dtype": export_dtype_name,
        "graph_summary": graph_summary,
        "input_names": input_names,
        "output_names": output_names,
        "rgb_keys": list(export_module.rgb_keys),
        "export_target": "dinov3cnn",
        "image_preprocess_in_onnx": bool(image_preprocess_in_onnx),
        "calibration_data_dir": saved_calibration_dir,
        "num_calibration_samples": calibration_info["num_samples"],
        "verification_samples": len(verification_outputs),
    }


def export_visual_encoder_to_onnx(
    checkpoint_dir: str,
    onnx_path: str,
    args=None,
    device: str = "cpu",
    batch_size: int = 1,
    opset_version: int = 19,
    verify: bool = False,
    rtol: float = 1e-4,
    atol: float = 1e-4,
    save_calibration_data_flag: bool = True,
    calibration_data_dir: Optional[str] = None,
    use_external_data: bool = True,
    external_data_filename: Optional[str] = None,
    external_data_size_threshold: int = 1024,
    export_modules_as_functions: Any = False,
    dynamic_batch: bool = True,
    optimize: bool = False,
    report: bool = False,
    save_netron_model: bool = True,
    netron_model_path: Optional[str] = None,
    export_dtype: Any = torch.float32,
    fail_on_verify_mismatch: bool = True,
    image_preprocess_in_onnx: bool = False,
):
    patch_for_onnx_export()
    export_dtype = _resolve_export_dtype(export_dtype)
    export_dtype_name = _export_dtype_name(export_dtype)
    if export_dtype == torch.float16 and torch.device(device).type == "cpu":
        logger.warning(
            "export_dtype=fp16 with device=cpu may fail because some PyTorch/ONNXRuntime CPU ops "
            "do not support float16. Prefer device=cuda for fp16 export."
        )

    policy = _load_policy_for_onnx(checkpoint_dir, args=args, device=device)
    policy = policy.to(device=device, dtype=export_dtype)
    _disable_export_only_features(policy, export_dtype=export_dtype)
    _disable_scene_encoder_preprocess(policy)
    policy.eval()
    visual_encoder_rgb_keys = _resolve_visual_encoder_rgb_keys(args)
    if visual_encoder_rgb_keys is not None:
        logger.info(
            "Exporting visual encoder with rgb key override: %s",
            visual_encoder_rgb_keys,
        )

    export_module = VisualEncoderExportWrapper(
        policy,
        image_preprocess_in_onnx=image_preprocess_in_onnx,
        rgb_keys=visual_encoder_rgb_keys,
    )
    export_module = export_module.to(device=device, dtype=export_dtype)
    export_module.eval()

    dummy_input = build_dinov3cnn_dummy_input(
        policy,
        batch_size=batch_size,
        device=device,
        image_preprocess_in_onnx=image_preprocess_in_onnx,
        rgb_keys=visual_encoder_rgb_keys,
    )
    input_names = _ordered_input_names_for_export(export_module, dummy_input)
    ordered_dummy_input = _cast_floating_tensors(
        {name: dummy_input[name] for name in input_names},
        device=device,
        dtype=export_dtype,
    )
    output_names = ["scene_features"]

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {name: {0: "batch_size"} for name in input_names}
        dynamic_axes["scene_features"] = {0: "batch_size"}

    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)

    logger.info(
        "Using legacy torch.onnx exporter with dynamo=False, export_dtype=%s, and image_preprocess_in_onnx=%s for export_target=visual_encoder.",
        export_dtype_name,
        bool(image_preprocess_in_onnx),
    )
    if _resolve_export_modules_as_functions(policy, export_modules_as_functions):
        logger.warning(
            "Ignoring export_modules_as_functions=%s for export_target=visual_encoder.",
            export_modules_as_functions,
        )
    if optimize or report:
        logger.warning(
            "Ignoring optimize=%s and report=%s because RTPolicy ONNX export is forced to the legacy exporter (dynamo=False) to avoid onnxscript custom domains.",
            bool(optimize),
            bool(report),
        )

    export_kwargs = dict(
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        do_constant_folding=_should_do_constant_folding(device),
        export_params=True,
        keep_initializers_as_inputs=False,
        dynamo=False,
        verbose=False,
    )
    if dynamic_axes is not None:
        export_kwargs["dynamic_axes"] = dynamic_axes
    else:
        logger.info("Exporting ONNX with static batch dimension fixed to %s.", batch_size)

    with torch.no_grad():
        torch.onnx.export(
            export_module,
            kwargs=ordered_dummy_input,
            f=onnx_path,
            **export_kwargs,
        )

    onnx_data_path = None
    if use_external_data:
        onnx_data_path = _export_external_data_artifact(
            onnx_path=onnx_path,
            external_data_filename=external_data_filename,
            external_data_size_threshold=external_data_size_threshold,
        )
        logger.info("ONNX external data saved: %s", onnx_data_path)

    graph_summary = _summarize_onnx_graph(onnx_path)
    if graph_summary is not None:
        logger.info("ONNX graph summary: %s", graph_summary)
    saved_netron_model_path = None
    if save_netron_model:
        saved_netron_model_path = _write_netron_view_model(
            onnx_path=onnx_path,
            netron_model_path=netron_model_path,
        )
        if saved_netron_model_path is not None:
            logger.info("Netron view ONNX saved: %s", saved_netron_model_path)
    metadata_path = _write_onnx_runtime_metadata(
        onnx_path,
        _build_onnx_runtime_metadata(
            policy=policy,
            input_names=input_names,
            output_names=output_names,
            export_target="visual_encoder",
            export_dtype=export_dtype,
            image_preprocess_in_onnx=image_preprocess_in_onnx,
            rgb_keys=export_module.rgb_keys,
        ),
    )

    calibration_inputs: List[Dict[str, torch.Tensor]] = []
    calibration_info = {
        "source": "disabled",
        "split": None,
        "dataset_size": 0,
        "num_samples": 0,
    }
    if args is not None and save_calibration_data_flag:
        calibration_inputs, calibration_info = collect_dinov3cnn_calibration_inputs(
            policy,
            args,
            device=device,
            rgb_keys=export_module.rgb_keys,
        )

    verification_outputs = {}
    if verify:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("verify requested but onnxruntime is unavailable") from exc

        samples_to_verify = calibration_inputs or [
            build_dinov3cnn_raw_dummy_input(
                policy,
                batch_size=1,
                device=device,
                rgb_keys=export_module.rgb_keys,
            )
        ]
        ort_session = _create_onnxruntime_session_for_verify(
            onnx_path=onnx_path,
            device=device,
            export_dtype=export_dtype,
        )
        for idx, sample_input in enumerate(samples_to_verify):
            onnx_sample_input = _prepare_sample_for_onnx(
                policy,
                sample_input,
                device=device,
                export_dtype=export_dtype,
                image_preprocess_in_onnx=image_preprocess_in_onnx,
                rgb_keys=export_module.rgb_keys,
            )
            verification_outputs[idx] = verify_module_onnx_export(
                onnx_path=onnx_path,
                module=export_module,
                sample_input=onnx_sample_input,
                output_names=output_names,
                rtol=rtol,
                atol=atol,
                ort_session=ort_session,
                fail_on_mismatch=fail_on_verify_mismatch,
            )

    saved_calibration_dir = None
    if save_calibration_data_flag and calibration_inputs:
        metadata = {
            "checkpoint_dir": checkpoint_dir,
            "onnx_path": os.path.abspath(onnx_path),
            "onnx_data_path": onnx_data_path,
            "onnx_metadata_path": metadata_path,
            "netron_model_path": saved_netron_model_path,
            "export_dtype": export_dtype_name,
            "onnx_input_dtype": export_dtype_name,
            "input_names": input_names,
            "output_names": output_names,
            "rgb_keys": list(export_module.rgb_keys),
            "preprocess": _build_image_preprocess_metadata(policy),
            "image_preprocess_in_onnx": bool(image_preprocess_in_onnx),
            "calibration_input_format": "raw_images",
            "onnx_graph_input_format": "raw_images" if image_preprocess_in_onnx else "preprocessed_images",
            "num_calibration_samples": len(calibration_inputs),
            "verification_enabled": verify,
            "fail_on_verify_mismatch": bool(fail_on_verify_mismatch),
            "calibration_source": calibration_info["source"],
            "calibration_split": calibration_info["split"],
            "dataset_size": calibration_info["dataset_size"],
            "export_target": "visual_encoder",
        }
        saved_calibration_dir = save_calibration_data(
            calibration_inputs=calibration_inputs,
            checkpoint_dir=checkpoint_dir,
            calibration_data_dir=calibration_data_dir,
            verification_outputs=verification_outputs if verify else None,
            metadata=metadata,
        )

    return {
        "checkpoint_dir": checkpoint_dir,
        "onnx_path": onnx_path,
        "onnx_data_path": onnx_data_path,
        "onnx_metadata_path": metadata_path,
        "netron_model_path": saved_netron_model_path,
        "export_dtype": export_dtype_name,
        "graph_summary": graph_summary,
        "input_names": input_names,
        "output_names": output_names,
        "rgb_keys": list(export_module.rgb_keys),
        "export_target": "visual_encoder",
        "image_preprocess_in_onnx": bool(image_preprocess_in_onnx),
        "calibration_data_dir": saved_calibration_dir,
        "num_calibration_samples": calibration_info["num_samples"],
        "verification_samples": len(verification_outputs),
    }


def onnx(args, logger=None):
    checkpoint_dir = args.checkpoint_dir
    onnx_path = args.onnx_path
    device = getattr(args, "device", "cpu")
    batch_size = int(getattr(args, "batch_size", 1))
    opset_version = int(getattr(args, "opset_version", 19))
    verify = bool(getattr(args, "verify", False))
    rtol = float(getattr(args, "rtol", 1e-4))
    atol = float(getattr(args, "atol", 1e-4))
    save_calibration_data_flag = bool(getattr(args, "save_calibration_data", True))
    calibration_data_dir = getattr(args, "calibration_data_dir", None)
    use_external_data = bool(getattr(args, "use_external_data", True))
    external_data_filename = getattr(args, "external_data_filename", None)
    external_data_size_threshold = int(getattr(args, "external_data_size_threshold", 1024))
    export_modules_as_functions = getattr(args, "export_modules_as_functions", "diffusion")
    dynamic_batch = bool(getattr(args, "dynamic_batch", True))
    optimize = bool(getattr(args, "optimize", False))
    report = bool(getattr(args, "report", False))
    save_netron_model = bool(getattr(args, "save_netron_model", True))
    netron_model_path = getattr(args, "netron_model_path", None)
    export_target = _resolve_export_target(getattr(args, "export_target", "policy"))
    export_dtype = _resolve_export_dtype(getattr(args, "export_dtype", "fp32"))
    fail_on_verify_mismatch = bool(getattr(args, "fail_on_verify_mismatch", True))
    image_preprocess_in_onnx = bool(getattr(args, "image_preprocess_in_onnx", False))

    if export_target == "policy":
        export_fn = export_policy_to_onnx
    elif export_target == "dinov3cnn":
        export_fn = export_dinov3cnn_to_onnx
    elif export_target == "visual_encoder":
        export_fn = export_visual_encoder_to_onnx
    else:
        raise ValueError(f"Unsupported export_target={export_target!r}.")
    result = export_fn(
        checkpoint_dir=checkpoint_dir,
        onnx_path=onnx_path,
        args=args,
        device=device,
        batch_size=batch_size,
        opset_version=opset_version,
        verify=verify,
        rtol=rtol,
        atol=atol,
        save_calibration_data_flag=save_calibration_data_flag,
        calibration_data_dir=calibration_data_dir,
        use_external_data=use_external_data,
        external_data_filename=external_data_filename,
        external_data_size_threshold=external_data_size_threshold,
        export_modules_as_functions=export_modules_as_functions,
        dynamic_batch=dynamic_batch,
        optimize=optimize,
        report=report,
        save_netron_model=save_netron_model,
        netron_model_path=netron_model_path,
        export_dtype=export_dtype,
        fail_on_verify_mismatch=fail_on_verify_mismatch,
        image_preprocess_in_onnx=image_preprocess_in_onnx,
    )
    if logger is not None:
        try:
            logger.info("ONNX export finished: %s", result)
        except RuntimeError as exc:
            if "initialize the accelerate state" not in str(exc):
                raise
            logging.getLogger(__name__).info("ONNX export finished: %s", result)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Export RTPolicy FM checkpoint to ONNX.")
    parser.add_argument("--checkpoint-dir", required=True, help="Checkpoint directory.")
    parser.add_argument("--onnx-path", required=True, help="Output ONNX file path.")
    parser.add_argument("--device", default="cpu", help="Export device, e.g. cpu or cuda.")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy input batch size.")
    parser.add_argument("--opset-version", type=int, default=19, help="ONNX opset version.")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX with onnxruntime.")
    parser.add_argument(
        "--fail-on-verify-mismatch",
        dest="fail_on_verify_mismatch",
        action="store_true",
        default=True,
        help="Raise an error when ONNXRuntime outputs are outside rtol/atol.",
    )
    parser.add_argument(
        "--no-fail-on-verify-mismatch",
        dest="fail_on_verify_mismatch",
        action="store_false",
        help="Log and save verification differences without failing the export.",
    )
    parser.add_argument("--rtol", type=float, default=1e-4, help="Verification relative tolerance.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Verification absolute tolerance.")
    parser.add_argument(
        "--save-calibration-data",
        dest="save_calibration_data",
        action="store_true",
        default=True,
        help="Save calibration inputs and verification outputs under checkpoint_dir/onnx_calibration.",
    )
    parser.add_argument(
        "--no-save-calibration-data",
        dest="save_calibration_data",
        action="store_false",
        help="Disable saving calibration inputs and verification outputs.",
    )
    parser.add_argument(
        "--calibration-data-dir",
        default=None,
        help="Optional calibration output directory. Relative paths are resolved under checkpoint_dir.",
    )
    parser.add_argument(
        "--use-external-data",
        dest="use_external_data",
        action="store_true",
        default=True,
        help="Save model weights as external ONNX data file (e.g. model.onnx.data).",
    )
    parser.add_argument(
        "--no-use-external-data",
        dest="use_external_data",
        action="store_false",
        help="Disable ONNX external data and keep all weights inside the .onnx file.",
    )
    parser.add_argument(
        "--external-data-filename",
        default=None,
        help="Optional external data filename. Default: <onnx filename>.data",
    )
    parser.add_argument(
        "--external-data-size-threshold",
        type=int,
        default=1024,
        help="Tensor size threshold (bytes) for external data conversion. 0 forces all tensors to external data.",
    )
    parser.add_argument(
        "--dynamic-batch",
        dest="dynamic_batch",
        action="store_true",
        default=True,
        help="Export ONNX with dynamic batch axis on inputs and outputs.",
    )
    parser.add_argument(
        "--static-batch",
        dest="dynamic_batch",
        action="store_false",
        help="Disable dynamic batch axis and keep the exported ONNX batch dimension fixed to --batch-size.",
    )
    parser.add_argument(
        "--optimize",
        dest="optimize",
        action="store_true",
        default=False,
        help="Enable PyTorch ONNX dynamo optimizer passes after graph translation.",
    )
    parser.add_argument(
        "--no-optimize",
        dest="optimize",
        action="store_false",
        help="Disable PyTorch ONNX dynamo optimizer passes to avoid slow large-graph rewrites.",
    )
    parser.add_argument(
        "--report",
        dest="report",
        action="store_true",
        default=False,
        help="Generate PyTorch ONNX exporter debug reports in the current working directory.",
    )
    parser.add_argument(
        "--no-report",
        dest="report",
        action="store_false",
        help="Disable PyTorch ONNX exporter debug reports.",
    )
    parser.add_argument(
        "--export-modules-as-functions",
        nargs="?",
        const="true",
        default="diffusion",
        help=(
            "Export submodules as ONNX local functions. "
            "Supported values: false, true, safe, netron, minimal, policy, obs_encoder, "
            "scene_encoder, scene_adaptor, state_adaptor, diffusion."
        ),
    )
    parser.add_argument(
        "--no-export-modules-as-functions",
        dest="export_modules_as_functions",
        action="store_const",
        const="false",
        help="Disable ONNX local-function export.",
    )
    parser.add_argument(
        "--save-netron-model",
        dest="save_netron_model",
        action="store_true",
        default=True,
        help="Save a lightweight <model>.netron.onnx view file with initializer payloads stripped.",
    )
    parser.add_argument(
        "--no-save-netron-model",
        dest="save_netron_model",
        action="store_false",
        help="Disable the lightweight Netron view ONNX sidecar.",
    )
    parser.add_argument(
        "--netron-model-path",
        default=None,
        help="Optional output path for the lightweight Netron view ONNX. Default: <onnx-path>.netron.onnx.",
    )
    parser.add_argument(
        "--export-target",
        default="policy",
        help="ONNX export target. Supported values: policy, dinov3cnn, visual_encoder.",
    )
    parser.add_argument(
        "--export-dtype",
        choices=("fp32", "fp16"),
        default="fp32",
        help="Floating point dtype used for model weights and ONNX graph inputs.",
    )
    parser.add_argument(
        "--image-preprocess-in-onnx",
        dest="image_preprocess_in_onnx",
        action="store_true",
        default=False,
        help="Embed tensor image preprocessing (resize/crop/normalize) into the ONNX graph.",
    )
    parser.add_argument(
        "--no-image-preprocess-in-onnx",
        dest="image_preprocess_in_onnx",
        action="store_false",
        help="Keep tensor image preprocessing outside the ONNX graph.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    result = onnx(cli_args)
    print(result)
