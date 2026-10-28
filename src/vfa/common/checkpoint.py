"""Resolve historical checkpoint metadata through the installed VFA package.

Historical EfficientViT checkpoints missing a fusion mode retain their projected
architecture. Dataset IDs, paths, tensors and checkpoint files are untouched.
"""

from copy import deepcopy
import json
from pathlib import Path


_LEGACY_CLASS_NAMES = {
    "VODPPlusDitFlowMatching": "VFADitFlowMatching",
    "VODPPlusDitFlowMatchingConfig": "VFADitFlowMatchingConfig",
    "RTPolicyDitFlowMatching": "VFADitFlowMatching",
    "RTPolicyDitFlowMatchingConfig": "VFADitFlowMatchingConfig",
    "RTPolicyDataset": "VFADataset",
    "DRRMDataset": "VFADataset",
}


def canonical_class_path(path):
    """Return the current spelling of a serialized local class reference."""
    if not isinstance(path, str):
        return path
    for package in ("drrm", "rtpolicy"):
        if path.startswith(package + "."):
            path = "vfa." + path[len(package) + 1:]
            break
    if path.startswith("vfa."):
        path = path.replace(".policy.vodpp.", ".policy.vfa.")
        path = path.replace(".policy.rtpolicy.", ".policy.vfa.")
        path = path.replace(".datasets.drrm_dataset", ".datasets.vfa_dataset")
        path = path.replace(".datasets.rtpolicy_dataset", ".datasets.vfa_dataset")
    module, separator, name = path.rpartition(".")
    if path.startswith("vfa.") or module in ("", "dit_fm_policy"):
        name = _LEGACY_CLASS_NAMES.get(name, name)
    return module + separator + name


def _normalize_reference(value):
    if isinstance(value, str):
        return canonical_class_path(value)
    if isinstance(value, list):
        return [_normalize_reference(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_normalize_reference(item) for item in value)
    return value


def normalize_checkpoint_metadata(config):
    """Copy loading metadata, preserving historical classes and visual shapes."""
    normalized = deepcopy(config)
    for field in ("pkg_map", "auto_map"):
        references = normalized.get(field)
        if isinstance(references, dict):
            normalized[field] = {
                name: _normalize_reference(value)
                for name, value in references.items()
            }
    if isinstance(normalized.get("architectures"), list):
        normalized["architectures"] = [
            _LEGACY_CLASS_NAMES.get(name, name)
            for name in normalized["architectures"]
        ]
    vision = normalized.get("obs_encoder", {}).get("emvis_config", {})
    if isinstance(vision, dict) and vision.get("backbone_target", "").lower() == "efficientvit":
        # Before native support, every missing-mode checkpoint was projected.
        vision.setdefault("efficientvit_fusion_mode", "projected")
    return normalized


def load_checkpoint_config(checkpoint_dir):
    """Read current checkpoint metadata without rewriting the source JSON."""
    path = Path(checkpoint_dir) / "config.json"
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return normalize_checkpoint_metadata(config)
