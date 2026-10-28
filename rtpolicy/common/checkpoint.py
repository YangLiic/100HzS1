"""Resolve historical checkpoint metadata through the installed RTPolicy package.

The mapping applies to class-reference fields only. Dataset IDs, paths, tensor
names, and checkpoint files themselves are left untouched.
"""

from copy import deepcopy
import json
from pathlib import Path


_LEGACY_CLASS_NAMES = {
    "VODPPlusDitFlowMatching": "RTPolicyDitFlowMatching",
    "VODPPlusDitFlowMatchingConfig": "RTPolicyDitFlowMatchingConfig",
    "DRRMDataset": "RTPolicyDataset",
}


def canonical_class_path(path):
    """Return the current spelling of a serialized local class reference."""
    if not isinstance(path, str):
        return path
    if path.startswith("drrm."):
        path = "rtpolicy." + path[len("drrm."):]
    if path.startswith("rtpolicy."):
        path = path.replace(".policy.vodpp.", ".policy.rtpolicy.")
        path = path.replace(".datasets.drrm_dataset", ".datasets.rtpolicy_dataset")
    module, separator, name = path.rpartition(".")
    if path.startswith("rtpolicy.") or module in ("", "dit_fm_policy"):
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
    """Copy a config and normalize only its dynamic class-reference metadata."""
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
    return normalized


def load_checkpoint_config(checkpoint_dir):
    """Read current checkpoint metadata without rewriting the source JSON."""
    path = Path(checkpoint_dir) / "config.json"
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return normalize_checkpoint_metadata(config)
