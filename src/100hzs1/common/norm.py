"""Checkpoint-compatible normalization choices for absolute joint state/action vectors."""
import numpy as np
import torch

from .normalizer import SingleFieldLinearNormalizer
from .normalize_util import get_range_normalizer_from_stat

NORM_MODES = ("minmax", "minmax_expanded", "pi", "groot")


def validate_norm(norm="minmax"):
    if not isinstance(norm, str) or norm not in NORM_MODES:
        raise ValueError(f"norm must be one of {NORM_MODES}, got {norm!r}")
    return norm


def make_normalizer(stats, norm="minmax"):
    """Build a field normalizer; missing norm keeps the legacy state schema."""
    validate_norm(norm)
    stat = {k: np.asarray(v, dtype=np.float32).copy() for k, v in stats.items()}
    if norm in ("minmax", "minmax_expanded"):
        bounds = {k: stat[k] for k in ("min", "max", "mean", "std")}
        if norm == "minmax_expanded":
            center = (bounds["min"] + bounds["max"]) / 2
            width = bounds["max"] - bounds["min"]
            dynamic = width >= 1e-7
            bounds["min"] = np.where(dynamic, center - 1.25 * width / 2, bounds["min"])
            bounds["max"] = np.where(dynamic, center + 1.25 * width / 2, bounds["max"])
        result = get_range_normalizer_from_stat(bounds)
    else:
        low, high = stat["q01"], stat["q99"]
        if norm == "pi":
            scale = 2 / (high - low + np.float32(1e-6))
            offset = -1 - low * scale
        else:
            constant = np.isclose(high, low)
            width = high - low
            scale = np.divide(np.float32(2), width, out=np.zeros_like(width), where=~constant)
            offset = np.where(constant, np.float32(0), -1 - low * scale)
        result = SingleFieldLinearNormalizer.create_manual(scale, offset, stat)
        if norm == "groot":
            for key, value in {
                "clip_bounds": np.array([-1, 1], dtype=np.float32),
                "inverse_scale": (high - low) / 2,
                "inverse_offset": (high + low) / 2,
            }.items():
                result.params_dict[key] = torch.nn.Parameter(
                    torch.from_numpy(value.copy()), requires_grad=False)
    result.requires_grad_(False)
    return result


def normalizer_from_data(values, norm):
    """Compute selected-training-frame statistics, without reading images."""
    validate_norm(norm)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Normalization requires nonempty finite data shaped (frames, dimensions)")
    stats = dict(min=values.min(0), max=values.max(0),
                 mean=values.mean(0), std=values.std(0))
    if norm in ("pi", "groot"):
        stats["q01"], stats["q99"] = np.quantile(values, [0.01, 0.99], axis=0).astype(np.float32)
    return make_normalizer(stats, norm)
