"""Conditional MeanFlow core; no guidance or dispersive regularizer.

Convention: z_t=(1-t)*action+t*noise, 0<=r<=t<=1.
Reference: LogSSim/MP1, commit e1e9e8fc60da9137b699ad4544d34fcb0222e624.
JVP is evaluated separately from the parameter-gradient prediction so its
stop-gradient target does not retain a higher-order parameter graph.
"""
from contextlib import contextmanager
import math
import torch

DEFAULTS = dict(time_distribution="logit_normal", time_mean=-0.4, time_std=1.0,
                equal_time_ratio=0.5, loss_type="adaptive_l2", loss_gamma=0.5,
                loss_eps=1e-3, guidance_scale=1.0, time_embed_scale=None)


def validate_options(options=None):
    options = dict(options or {})
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown MeanFlow options: {sorted(unknown)}")
    result = dict(DEFAULTS, **options)
    if result["time_distribution"] not in {"logit_normal", "uniform"}:
        raise ValueError("MeanFlow time_distribution must be logit_normal or uniform")
    if result["loss_type"] not in {"adaptive_l2", "mse"}:
        raise ValueError("MeanFlow loss_type must be adaptive_l2 or mse")
    for key in ("time_mean", "time_std", "equal_time_ratio", "loss_gamma", "loss_eps", "guidance_scale"):
        if not isinstance(result[key], (float, int)) or not math.isfinite(result[key]):
            raise ValueError(f"MeanFlow {key} must be finite")
    if not 0 <= result["equal_time_ratio"] <= 1 or result["time_std"] <= 0:
        raise ValueError("MeanFlow requires equal_time_ratio in [0,1] and time_std > 0")
    if not 0 < result["loss_gamma"] <= 1 or result["loss_eps"] <= 0:
        raise ValueError("MeanFlow requires loss_gamma in (0,1] and loss_eps > 0")
    if result["guidance_scale"] != 1.0:
        raise ValueError("Only pure MeanFlow is implemented: guidance_scale must be 1.0")
    scale = result["time_embed_scale"]
    if scale is not None and (
        not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0
    ):
        raise ValueError("MeanFlow time_embed_scale must be positive and finite, or null for legacy checkpoints")
    return result


def time_embedding_scale(options, legacy_scale):
    """Old MeanFlow weights used the FM bucket scale; new recipes save 1.0."""
    scale = (options or {}).get("time_embed_scale")
    return legacy_scale if scale is None else scale


def sample_time_pairs(batch_size, device, options):
    if options["time_distribution"] == "uniform":
        pairs = torch.rand(batch_size, 2, device=device, dtype=torch.float32)
    else:
        pairs = torch.sigmoid(torch.randn(batch_size, 2, device=device, dtype=torch.float32)
                              * options["time_std"] + options["time_mean"])
    t, r = pairs.max(dim=1).values, pairs.min(dim=1).values
    selected = torch.randperm(batch_size, device=device)[:int(batch_size * options["equal_time_ratio"])]
    equal = torch.zeros(batch_size, device=device, dtype=torch.bool)
    equal[selected] = True
    return t, torch.where(equal, t, r)


@contextmanager
def math_attention(module):
    """Use primitive attention only for JVP; restore every original setting."""
    saved = [(m, m.fused_attn) for m in module.modules() if hasattr(m, "fused_attn")]
    try:
        for m, _ in saved:
            m.fused_attn = False
        yield
    finally:
        for m, enabled in saved:
            m.fused_attn = enabled


def objective(velocity_fn, z, t, r, velocity, mask, options, attention_module):
    """velocity_fn includes the raw-action adapter and continuous time scaling."""
    with torch.no_grad(), math_attention(attention_module):
        _, derivative = torch.func.jvp(velocity_fn, (z, t, r),
                                      (velocity, torch.ones_like(t), torch.zeros_like(r)))
        correction = (t - r)[:, None, None] * derivative.float()
        target = (velocity.float() - correction).detach()
    prediction = velocity_fn(z, t, r)
    error = prediction.float() - target
    weights = mask.to(error.dtype)
    denominator = weights.flatten(1).sum(1).clamp_min(1)
    per_sample = (error.square() * weights).flatten(1).sum(1) / denominator
    if options["loss_type"] == "adaptive_l2":
        adaptive = (per_sample.detach() + options["loss_eps"]).pow(-(1 - options["loss_gamma"]))
        losses = per_sample * adaptive
    else:
        losses = per_sample
    loss = losses.mean()
    equal = t == r
    def subset_mean(values, selected):
        return ((values.detach() * selected).sum() / selected.sum().clamp_min(1)).detach()
    return {"loss": loss, "meanflow_mse": per_sample.detach().mean(),
            "meanflow_mse_equal_time": subset_mean(per_sample, equal),
            "meanflow_mse_interval": subset_mean(per_sample, ~equal),
            "meanflow_loss_equal_time": subset_mean(losses, equal),
            "meanflow_loss_interval": subset_mean(losses, ~equal),
            "meanflow_velocity_rms": ((velocity.float().square() * weights).sum() / weights.sum().clamp_min(1)).sqrt().detach(),
            "meanflow_jvp_rms": ((derivative.float().square() * weights).sum() / weights.sum().clamp_min(1)).sqrt().detach(),
            "meanflow_correction_rms": ((correction.square() * weights).sum() / weights.sum().clamp_min(1)).sqrt().detach(),
            "equal_time_ratio": (t == r).float().mean().detach()}
