"""Shared eager / native TensorRT inference entry point.

Inputs are RGB observations plus proprioception, outputs are owned NumPy arrays.
Training and export code should keep using models.base_runner.load_policy.
"""
from pathlib import Path

from ..common.inference import checkpoint_inference_options


def load_inference_policy(checkpoint, device="cuda:0", inference_config=None,
                          overrides=None):
    """Select the configured backend; never silently fall back to another one."""
    checkpoint = Path(checkpoint).resolve()
    options = checkpoint_inference_options(checkpoint, inference_config, overrides)
    if options["backend"] == "eager":
        from .eager import EagerPolicy
        return EagerPolicy(checkpoint, device, options)
    from .tensorrt import TensorRTPolicy
    return TensorRTPolicy(checkpoint, device, options)


__all__ = ["load_inference_policy"]
