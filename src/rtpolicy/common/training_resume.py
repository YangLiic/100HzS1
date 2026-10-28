"""Restore EMA progression and Adam scalar placement after Accelerate resume."""
from pathlib import Path

import torch
from safetensors.torch import load_model


def restore_normalizer_devices(models, device):
    """Custom normalizer state loading can recreate parameters on CPU."""
    for model in models:
        normalizer = getattr(model, "normalizer", None)
        if normalizer is not None:
            normalizer.to(device=device)


def restore_adam_step_devices(optimizer):
    optimizer = getattr(optimizer, "optimizer", optimizer)
    if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
        return 0
    restored = 0
    for group in optimizer.param_groups:
        if group.get("capturable", False) or group.get("fused", False):
            continue
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            step = state.get("step")
            if isinstance(step, torch.Tensor) and step.is_cuda:
                # A fresh noncapturable Adam keeps this scalar on CPU.
                # CUDA placement after load causes .item() syncs per parameter.
                state["step"] = step.cpu()
                restored += 1
    return restored


def restore_training_ema(ema_policy_model, ema_model, optimizer, checkpoint_path,
                         global_step, gradient_accumulation_steps=1):
    load_model(ema_policy_model, str(Path(checkpoint_path)/"ema/model.safetensors"), strict=False)
    # The trainer updates EMA on each microbatch, including accumulation steps.
    ema_model.optimization_step = int(global_step)*int(gradient_accumulation_steps)
    return dict(ema_optimization_step=ema_model.optimization_step,
                adam_step_tensors_restored_to_cpu=restore_adam_step_devices(optimizer))
