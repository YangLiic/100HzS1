"""Verify a saved policy loads offline and produces finite action predictions."""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_model
from vfa.models.base_runner import load_policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    assert not config["obs_encoder"]["emvis_config"]["load_backbone_pretrain"]
    torch.manual_seed(42)
    model = load_policy(str(args.checkpoint), device=args.device).eval()
    # The existing loader tolerates historical metadata. New checkpoints must
    # additionally match every current parameter and buffer.
    load_model(model, str(args.checkpoint / "model.safetensors"),
               strict=True, device=args.device)
    obs = {}
    for key, meta in config["shape_meta"]["obs"].items():
        steps = model.n_image_obs_steps if meta.get("type") == "rgb" else model.n_obs_steps
        shape = (1, steps, *meta["shape"])
        obs[key] = torch.rand(shape, device=args.device) if meta.get("type") == "rgb" else torch.zeros(shape, device=args.device)
    with torch.inference_mode(), torch.autocast(device_type=torch.device(args.device).type,
                                                dtype=torch.bfloat16):
        result = model.predict_action(obs)
    action = result["action"]
    assert action.shape == (1, model.n_action_steps, 14), action.shape
    assert torch.isfinite(action).all(), action
    print(json.dumps({"checkpoint": str(args.checkpoint), "strict_reload": True,
                      "action_shape": list(action.shape),
                      "finite": True, "backbone": config["obs_encoder"]["emvis_config"]["backbone_target"]}),
          flush=True)


if __name__ == "__main__":
    main()
