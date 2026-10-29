"""Strictly reload fresh main/EMA weights and exercise YAML-default prediction."""
from importlib import import_module
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_model
load_policy = import_module("100hzs1.models.base_runner").load_policy

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--inference-config",type=Path)
    args=parser.parse_args()
    config=json.loads((args.checkpoint/"config.json").read_text())
    assert config["pkg_map"]["AutoModel"].startswith("100hzs1.")
    vision=config["obs_encoder"]["emvis_config"]
    assert not vision["load_backbone_pretrain"]
    if vision["backbone_target"]=="dinov3cnn":
        assert vision["backbone_config"],"New DINO checkpoints must contain their architecture"
    torch.manual_seed(42)
    model=load_policy(args.checkpoint, device=args.device, inference_config=args.inference_config).eval()
    observations={}
    for key,meta in config["shape_meta"]["obs"].items():
        count=model.n_image_obs_steps if meta.get("type")=="rgb" else model.n_obs_steps
        shape=(1,count,*meta["shape"])
        observations[key]=torch.rand(shape,device=args.device) if meta.get("type")=="rgb" else torch.zeros(shape,device=args.device)
    weights=[("main",args.checkpoint/"model.safetensors")]
    if (args.checkpoint/"ema/model.safetensors").exists():
        weights.append(("ema",args.checkpoint/"ema/model.safetensors"))
    calls = []
    model.model.register_forward_hook(lambda *args: calls.append(1))
    for label,path in weights:
        calls.clear()
        load_model(model,str(path),strict=True,device=args.device)
        with torch.inference_mode(),torch.autocast(torch.device(args.device).type,dtype=torch.bfloat16,
                    enabled=model.inference_options["precision"]=="bf16"):
            action=model.predict_action(observations)["action"]
        assert len(calls) == model.num_inference_timesteps, (len(calls), model.num_inference_timesteps)
        assert action.shape==(1,model.n_action_steps,14) and torch.isfinite(action).all()
        runner=model.obs_encoder.scene_encoder._backbone_cuda_graph
        if model.inference_options["use_cuda_graph"] and torch.device(args.device).type=="cuda":
            assert runner is not None and runner.captures==1 and runner.replays>=1
        print(json.dumps(dict(checkpoint=str(args.checkpoint),weights=label,strict_reload=True,
            action_shape=list(action.shape),finite=True,flow_type=model.flow_type,network_calls=len(calls),inference=model.inference_options,
            graph_captures=runner.captures if runner else 0,graph_replays=runner.replays if runner else 0)),flush=True)

if __name__=="__main__":
    main()
