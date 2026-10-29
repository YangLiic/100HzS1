"""One YAML source for prediction defaults and explicit runtime overrides."""
from pathlib import Path
import json
import yaml
DEFAULT_INFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "config/inference.yaml"

def load_inference_options(path=None, overrides=None, checkpoint_config=None):
    """Precedence: algorithm defaults < checkpoint < selected YAML < explicit CLI."""
    options = yaml.safe_load(DEFAULT_INFERENCE_CONFIG.read_text())
    if checkpoint_config:
        mode = checkpoint_config.get("flow_type", "fm")
        if mode not in {"fm", "meanflow"}:
            raise ValueError("Checkpoint flow_type must be fm or meanflow")
        if mode == "meanflow":
            options["num_inference_timesteps"] = 1
        saved = checkpoint_config.get("inference") or {}
        unknown = set(saved) - set(options)
        if unknown:
            raise ValueError(f"Unknown checkpoint inference options: {sorted(unknown)}")
        options.update({k: v for k, v in saved.items() if v is not None})
    if path is not None:
        selected = yaml.safe_load(Path(path).read_text())
        if not isinstance(selected, dict):
            raise ValueError("Inference YAML must contain a mapping")
        unknown = set(selected) - set(options)
        if unknown:
            raise ValueError(f"Unknown inference options: {sorted(unknown)}")
        options.update(selected)
    if overrides:
        unknown = set(overrides) - set(options)
        if unknown:
            raise ValueError(f"Unknown inference overrides: {sorted(unknown)}")
        options.update({k: v for k, v in overrides.items() if v is not None})
    if type(options["num_inference_timesteps"]) is not int or options["num_inference_timesteps"] < 1:
        raise ValueError("num_inference_timesteps must be a positive integer")
    for name in ["use_kv_cache", "use_cuda_graph"]:
        if type(options[name]) is not bool:
            raise ValueError(f"{name} must be a YAML boolean")
    if options["precision"] not in {"bf16", "fp32"}:
        raise ValueError("precision must be bf16 or fp32")
    return dict(options)

def configure_policy_inference(model, path=None, overrides=None):
    config = model.config.to_dict() if hasattr(model, "config") else None
    options = load_inference_options(path, overrides, checkpoint_config=config)
    model.inference_options = options
    model.num_inference_timesteps = options["num_inference_timesteps"]
    if options["precision"] == "fp32":
        import torch
        model.float()
        model.model.t_embedder.dtype = torch.float32
        if getattr(model.model, "r_embedder", None) is not None:
            model.model.r_embedder.dtype = torch.float32
    return options


def checkpoint_inference_options(checkpoint, path=None, overrides=None):
    """Resolve runtime before loading tensors (also supports evaluation dry runs)."""
    config_path = Path(checkpoint) / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else None
    return load_inference_options(path, overrides, checkpoint_config=config)
