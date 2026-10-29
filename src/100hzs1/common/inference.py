"""Shared inference defaults and explicit backend/runtime overrides."""
from pathlib import Path
import json
import yaml

DEFAULT_INFERENCE_CONFIG = Path(__file__).resolve().parents[1] / "config/inference.yaml"


def load_inference_options(path=None, overrides=None, checkpoint_config=None):
    """Precedence: defaults < checkpoint < selected YAML < explicit arguments.

    Old Torch checkpoints predate backend selection. Their saved precision and
    Graph switches must not override the new deployment defaults. Their solver
    step count and K/V choice remain part of the restored algorithm recipe.
    """
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
        if "backend" not in saved:
            saved = {k: v for k, v in saved.items() if k in {"num_inference_timesteps", "use_kv_cache"}}
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
    for name in ["num_inference_timesteps", "tensorrt_batch_size"]:
        if type(options[name]) is not int or options[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ["use_kv_cache", "use_cuda_graph"]:
        if type(options[name]) is not bool:
            raise ValueError(f"{name} must be a YAML boolean")
    backend = options["backend"]
    if not isinstance(backend, str) or backend.lower() not in {"eager", "tensorrt"}:
        raise ValueError("backend must be eager or tensorrt")
    options["backend"] = backend.lower()
    if options["precision"] not in {"bf16", "fp16", "fp32"}:
        raise ValueError("precision must be fp16, bf16 or fp32")
    if options["backend"] == "tensorrt" and options["precision"] not in {"fp16", "fp32"}:
        raise ValueError("TensorRT deployment precision must be fp16 or fp32; use eager for bf16")
    directory = options["tensorrt_engine_dir"]
    if directory is not None and not isinstance(directory, (str, Path)):
        raise ValueError("tensorrt_engine_dir must be a path or null")
    if directory is not None:
        options["tensorrt_engine_dir"] = str(directory)
    return dict(options)


def configure_policy_inference(model, path=None, overrides=None):
    """Set a low-level Torch model's recipe; backend dispatch lives in inference."""
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
    """Resolve backend and solver without importing Torch or loading weights."""
    config_path = Path(checkpoint) / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else None
    return load_inference_options(path, overrides, checkpoint_config=config)
