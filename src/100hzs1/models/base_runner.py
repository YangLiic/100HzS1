"""Torch model loader for training/export; configured deployment uses load_inference_policy."""
from pathlib import Path
from typing import Dict
import hydra
from safetensors.torch import load_model
from ..common.checkpoint import load_checkpoint_config
from ..common.inference import configure_policy_inference
from .base_policy import BasePolicy

def load_policy(ckp_path, device=None, inference_config=None) -> BasePolicy:
    config = load_checkpoint_config(ckp_path)
    ConfigClass = hydra.utils.get_class(config["pkg_map"]["AutoConfig"])
    PolicyClass = hydra.utils.get_class(config["pkg_map"]["AutoModel"])
    policy = PolicyClass(ConfigClass.from_dict(config))
    if device is not None:
        policy = policy.to(device)
    load_model(policy, str(Path(ckp_path) / "model.safetensors"), strict=True, device=device or "cpu")
    configure_policy_inference(policy, inference_config)
    return policy

def load_inference_policy(ckp_path, device="cuda:0", inference_config=None, overrides=None):
    """Load the YAML-selected eager or TensorRT deployment backend."""
    from ..inference import load_inference_policy as load
    return load(ckp_path, device, inference_config, overrides)

class BaseRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()
