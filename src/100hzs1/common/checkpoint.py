"""Metadata validation for checkpoints produced by the 100hzs1 package."""
from copy import deepcopy
import json
from pathlib import Path

def validate_class_path(path):
    if not isinstance(path, str) or not path.startswith("100hzs1."):
        raise ValueError("Expected a class from the 100hzs1 package")
    return path

def prepare_checkpoint_metadata(config):
    config = deepcopy(config)
    references = config.get("pkg_map")
    if not isinstance(references, dict):
        raise ValueError("Checkpoint is missing pkg_map")
    for name in ["AutoConfig", "AutoModel"]:
        validate_class_path(references.get(name))
    return config

def load_checkpoint_config(checkpoint_dir):
    path = Path(checkpoint_dir) / "config.json"
    return prepare_checkpoint_metadata(json.loads(path.read_text()))
