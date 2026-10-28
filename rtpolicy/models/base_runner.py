import json
import logging
import os
from typing import Dict

import hydra
from transformers import AutoModel
from rtpolicy.common.checkpoint import normalize_checkpoint_metadata
from rtpolicy.models.base_policy import BasePolicy
from safetensors.torch import load_model

logger = logging.getLogger(__name__)


def load_policy(ckp_path, use_ckp_code = False, device = None) -> BasePolicy:
    """Load a checkpoint through installed class metadata by default.

    ``use_ckp_code=True`` opts into an embedded implementation. Historical
    package/class references use the installed runtime after normalization.
    Neither the source config nor its weights are rewritten.
    """
    config_path = os.path.join(ckp_path, "config.json")
    config_dict = None
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            stored_config = json.load(handle)
        config_dict = normalize_checkpoint_metadata(stored_config)
        if config_dict != stored_config and config_dict.get("pkg_map"):
            # Embedded historical code imports the former package. Resolve its
            # known policy metadata through the installed implementation instead.
            logger.info("Loading legacy checkpoint class paths through rtpolicy")
            use_ckp_code = False
    if use_ckp_code:
        policy_model = AutoModel.from_pretrained(ckp_path, trust_remote_code=True)
        # load state dict of normalizer
        load_model(policy_model, os.path.join(ckp_path, "model.safetensors"), strict=False, device=device)
    else:
        if config_dict is None:
            raise FileNotFoundError(config_path)
        ConfigClass = hydra.utils.get_class(config_dict['pkg_map']['AutoConfig'])
        PolicyClass = hydra.utils.get_class(config_dict['pkg_map']['AutoModel'])
        config = ConfigClass.from_dict(config_dict)
        policy_model = PolicyClass(config)
        policy_model = policy_model.to(device)
        # load state dict of normalizer
        load_model(policy_model, os.path.join(ckp_path, "model.safetensors"), strict=False, device=device)
    return policy_model

class BaseRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()
