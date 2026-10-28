import re
from pathlib import Path
from typing import Dict
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler

from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub.constants import (PYTORCH_WEIGHTS_NAME,
                                       SAFETENSORS_SINGLE_FILE)
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, is_torch_available

from rtpolicy.common.hub_mixin import CompatiblePyTorchModelHubMixin


class BasePolicy(
        nn.Module, 
        CompatiblePyTorchModelHubMixin
    ):

    n_action_steps: int
    n_obs_steps: int
    
    # ========= Train  ============
    def compute_loss(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("compute_loss is not implemented")
    
    # ========= Inference  ============
    def predict_action(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("predict_action is not implemented")
    
    def forward(self, obs: Dict[str, torch.Tensor], action: torch.Tensor = None, **kwargs) -> torch.Tensor:
        pretraining = getattr(self, 'pretraining', False)
        
        if self.training or pretraining:
            if 'batch' in inspect.signature(self.compute_loss).parameters:
                batch = {'obs': obs, 'action': action}
                return self.compute_loss(batch, **kwargs)
            return self.compute_loss(obs, action, **kwargs)
        return self.predict_action(obs, gt=action, **kwargs)
