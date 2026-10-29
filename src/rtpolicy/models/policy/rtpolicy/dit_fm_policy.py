import re
from typing import Dict
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import \
    DPMSolverMultistepScheduler

from rtpolicy.models.base_policy import BasePolicy
# from .module.obs_encoder import SceneEncoder as RTPolicyEncoder
from .emvis.embodied_vision import VisualEncoder
from .common.normalizer import LinearNormalizer
from .common.module_attr_mixin import ModuleAttrMixin

import yaml
import json
from dataclasses import dataclass
from typing import Optional
from transformers import PretrainedConfig, PreTrainedModel

@dataclass
class RTPolicyDitFlowMatchingConfig(PretrainedConfig):
    shape_meta: dict
    noise_scheduler: dict
    obs_encoder: dict
    hidden_size: int
    depth: int
    num_heads: int
    horizon: int
    n_action_steps: int
    n_obs_steps: int
    # Observation field used as the state condition.
    state_key: str
    # None keeps old checkpoints/configs coupled to n_obs_steps.
    n_image_obs_steps: Optional[int] = None
    # Key supervised by the flow-matching loss (column of batch['action']).
    # None -> same as state_key (legacy statecopy behavior: condition and
    # supervision share one key, e.g. both on 'state').
    # Set 'action' to keep the state condition while supervising the
    # commanded expert action.
    supervision_key: Optional[str] = None
    obs_as_global_cond: bool = True
    diffusion_step_embed_dim: int = 256
    down_dims: tuple = (256,512,1024)
    kernel_size: int = 5
    n_groups: int = 8
    cond_predict_scale: bool = True
    self_attn_first: bool = True
    

    def __init__(self, *args, **kwargs):
        # Migrate old configs/checkpoints; serialize only the canonical state_key.
        legacy_state_key = kwargs.pop("action_key", None)
        state_key = kwargs.get("state_key")
        if legacy_state_key is not None:
            if state_key is not None and state_key != legacy_state_key:
                raise ValueError("state_key and legacy action_key must agree")
            kwargs["state_key"] = legacy_state_key
        super().__init__(*args, **kwargs)
        
        self.auto_map = {}
        self.pkg_map = {}
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_customed_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_customed_json(cls, json_path: str):
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_customed_dict(cls, config_dict):
        return cls(**config_dict)


def create_dit_model(
    config: RTPolicyDitFlowMatchingConfig,
    action_dim: int,
    scene_cond_len: int,
    scene_pos_embed_config: list
):
    horizon = config.horizon
    n_obs_steps = config.n_obs_steps
    hidden_size = config.hidden_size

    from .module.conditional_dit_head import DiT
    return DiT(
        output_dim=action_dim,
        horizon=horizon,
        hidden_size=hidden_size,
        depth=config.depth,
        num_heads=config.num_heads,
        self_attn_first=config.self_attn_first,
        scene_cond_len=scene_cond_len,
        scene_pos_embed_config=scene_pos_embed_config
    )

class RTPolicyDitFlowMatching(BasePolicy, PreTrainedModel, ModuleAttrMixin):
    config_class = RTPolicyDitFlowMatchingConfig
    state_keys = ['state', 'delta_state', 'endpose', 'delta_endpose']

    def __init__(self, config: RTPolicyDitFlowMatchingConfig):
        super().__init__(config)
        self.num_timestep_buckets = config.noise_scheduler['num_train_timesteps']
        self.num_inference_timesteps = config.noise_scheduler['num_inference_timesteps']
        self.beta_dist = torch.distributions.Beta(
            config.noise_scheduler['noise_beta_alpha'], 
            config.noise_scheduler['noise_beta_beta']
        )
        self.noise_s = config.noise_scheduler['noise_s']
        # self.obs_encoder = hydra.utils.instantiate(config.obs_encoder)
        # EncoderClass = hydra.utils.get_class(config.obs_encoder['_target_'])
        # self.obs_encoder = EncoderClass(**config.obs_encoder)
        self.obs_encoder = VisualEncoder(**config.obs_encoder)

        state_key = config.state_key
        action_meta = config.shape_meta['action']
        if 'shape' in action_meta:
            action_shape = action_meta['shape']
        else:
            action_shape = action_meta[state_key]['shape']
        horizon = config.horizon
        hidden_size = config.hidden_size
        n_action_steps = config.n_action_steps
        n_obs_steps = config.n_obs_steps
        configured_image_obs_steps = getattr(config, 'n_image_obs_steps', None)
        n_image_obs_steps = (
            n_obs_steps
            if configured_image_obs_steps is None
            else int(configured_image_obs_steps)
        )
        if n_image_obs_steps <= 0:
            raise ValueError("n_image_obs_steps must be positive")
        # parse shapes
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        V, H, W, dim = self.obs_encoder.output_shape_meta() # V, H, W, D
        obs_feature_dim = dim
        action_mask = torch.zeros(1, horizon, action_dim).bool()
        action_mask[:,n_obs_steps:,:] = True

        # create diffusion model
        num_patches = H * W
        scene_cond_len = (n_image_obs_steps * V * num_patches)
        scene_pos_embed_config = [("image", (n_image_obs_steps, V, num_patches)),]
        
        from .module.conditional_dit_head import DiT
        self.model = DiT(
            output_dim=action_dim,
            horizon=horizon,
            hidden_size=hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            self_attn_first=config.self_attn_first,
            scene_cond_len=scene_cond_len,
            scene_pos_embed_config=scene_pos_embed_config
        )

        self.scene_adaptor = self.build_condition_adapter(
            config.scene_adaptor, 
            in_features=obs_feature_dim, 
            out_features=hidden_size
        )

        # A `state` refers to an action or a proprioception vector
        self.state_adaptor = self.build_condition_adapter(
            config.state_adaptor, 
            in_features=action_dim,    # state + state mask (indicator)
            out_features=hidden_size
        )

        # self.mask_generator = LowdimMaskGenerator(
        #     action_dim=action_dim,
        #     obs_dim=0 if obs_as_global_cond else obs_feature_dim,
        #     max_n_obs_steps=n_obs_steps,
        #     fix_obs_steps=True,
        #     action_visible=False
        # )
        self.normalizer = LinearNormalizer()
        # self.normalizer = None
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.n_image_obs_steps = n_image_obs_steps
        self.action_mask = action_mask
        self.state_key = state_key
        # Supervision target key: defaults to state_key (legacy behavior).
        self.supervision_key = (
            getattr(config, 'supervision_key', None) or config.state_key
        )
        # Persisted in checkpoints. Old checkpoints load with the all-False
        # default and therefore keep the legacy behavior.
        self.register_buffer(
            'static_joint_mask',
            torch.zeros(action_dim, dtype=torch.bool),
        )
        # self.kwargs = kwargs
        self.kwargs = {} #

    @torch.no_grad()
    def set_static_joint_mask(self, mask):
        mask = torch.as_tensor(
            mask, dtype=torch.bool, device=self.static_joint_mask.device
        ).reshape(-1)
        if mask.numel() != self.action_dim:
            raise ValueError(
                f"static joint mask has {mask.numel()} dims, expected {self.action_dim}"
            )
        self.static_joint_mask.copy_(mask)

    def _has_static_joints(self):
        return bool(self.static_joint_mask.any().item())

    def _zero_static_joints(self, value):
        if not self._has_static_joints():
            return value
        return value.masked_fill(self.static_joint_mask.view(1, 1, -1), 0)

    def _hold_static_joints(self, action, current_state):
        if not self._has_static_joints():
            return action
        mask = self.static_joint_mask.view(1, 1, -1)
        return torch.where(mask, current_state[:, 0:1, :], action)

    def _masked_mse(self, pred, target):
        if not self._has_static_joints():
            return F.mse_loss(pred, target, reduction="mean")
        active = ~self.static_joint_mask
        if not bool(active.any().item()):
            return (pred - target).sum() * 0
        return F.mse_loss(pred[..., active], target[..., active], reduction="mean")

    def build_condition_adapter(
        self, projector_type, in_features, out_features):
        projector = None
        if projector_type == 'linear':
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f'Unknown projector type: {projector_type}')

        return projector
    
    def adapt_conditions(self, scene_tokens):
        '''
        scene_cond: (batch_size, patch_num, img_token_dim)
        
        return: adpated (..., hidden_size) for all input tokens
        '''
        adpated_scene = self.scene_adaptor(scene_tokens)
        return adpated_scene

    def encode_image_conditions(self, nobs, batch_size):
        """Encode the latest configured image frames, independently of state."""
        image_nobs = {}
        for key in self.obs_encoder.rgb_key:
            image = nobs[key]
            if image.shape[1] < self.n_image_obs_steps:
                raise ValueError(
                    f"{key} has {image.shape[1]} image frames, "
                    f"expected at least {self.n_image_obs_steps}"
                )
            image_nobs[key] = image[:, -self.n_image_obs_steps:].reshape(
                -1, *image.shape[2:]
            )

        image_features = self.obs_encoder(image_nobs)
        scene_cond = self.adapt_conditions(image_features)
        return scene_cond.view(batch_size, -1, scene_cond.shape[-1])
    
    # ========= inference  ============
    def conditional_sample(
            self, scene_cond, state_traj, action_mask, return_flow:bool = False, **kwargs
        ) -> torch.Tensor:
        '''
        scene_cond: image conditional data, (batch_size, patch_num, hidden_size).
        state_traj: (batch_size, 1, hidden_size), state trajectory.
        
        return: (batch_size, horizon, action_dim)
        '''
        device = state_traj.device
        dtype = state_traj.dtype
        batch_size = state_traj.shape[0]
        noisy_action = torch.randn(
            size=(batch_size, self.horizon, self.action_dim), 
            dtype=dtype, device=device
        )
        noisy_action = self._zero_static_joints(noisy_action)
        flow = []
        if return_flow:
            flow.append(noisy_action.detach().cpu().numpy())

        # Set step values
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_mask = ~action_mask
        for t in range(num_steps):
            # timesteps = t.unsqueeze(-1).to(device)
            t_discretized = int(t / float(num_steps) * self.num_timestep_buckets)
            timesteps = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            # Prepare state-action trajectory
            noisy_action[state_mask] = state_traj[state_mask]
            noisy_action = self._zero_static_joints(noisy_action)
            state_action_traj = self.state_adaptor(noisy_action)
            
            # Predict the model output
            pred_velocity = self.model(state_action_traj, timesteps, scene_cond)
            pred_velocity = self._zero_static_joints(pred_velocity)
            
            # Compute previous actions: x_t -> x_t-1
            noisy_action = noisy_action + dt * pred_velocity
            noisy_action = self._zero_static_joints(noisy_action)
            if return_flow:
                flow.append(noisy_action.detach().cpu().numpy())
        
        # Finally apply the action mask to mask invalid action dimensions
        noisy_action[state_mask] = state_traj[state_mask]

        return noisy_action, flow

    @torch.no_grad()
    def predict_action(
        self, 
        obs_dict: Dict[str, torch.Tensor], 
        return_flow:bool = False, 
        gt: Dict[str, torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        device = self.device
        dtype = self.dtype
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = {k: v.to(device=device, dtype=dtype) for k, v in obs_dict.items()}
        # filtered_obs_dict = {key: value for key, value in obs_dict.items() 
        #         if key in self.normalizer.params_dict}
        filtered_obs_dict = {
            key: value for key, value in obs_dict.items() 
            if key in self.state_keys
        }
        nobs.update(self.normalizer.normalize(filtered_obs_dict))
        if self.state_key in nobs:
            nobs[self.state_key] = self._zero_static_joints(nobs[self.state_key])
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # handle different ways of passing observation
        # condition through global feature
        scene_cond = self.encode_image_conditions(nobs, B)
        state_traj = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        state_traj[:,:To,:] = nobs[self.state_key][:, :To].view(B, To, -1)
        action_mask = self.action_mask.expand(B, -1, -1).to(device=device)

        # run sampling
        nsample, flow = self.conditional_sample(scene_cond, state_traj, action_mask, return_flow)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer[self.supervision_key].unnormalize(naction_pred)

        # get action
        start = To
        end = start + self.n_action_steps
        if gt is not None:
            if self.supervision_key in nobs:
                ngt = nobs[self.supervision_key]
            else:
                ngt = self.normalizer.normalize(
                    {self.supervision_key: gt[self.supervision_key].to(dtype=naction_pred.dtype)}
                )[self.supervision_key]
            action_loss = self._masked_mse(
                naction_pred[:,start:end], ngt[:,start:end]
            )
            mse_loss = self._masked_mse(
                action_pred[:,start:end], gt[self.supervision_key][:,start:end]
            )
            return {
                'loss': action_loss.detach(),
                'action loss': action_loss.detach(),
                'mse loss': mse_loss.detach()
            }
        
        action = action_pred[:,start:end]
        if 'delta' in self.state_key:
            key = self.state_key.replace('delta_', '')
            action = self._zero_static_joints(action)
            result = {
                'action': action + obs_dict[key][:,0:1,...],
                'delta_action': action,
                'predicted_raw': action_pred,
            }
        else:
            action = self._hold_static_joints(
                action, obs_dict[self.state_key]
            )
            result = {
                'action': action,
                'predicted_raw': action_pred,
            }

        if return_flow:
            flow = [smp[:,start:end] for smp in flow]
            result.update({
                'flow': flow,
                'gt': nobs[self.state_key][:,start:end]
            })
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.noise_s - sample) / self.noise_s

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = {**batch['obs']}
        nobs.update(self.normalizer.normalize(
            {key: nobs[key] for key in self.state_keys if key in nobs}
        ))
        state_traj = nobs[self.state_key]
        actions = self.normalizer.normalize(batch['action'])[self.supervision_key]
        state_traj = self._zero_static_joints(state_traj)
        actions = self._zero_static_joints(actions)
        batch_size = actions.shape[0]

        # handle different ways of passing observation
        # reshape B, T, ... to B*T
        scene_cond = self.encode_image_conditions(nobs, batch_size)
        action_mask = self.action_mask.expand(batch_size, -1, -1).to(device=actions.device)

        # Sample noise that we'll add to the images
        noise = torch.randn(actions.shape, device=actions.device)
        noise = self._zero_static_joints(noise)
        # Sample a random timestep for each image
        t = self.sample_time(
            actions.shape[0],
            device=actions.device,
            dtype=actions.dtype
        )
        t = t[:, None, None]  # shape (B,1,1) for broadcast
        # Convert (continuous) t -> discrete if needed
        timesteps = (t[:, 0, 0] * self.num_timestep_buckets).long() # shape (B,)
        
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_traj = (1 - t) * noise + t * actions
        velocity = actions - noise
        
        # compute loss mask
        cond_mask = ~action_mask   # leading obs steps False (condition), later action steps True
        # apply conditioning
        noisy_traj[cond_mask] = state_traj[cond_mask]   # leading frames become state, later frames stay noisy actions
        state_noisy_tokens = self.state_adaptor(noisy_traj)
        
        # Predict the noise residual
        pred = self.model(state_noisy_tokens, timesteps, scene_cond)
        # compute loss
        loss = F.mse_loss(pred, velocity, reduction='none')
        if self._has_static_joints():
            active_mask = (~self.static_joint_mask).view(1, 1, -1)
            loss_mask = action_mask & active_mask
            loss = (loss * loss_mask.type(loss.dtype)).sum()
            loss = loss / loss_mask.sum().clamp_min(1)
        else:
            loss = loss * action_mask.type(loss.dtype)
            loss = reduce(loss, 'b ... -> b (...)', 'mean')
            loss = loss.mean()
        return loss
