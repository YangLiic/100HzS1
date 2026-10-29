"""PyTorch inference adapter, sharing the native runtime's input/noise contract."""
from importlib import import_module

import numpy as np
import torch

from .common import PolicyRuntime


def policy_metadata(policy):
    keys = list(policy.obs_encoder.rgb_key)
    return dict(rgb_keys=keys, raw_image_shapes={k: list(policy.config.shape_meta['obs'][k]['shape']) for k in keys},
                horizon=policy.horizon, action_dim=policy.action_dim,
                n_action_steps=policy.n_action_steps, n_obs_steps=policy.n_obs_steps,
                n_image_obs_steps=policy.n_image_obs_steps, state_key=policy.state_key,
                supervision_key=policy.supervision_key, flow_type=policy.flow_type)


class EagerPolicy(PolicyRuntime):
    def __init__(self, checkpoint, device, options):
        if torch.device(device).type == 'cpu' and options['precision'] != 'fp32':
            raise ValueError('CPU eager inference requires precision: fp32')
        load_policy = import_module('100hzs1.models.base_runner').load_policy
        configure = import_module('100hzs1.common.inference').configure_policy_inference
        self.policy = load_policy(str(checkpoint), device='cpu').eval()
        configure(self.policy, overrides=options)
        self.policy.float()
        # Plain attributes are not changed by Module.float().
        self.policy.model.dtype = self.policy.model.t_embedder.dtype = torch.float32
        if getattr(self.policy.model, 'r_embedder', None) is not None:
            self.policy.model.r_embedder.dtype = torch.float32
        self.policy.to(device)
        self._initialize(policy_metadata(self.policy), options, device)
        if options['use_cuda_graph'] and torch.device(device).type != 'cuda':
            raise ValueError('CUDA Graph requires a CUDA device')
        self._hook = self.policy.model.register_forward_hook(self._count)

    def _count(self, *_):
        self.network_calls += 1

    @property
    def graph_stats(self):
        graph = self.policy.obs_encoder.scene_encoder._backbone_cuda_graph
        return dict(graph_captures=graph.captures if graph else 0,
                    graph_replays=graph.replays if graph else 0,
                    graph_scope='visual_backbone')

    def predict_action(self, observations, *, seed=None, noise_trajectory=None,
                       return_flow=False, use_cuda_graph=None, use_kv_cache=None):
        with self._lock:
            arrays, batch = self._inputs(observations)
            noise = self._noise(batch, seed, noise_trajectory)
            obs = {k: torch.from_numpy(v).to(self.device) for k, v in arrays.items()}
            noise = torch.tensor(noise, device=self.device)
            graph = self.inference_options['use_cuda_graph'] if use_cuda_graph is None else use_cuda_graph
            cache = self.inference_options['use_kv_cache'] if use_kv_cache is None else use_kv_cache
            device_type = torch.device(self.device).type
            dtype = torch.bfloat16 if self.precision == 'bf16' else torch.float16
            before = self.network_calls
            with torch.inference_mode(), torch.autocast(device_type, dtype=dtype,
                    enabled=device_type == 'cuda' and self.precision != 'fp32'):
                result = self.policy.predict_action(obs, noise_trajectory=noise,
                    return_flow=return_flow, use_cuda_graph=graph, use_kv_cache=cache)
            self.last_nfe = self.network_calls - before
            if self.last_nfe != self.num_inference_timesteps:
                raise RuntimeError(f'Expected {self.num_inference_timesteps} steps, got {self.last_nfe}')
            if graph and not self.graph_stats['graph_replays']:
                raise RuntimeError('CUDA Graph was requested but the backbone did not replay')
            out = {k: v.detach().float().cpu().numpy().copy() if torch.is_tensor(v) else v for k, v in result.items()}
            return self._validate_output(out, batch)

    def close(self):
        if getattr(self, '_hook', None) is not None:
            self._hook.remove()
            self._hook = None
        self.policy = None
