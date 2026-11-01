# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT: https://github.com/facebookresearch/DiT
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
from typing import Literal


from collections import OrderedDict

import torch
import torch.nn as nn

from .blocks import AdaRMSBlock, FinalLayer, Block, TimestepEmbedder, get_1d_sincos_pos_embed_from_grid, get_multimodal_cond_pos_embed


class DiT(nn.Module):
    """
    Class for Robotics Diffusion Transformers.
    """
    def __init__(
        self,
        output_dim: int,
        horizon: int,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        self_attn_first=True,
        scene_cond_len=4096, # S * V * P
        scene_pos_embed_config=None, # S, V, P
        dtype=torch.bfloat16,
        use_final_layer=True,
        flow_type="fm",
        time_injection="tau_token",
        n_obs_steps=1,
    ):
        super().__init__()
        if time_injection not in {"tau_token", "tau_AdaRMS"}:
            raise ValueError("time_injection must be tau_token or tau_AdaRMS")
        if time_injection == "tau_AdaRMS" and not 0 <= n_obs_steps <= horizon:
            raise ValueError("n_obs_steps must be between 0 and horizon")
        self.time_injection = time_injection
        self.n_obs_steps = n_obs_steps
        self.use_final_layer = use_final_layer
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.scene_cond_len = scene_cond_len
        self.dtype = dtype
        self.scene_pos_embed_config = scene_pos_embed_config

        self.t_embedder = TimestepEmbedder(hidden_size, dtype=dtype)
        self.r_embedder = TimestepEmbedder(hidden_size, dtype=dtype) if flow_type == "meanflow" else None
        
        # Trainable sin/cos positions: token mode includes a leading time slot;
        # AdaRMS mode contains only the observed prefix and future actions.
        self.x_pos_embed = nn.Parameter(
            torch.zeros(1, horizon + (time_injection == "tau_token"), hidden_size))
        # Image conditions
        self.scene_cond_pos_embed = nn.Parameter(
            torch.zeros(1, scene_cond_len, hidden_size))
        
        block_class = Block if time_injection == "tau_token" else AdaRMSBlock
        self.blocks = nn.ModuleList([
            block_class(hidden_size, num_heads, self_attn_first) for _ in range(depth)
        ])
        if self.use_final_layer:
            self.final_layer = FinalLayer(hidden_size, output_dim)
        if time_injection == "tau_AdaRMS":
            self.action_time_fusion = nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.GELU(approximate="tanh"),
                nn.Linear(hidden_size, hidden_size),
            )
            if self.use_final_layer:
                self.final_modulation = nn.Sequential(
                    nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
                )
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize pos_embed by sin-cos embedding
        x_pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict[Literal['timestep', 'action'], int]([
                ('timestep', 1),
                ('action', self.horizon),
            ])
        )
        if self.time_injection == "tau_AdaRMS":
            # Preserve the old state/action position slots, omitting the time token.
            x_pos_embed = x_pos_embed[1:]
        self.x_pos_embed.data.copy_(torch.from_numpy(x_pos_embed).float().unsqueeze(0))

        scene_cond_pos_embed = get_multimodal_cond_pos_embed(
            embed_dim=self.hidden_size,
            mm_cond_lens=OrderedDict(self.scene_pos_embed_config),
            embed_modality=False
        )
        self.scene_cond_pos_embed.data.copy_(
            torch.from_numpy(scene_cond_pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.r_embedder is not None:
            nn.init.normal_(self.r_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.r_embedder.mlp[2].weight, std=0.02)
        if self.time_injection == "tau_AdaRMS":
            for block in self.blocks:
                block.initialize_modulations()
            if self.use_final_layer:
                nn.init.zeros_(self.final_modulation[1].weight)
                nn.init.zeros_(self.final_modulation[1].bias)
        
        # Initialize the final layer: zero-out the final linear layer
        if self.use_final_layer:
            nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
            nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)
        
        # Move all the params to given data type:
        self.to(self.dtype)

    def prepare_kv_cache(self, scene_c):
        """Return per-layer K/V for one prediction; never store it on the model."""
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("K/V caching requires eval mode with gradients disabled")
        scene_c = scene_c + self.scene_cond_pos_embed
        return tuple(block.cross_attn.prepare_kv(scene_c) for block in self.blocks)

    def forward(self, x, t, scene_c, scene_mask=None, kv_cache=None, r=None):
        """Predict all horizon slots; the policy masks the observed prefix.

        x: [B, horizon, D], already encoded state and noisy action tokens.
        t/r: [B] or [1]; MeanFlow supplies continuous t and r, FM supplies t.
        scene_c: [B, scene_cond_len, D], or None when using a K/V cache.
        scene_mask: valid visual condition positions, if provided.
        Token mode adds one time slot internally and discards it at output;
        AdaRMS mode uses horizon slots throughout the network.
        """
        t = self.t_embedder(t)
        if self.r_embedder is not None:
            if r is None:
                raise ValueError("MeanFlow DiT requires continuous target time r")
            t = t + self.r_embedder(r)
        elif r is not None:
            raise ValueError("FM DiT does not accept MeanFlow target time r")
        if t.shape[0] == 1:
            t = t.expand(x.shape[0], -1)
        if self.time_injection == "tau_token":
            # Preserve the legacy sequence and module path exactly.
            x = torch.cat([t.unsqueeze(1), x], dim=1)
        else:
            prefix, action = x[:, :self.n_obs_steps], x[:, self.n_obs_steps:]
            action_time = t[:, None, :].expand(-1, action.shape[1], -1)
            action = self.action_time_fusion(torch.cat([action, action_time], dim=-1))
            x = torch.cat([prefix, action], dim=1)
        
        # Add multimodal position embeddings
        x = x + self.x_pos_embed
        if kv_cache is None:
            scene_c = scene_c + self.scene_cond_pos_embed
        else:
            if self.training or torch.is_grad_enabled():
                raise RuntimeError("K/V caching requires eval mode with gradients disabled")
            if len(kv_cache) != len(self.blocks):
                raise ValueError("K/V cache must contain one entry per DiT block")

        # Forward pass
        for i, block in enumerate(self.blocks):
            if self.time_injection == "tau_AdaRMS":
                cache = None if kv_cache is None else kv_cache[i]
                x = block(x, scene_c, scene_mask, kv_cache=cache, time_condition=t)
            elif kv_cache is None:
                x = block(x, scene_c, scene_mask)       # (B, T+1, D)
            else:
                x = block(x, scene_c, scene_mask, kv_cache=kv_cache[i])
        # Final normalization and velocity decoder.
        if self.use_final_layer:
            if self.time_injection == "tau_AdaRMS":
                scale, shift = self.final_modulation(t).chunk(2, dim=-1)
                x = self.final_layer.norm_final(x) * (1 + scale[:, None, :]) + shift[:, None, :]
                x = self.final_layer.ffn_final(x)
            else:
                x = self.final_layer(x)                     # (B, T+1, out_channels)

        # Drop only the optional time token. State slots remain here and are
        # excluded from the direct action loss/sampling output by the policy.
        x = x[:, -self.horizon:]
        return x
