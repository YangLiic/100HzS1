"""MIT EfficientViT spatial features with 100HzS1's multiview token contract."""
from pathlib import Path

import torch
from torch import nn


class EfficientViTEncoder(nn.Module):
    """Keep the last stride-32 spatial map, excluding the classification head."""

    patch_size_log = 32

    def __init__(self, model_name="efficientvit_b2.r256_in1k", frozen=True):
        super().__init__()
        import timm

        if not model_name.startswith("efficientvit_b"):
            raise ValueError("Use a MIT EfficientViT B-series timm model name")
        self.model_name = model_name
        self.frozen = bool(frozen)
        # No implicit network request: pretrained weights are loaded explicitly.
        self.model = timm.create_model(model_name, pretrained=False)
        if not type(self.model).__module__.endswith("efficientvit_mit"):
            raise ValueError(f"Expected MIT EfficientViT, got {type(self.model)}")
        self.out_channels = int(self.model.num_features)
        self.model.head = nn.Identity()
        self._position_cache = {}
        self.requires_grad_(not self.frozen)
        self.train(not self.frozen)

    def load_pretrained_model(self, model_name, freeze=True):
        from safetensors.torch import load_file

        path = Path(model_name)
        if path.is_dir():
            path = path / "model.safetensors"
        if path.suffix != ".safetensors" or not path.is_file():
            raise ValueError(f"Expected a local EfficientViT model.safetensors: {path}")
        state = load_file(str(path), device="cpu")
        # timm checkpoints include a classifier; 100HzS1 uses only its spatial trunk.
        state = {key: value for key, value in state.items() if not key.startswith("head.")}
        self.model.load_state_dict(state, strict=True)
        self.frozen = bool(freeze)
        self.requires_grad_(not self.frozen)
        self.train(self.training)

    def train(self, mode=True):
        # Frozen BatchNorm must retain the pretrained running statistics when
        # the surrounding policy switches to training mode.
        return super().train(bool(mode) and not self.frozen)

    def forward(self, images):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"Expected [B,V,3,H,W], got {tuple(images.shape)}")
        batch, views, channels, height, width = images.shape
        features = self.model.forward_features(images.reshape(-1, channels, height, width))
        _, dim, grid_h, grid_w = features.shape
        tokens = features.flatten(2).transpose(1, 2).reshape(batch, views, grid_h * grid_w, dim)
        key = (grid_h, grid_w, features.device.type, features.device.index)
        if key not in self._position_cache:
            self._position_cache[key] = torch.cartesian_prod(
                torch.arange(grid_h, device=features.device),
                torch.arange(grid_w, device=features.device),
            )
        positions = self._position_cache[key].view(1, 1, -1, 2).expand(batch, views, -1, -1)
        return {
            "image_tokens": tokens,
            "image_tokens_pos": positions,
            "camera_tokens_list": None,
            "spatial_tokens_pos": positions,
            "spatial_tokens_list": None,
        }
