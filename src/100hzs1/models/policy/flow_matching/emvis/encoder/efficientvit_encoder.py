"""Frozen EfficientViT features with the supported shared-stage4 compression."""
from pathlib import Path
import torch
from torch import nn
from ...visual_variants import STAGE4_VARIANTS, validate_visual_variant


class EfficientViTEncoder(nn.Module):
    patch_size_log = 32

    def __init__(self, model_name="efficientvit_b2.r256_in1k", frozen=True,
                 visual_variant="baseline"):
        super().__init__()
        import timm
        if not model_name.startswith("efficientvit_b"):
            raise ValueError("Use a MIT EfficientViT B-series timm model name")
        self.model_name = model_name
        self.frozen = bool(frozen)
        self.visual_variant = validate_visual_variant(visual_variant)
        self.shared_stage4 = self.visual_variant in STAGE4_VARIANTS
        if self.shared_stage4 and (not model_name.startswith(("efficientvit_b0.", "efficientvit_b1.")) or not self.frozen):
            raise ValueError("shared stage4 requires a frozen EfficientViT-B0 or B1 backbone")
        self.model = timm.create_model(model_name, pretrained=False)
        if not type(self.model).__module__.endswith("efficientvit_mit"):
            raise ValueError(f"Expected MIT EfficientViT, got {type(self.model)}")
        self.out_channels = int(self.model.num_features)
        self.model.head = nn.Identity()
        self._position_cache = {}
        self.model.requires_grad_(not self.frozen)
        self.stage4_repeat_projection = None
        if self.shared_stage4:
            stage4_in = self.model.stages[-1].blocks[0].main.inverted_conv.conv.in_channels
            self.stage4_repeat_projection = nn.Conv2d(self.out_channels, stage4_in, 1, bias=False)
        self.train(not self.frozen)

    def load_pretrained_model(self, model_name, freeze=True):
        from safetensors.torch import load_file
        path = Path(model_name)
        if path.is_dir():
            path = path / "model.safetensors"
        if path.suffix != ".safetensors" or not path.is_file():
            raise ValueError(f"Expected a local EfficientViT model.safetensors: {path}")
        state = {k: v for k, v in load_file(str(path), device="cpu").items() if not k.startswith("head.")}
        self.model.load_state_dict(state, strict=True)
        self.frozen = bool(freeze)
        if self.shared_stage4 and not self.frozen:
            raise ValueError("shared stage4 keeps the pretrained backbone frozen")
        self.model.requires_grad_(not self.frozen)
        self.train(self.training)

    def train(self, mode=True):
        super().train(bool(mode) and not self.frozen)
        if self.stage4_repeat_projection is not None:
            self.stage4_repeat_projection.train(bool(mode))
        return self

    def forward(self, images):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"Expected [B,V,3,H,W], got {tuple(images.shape)}")
        batch, views, channels, height, width = images.shape
        features = self.model.forward_features(images.reshape(-1, channels, height, width))
        if self.shared_stage4:
            # The SAME bridge and pretrained stage are reused exactly twice.
            # Gradients flow through the frozen stage into the trainable bridge.
            for _ in range(2):
                features = self.model.stages[-1](self.stage4_repeat_projection(features))
        # No pooling here: fusion consumes the natural 2x3 grid.
        return self._tokens_from_features(features, batch, views)

    def _tokens_from_features(self, features, batch, views):
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
