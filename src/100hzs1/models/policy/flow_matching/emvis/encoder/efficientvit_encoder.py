"""MIT EfficientViT spatial features with 100HzS1's multiview token contract."""
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn


class EfficientViTEncoder(nn.Module):
    """Spatial trunk with optional shared-stage or independent MBConv downsampling."""

    patch_size_log = 32

    def __init__(self, model_name="efficientvit_b2.r256_in1k", frozen=True,
                 stage4_repeats=0, mbconv_downsample_blocks=0,
                 mbconv_downsample_init="stage4_local", stage4_repeat_position="backbone"):
        super().__init__()
        import timm

        if not model_name.startswith("efficientvit_b"):
            raise ValueError("Use a MIT EfficientViT B-series timm model name")
        self.model_name = model_name
        self.frozen = bool(frozen)
        if type(stage4_repeats) is not int or stage4_repeats < 0:
            raise ValueError("stage4_repeats must be a nonnegative integer")
        self.stage4_repeats = stage4_repeats
        if stage4_repeat_position not in ("backbone", "post_fusion"):
            raise ValueError("stage4_repeat_position must be 'backbone' or 'post_fusion'")
        if stage4_repeat_position == "post_fusion" and not stage4_repeats:
            raise ValueError("post_fusion requires positive stage4_repeats")
        self.stage4_repeat_position = stage4_repeat_position
        if type(mbconv_downsample_blocks) is not int or mbconv_downsample_blocks < 0:
            raise ValueError("mbconv_downsample_blocks must be a nonnegative integer")
        if stage4_repeats and mbconv_downsample_blocks:
            raise ValueError("Choose stage4 repeats or independent MBConv downsampling, not both")
        if mbconv_downsample_init != "stage4_local":
            raise ValueError("mbconv_downsample_init must be 'stage4_local'")
        self.mbconv_downsample_blocks = mbconv_downsample_blocks
        self.mbconv_downsample_init = mbconv_downsample_init
        # No implicit network request: pretrained weights are loaded explicitly.
        self.model = timm.create_model(model_name, pretrained=False)
        if not type(self.model).__module__.endswith("efficientvit_mit"):
            raise ValueError(f"Expected MIT EfficientViT, got {type(self.model)}")
        self.out_channels = int(self.model.num_features)
        self.model.head = nn.Identity()
        self._position_cache = {}
        self.model.requires_grad_(not self.frozen)
        self.stage4_repeat_projection = None
        if self.stage4_repeats:
            # The B0 stage4 downsamples 64->128. Its 128-channel output needs
            # a channel-only bridge before calling the SAME stage again.
            # The bridge is shared across repeats, trainable even for a frozen
            # pretrained trunk, and adds no spatial pooling or activation.
            stage4_in = self.model.stages[-1].blocks[0].main.inverted_conv.conv.in_channels
            self.stage4_repeat_projection = nn.Conv2d(self.out_channels, stage4_in, 1, bias=False)
        self.mbconv_downsample_layers = nn.ModuleList()
        if self.mbconv_downsample_blocks:
            sources = self.model.stages[-1].blocks[1:]
            if self.mbconv_downsample_blocks > len(sources):
                raise ValueError("Not enough stage4 local MBConv blocks for independent initialization")
            for source in sources[:self.mbconv_downsample_blocks]:
                # Copy only MBConv, without its stride-1 identity residual or
                # LiteMLA. Each copy owns independent trainable parameters.
                local = source.local_module.main
                if (local.inverted_conv.conv.in_channels != self.out_channels
                        or local.point_conv.conv.out_channels != self.out_channels):
                    raise ValueError("Stage4 local MBConv must preserve the native channel width")
                block = deepcopy(local).requires_grad_(True)
                block.depth_conv.conv.stride = (2, 2)
                self.mbconv_downsample_layers.append(block)
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
        # The constructor precedes explicit pretrained loading. Refresh copied
        # initializers HERE so they receive pretrained, not random trunk weights.
        # Checkpoint reload sets load_backbone_pretrain=False and restores the
        # learned copies strictly instead of reinitializing them.
        for index, block in enumerate(self.mbconv_downsample_layers):
            source = self.model.stages[-1].blocks[index + 1].local_module.main
            block.load_state_dict(source.state_dict(), strict=True)
        self.frozen = bool(freeze)
        self.model.requires_grad_(not self.frozen)
        self.train(self.training)

    def train(self, mode=True):
        # Frozen BatchNorm must retain the pretrained running statistics when
        # the surrounding policy switches to training mode.
        super().train(bool(mode) and not self.frozen)
        if self.stage4_repeat_projection is not None:
            self.stage4_repeat_projection.train(bool(mode))
        self.mbconv_downsample_layers.train(bool(mode))
        return self

    def forward(self, images):
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError(f"Expected [B,V,3,H,W], got {tuple(images.shape)}")
        batch, views, channels, height, width = images.shape
        features = self.model.forward_features(images.reshape(-1, channels, height, width))
        if self.stage4_repeat_position == "backbone":
            features = self.repeat_stage4(features)
        for block in self.mbconv_downsample_layers:
            features = block(features)
        return self._tokens_from_features(features, batch, views)

    def repeat_stage4(self, features):
        # Reuse the original module without aliases/duplicate state keys. Do not
        # use no_grad: post-fusion gradients must reach the fuser and bridge.
        for _ in range(self.stage4_repeats):
            features = self.model.stages[-1](self.stage4_repeat_projection(features))
        return features

    def repeat_stage4_tokens(self, tokens, positions):
        """Compress each fused view's preserved spatial grid, retaining autograd."""
        batch, views, count, dim = tokens.shape
        grid_h = int((positions[0, 0, :, 1] == 0).sum().item())
        grid_w = int((positions[0, 0, :, 0] == 0).sum().item())
        if grid_h * grid_w != count or dim != self.out_channels:
            raise ValueError("Fused tokens must preserve the native EfficientViT spatial grid")
        features = tokens.reshape(batch * views, grid_h, grid_w, dim).permute(0, 3, 1, 2).contiguous()
        return self._tokens_from_features(self.repeat_stage4(features), batch, views)

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
