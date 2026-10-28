import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MultiViewFuser(nn.Module):
    """Token fusion over view tokens before the spatial adapter."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int = None,
        hidden: int = None,
        depth: int = 1,
        num_heads: int = 8,
        mlp_ratio: float = 1.0,
        drop_p: float = 0.0,
        use_view_embed: bool = True,
        num_views: int = 3, #TODO: set num_views according to the robot configuration
        ffn_layer_num: int = 1,
        ** kwargs
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"pre-adapter fusion depth must be >= 1, got {depth}")

        if dim_out is not None and dim_out != dim_in:
            raise ValueError(
                f"pre-adapter fusion does not change dim: dim_in={dim_in}, dim_out={dim_out}"
            )
        dim = dim_in
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = dim
        self.num_views = num_views
        self.use_view_embed = use_view_embed

        if use_view_embed:
            self.view_embed = nn.Parameter(torch.zeros(1, num_views, 1, dim))
            nn.init.normal_(self.view_embed, std=0.02)
        else:
            self.view_embed = None

        def make_layer() -> nn.Module:
            return nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=int(dim * mlp_ratio),
                dropout=drop_p,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )

        self.frame_blocks = nn.ModuleList([make_layer() for _ in range(depth)])
        self.global_blocks = nn.ModuleList([make_layer() for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def _alternating_attention(
        self,
        x: Tensor,
        batch_size: int,
        num_views: int,
        num_patches: int,
        dim: int,
    ) -> Tensor:
        for frame_block, global_block in zip(self.frame_blocks, self.global_blocks):
            x = x.reshape(batch_size * num_views, num_patches, dim)
            x = frame_block(x)
            x = x.reshape(batch_size, num_views, num_patches, dim)

            x = x.reshape(batch_size, num_views * num_patches, dim)
            x = global_block(x)
            x = x.reshape(batch_size, num_views, num_patches, dim)
        return x

    def forward(self, scene_tokens: Tensor) -> Tensor:
        if scene_tokens.ndim != 4:
            raise ValueError(f"Expected scene tokens with shape [B, V, P, D], got {scene_tokens.shape}")

        batch_size, num_views, num_patches, dim = scene_tokens.shape
        if dim != self.dim:
            raise ValueError(f"Feature dim mismatch: expected {self.dim}, got {dim}")
        if num_views == 1:
            return scene_tokens
        if num_views > self.num_views:
            raise ValueError(f"num_views ({num_views}) exceeds num_views ({self.num_views})")

        x = scene_tokens
        if self.view_embed is not None:
            #position embedding
            x = x + self.view_embed[:, :num_views].to(device=x.device, dtype=x.dtype)

        fused = self._alternating_attention(x, batch_size, num_views, num_patches, dim)
        fused = self.norm(fused)
        return fused

