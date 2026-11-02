import torch
from torch import Tensor, nn


class RMSNormTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """PreNorm encoder with RMSNorm and the original attention/FFN parameters.

    The parent inference fast path implements LayerNorm internally. Always use
    the explicit residual path here, including in eval/inference mode, so that
    training and inference execute the same normalization operation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.norm_first:
            raise ValueError("RMSNorm fusion requires norm_first=True")
        width = self.self_attn.embed_dim
        factory = dict(device=self.norm1.weight.device, dtype=self.norm1.weight.dtype)
        eps = self.norm1.eps
        self.norm1 = nn.RMSNorm(width, eps=eps, **factory)
        self.norm2 = nn.RMSNorm(width, eps=eps, **factory)

    def forward(self, src: Tensor, src_mask: Tensor = None,
                src_key_padding_mask: Tensor = None, is_causal: bool = False) -> Tensor:
        x = src + self._sa_block(
            self.norm1(src), src_mask, src_key_padding_mask, is_causal=is_causal,
        )
        return x + self._ff_block(self.norm2(x))


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
        norm_type: str = "layernorm",
        ** kwargs
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"pre-adapter fusion depth must be >= 1, got {depth}")

        if dim_out is not None and dim_out != dim_in:
            raise ValueError(
                f"pre-adapter fusion does not change dim: dim_in={dim_in}, dim_out={dim_out}"
            )
        if norm_type not in ("layernorm", "rmsnorm"):
            raise ValueError(f"fusion norm_type must be 'layernorm' or 'rmsnorm', got {norm_type!r}")
        # Missing metadata preserves the original LayerNorm checkpoint layout.
        self.norm_type = norm_type
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
            layer = (nn.TransformerEncoderLayer if norm_type == "layernorm"
                     else RMSNormTransformerEncoderLayer)
            return layer(
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
        self.norm = (nn.LayerNorm(dim) if norm_type == "layernorm"
                     else nn.RMSNorm(dim, eps=1e-5))

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

