"""VFA visual adapter: ResNetAdapter.

Flow: tokens [B,V,P,C] + patch grid -> ResBlock downsampling chain -> AdaptivePatchMerging to the target grid.
"""
from functools import partial
from typing import List, Tuple
from torch import Tensor, nn
from collections import OrderedDict

from .layers import MyLayerNorm, MyGroupNorm
from torch.nn import GELU as ActFunc


class AdaptivePatchMerging(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        use_pixelshuffle: bool = True,
        drop: float = 0.0,
        bias: bool = True,
        use_max_pool=False,
        ** kwargs
    ) -> None:
        super().__init__()
        if use_pixelshuffle:
            raise NotImplementedError("AdaptivePixelShuffle not implemented (kept only for config compat)")
        else:
            self.adapter = nn.AdaptiveMaxPool2d if use_max_pool else nn.AdaptiveAvgPool2d
            self.proj = nn.Linear(dim_in, dim_out, bias=bias)
            self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor, shape_out: Tuple[int, int]) -> Tensor:
        # x: [B * S * V, C_In, H', W']
        x = self.adapter(shape_out)(x) # [B * S * V, C_In, H'', W'']
        x = x.permute(0, 2, 3, 1) # [B * S * V, H'', W'', C_In]
        x = self.proj(x) # [B * S * V, H'', W'', C_Out]
        x = self.drop(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, dim_in, dim_out, stride=2, use_max_pool=False):
        super(ResBlock, self).__init__()
        self.dim_in, self.dim_out, self.stride = dim_in, dim_out, stride
        self.mainstream = nn.Sequential(
            OrderedDict(self.get_main_layers())
        )
        if use_max_pool:
            self.downsample = nn.Sequential(
                OrderedDict([('maxpool', nn.MaxPool2d(3, stride=stride, padding=1))])
            ) if stride == 2 else nn.Identity()
        else:
            self.downsample = nn.Sequential(
                OrderedDict(self.get_downsample_layers())
            ) if stride == 2 else nn.Identity()
        self.act = ActFunc()

    def forward(self, x):
        out = self.mainstream(x)
        res = self.downsample(x)
        out += res
        out = self.act(out)
        return out

    def get_main_layers(self, groups_num: int = 0):
        dim_in, dim_out, stride = self.dim_in, self.dim_out, self.stride
        conv_list = []
        if groups_num:
            get_norm = partial(MyGroupNorm, group=groups_num)
        else:
            get_norm = partial(MyLayerNorm)

        conv_list += (
            (f'conv1', nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            )),
            (f'norm1', get_norm(dim=dim_out)),
            (f'act', ActFunc()),
            (f'conv2', nn.Conv2d(
                in_channels=dim_out,
                out_channels=dim_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            )),
            (f'norm2', get_norm(dim=dim_out)),
        )
        return conv_list

    def get_downsample_layers(self, groups_num: int = 0):
        dim_in, dim_out, stride = self.dim_in, self.dim_out, self.stride
        conv_list = []
        if groups_num:
            get_norm = partial(MyGroupNorm, group=groups_num)
        else:
            get_norm = partial(MyLayerNorm)

        conv_list += (
            (f'conv', nn.Conv2d(
                in_channels=dim_in,
                out_channels=dim_out,
                kernel_size=1,
                stride=stride,
                bias=False
            )),
            (f'norm', get_norm(dim=dim_out))
        )
        return conv_list


arch_settings = {
    2: ((1, 1), (1, 1)),
    3: ((1, 1, 1), (1, 1, 1)),
    5: ((1, 1, 1, 1, 1), (2, 4, 4, 8, 8)),
    6: ((1, 1, 1, 1, 1, 1), (2, 2, 4, 4, 8, 8)),
}


class ResNetAdapter(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        shape_out: tuple = None,
        conv_num: int = None,
        stage_blocks: List = None,
        dim_ratio: List = None, # expend ratio of `dim_in` in diffrent stage
        groups_num: int = 0,
        use_pixelshuffle: bool = False,
        use_max_pool: bool = False,
        **kwargs
    ) -> None:
        """
        Args:
            shape_out (Tuple[int, Optional[int]]): Output Shape with size 1 or 2.
                (x, y) processed as (x, y)
                (x,) processed as (x,1)
                (x, None) processed as (x,1)
                (None, y) processed as (1,y)
        """
        super().__init__()
        if conv_num != None:
            stage_blocks, dim_ratio = arch_settings[conv_num]
        if not stage_blocks:
            stage_blocks = []
        if not dim_ratio:
            dim_ratio = [2**i for i in range(1, len(stage_blocks) + 1)]
        assert len(stage_blocks) == len(dim_ratio)
        self.shape_out = shape_out
        self.dim_mid = [dim_in]
        self.stride_mid = []
        self.stage_idx = []
        for i, block_num in enumerate(stage_blocks):
            self.dim_mid += [int(dim_ratio[i] * dim_in)] * block_num
            self.stride_mid += [2] + [1] * (block_num - 1)
            self.stage_idx += [i] * (block_num)

        self.downsample_blocks = nn.Sequential(OrderedDict([
            (f'{si}_{bi}', ResBlock(dim_in=di, dim_out=do, stride=std, use_max_pool=use_max_pool))
            for bi, (si, di, do, std) in enumerate(zip(self.stage_idx, self.dim_mid[:-1], self.dim_mid[1:], self.stride_mid))
        ]))
        self.patch_merging = AdaptivePatchMerging(
            self.dim_mid[-1], dim_out, use_pixelshuffle, use_max_pool=use_max_pool, **kwargs
        )

    def forward(
        self,
        x: Tensor,
        xpos: Tensor,
        y: Tensor = None,
        ypos: Tensor = None
    ) -> Tensor:
        # x: [LBSV, H * W, C_In]
        # return: [LBSV, H'' * W'', C_Out]
        B, _, C = x.shape
        H, W = (xpos[0,:,1]==0).sum().item(), (xpos[0,:,0]==0).sum().item()
        if ypos:
            H_T, W_T = (ypos[0,:,1]==0).sum().item(), (ypos[0,:,0]==0).sum().item()
        else:
            assert self.shape_out, "Please input `shape_out` in init function or `y_pos` in forward function"
            H_T, W_T = self.shape_out
        x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous() # [B C_In, H, W]

        x = self.downsample_blocks(x) # [B, C_In, H', W']
        x = self.patch_merging(x, (H_T, W_T)) # [B, H'', W'', C_Out]
        C = x.shape[-1]
        x = x.view(B, -1, C) # [B, *H_W, C_Out]
        return x
