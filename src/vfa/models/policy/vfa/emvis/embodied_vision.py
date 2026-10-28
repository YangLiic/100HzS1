import logging
import time
from typing import Dict, List
import torch
from torch import Tensor, nn
from ..common.module_attr_mixin import ModuleAttrMixin

from .utils import format_number, preprocess_images, param_count
from .adapter import ResNetAdapter
from .mv_fuser import MultiViewFuser

logger = logging.getLogger(__name__)


class VisualEncoder(ModuleAttrMixin):
    def __init__(
        self,
        emvis_config: dict = None,
        shape_meta: dict = None,
        rgb_key: list[str] = ['head_cam'],
        **kwargs
    ):
        super().__init__()
        self.rgb_key = rgb_key
        self.scene_encoder = EmVisRM(**emvis_config)
        # update emvis_config after initialization
        if 'load_backbone_pretrain' in emvis_config:
            emvis_config['load_backbone_pretrain'] = False

        self.out_view_meta = 1 if self.scene_encoder.only_first_view else len(rgb_key)
        self.out_shape_meta = self.scene_encoder.shape_out
        self.out_dim_meta = self.scene_encoder.dim_out

    def forward(self, obs_dict):
        features = list()
        BS = next(iter(obs_dict.values())).shape[0]
        dim = self.scene_encoder.dim_out

        # process rgb input
        rgb_image = torch.cat([obs_dict[key].unsqueeze(1) for key in self.rgb_key], dim=1) # BS, V, C, H, W
        emvis_feat = self.scene_encoder(rgb_image) # BS, V, C, H, W -> BS, V, 1, dim
        out_format = (BS, -1, dim) # BS, V*P, dim / BS, dim
        emvis_feat = emvis_feat.reshape(*out_format)
        features.append(emvis_feat)

        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result

    def output_shape_meta(self):
        """
        return:
            Views, H, W, Channels
        """
        return torch.Size([self.out_view_meta, *self.out_shape_meta, self.out_dim_meta])


class EmVisRM(nn.Module):
    """Spatial backbone -> channel alignment -> MultiViewFuser -> ResNetAdapter.

    DINOv3 ConvNeXt keeps its original state keys and pixel processing.
    EfficientViT native mode fuses/compresses at the trunk width. Projected mode
    remains the default for configs/checkpoints created before native support.
    """

    def __init__(
        self,
        dim_2d: int = 1024, # image embedding dimension
        backbone_target: str = 'dinov3cnn',
        load_backbone_pretrain: bool = False,
        backbone_path = None,
        mlp_ratio = 4.0,
        drop_p = 0.,
        ffn_layer_num = 1,
        # preprocessing config
        processor_config: Dict = {},
        interpolate: str = 'bilinear', # None nearest bilinear(default) bicubic
        # feature config
        only_2d: bool = False,
        only_first_view: bool = False,
        # backbone config
        ft_layer_idx: List = [],
        backbone_config: Dict = {},
        # module config
        model_adapter_config: Dict = None,
        mv_fuser_config: Dict = None,
        backbone_trainable: bool = False,
        efficientvit_fusion_mode: str = "projected",
        **kwargs
    ):
        super().__init__()
        self.backbone_trainable = backbone_trainable
        self.img_processing = None
        self.dim_2d = dim_2d if dim_2d != None else 1024

        self.backbone_target = backbone_target.lower()
        if 'dinov3' in self.backbone_target and 'cnn' in self.backbone_target:
            # Preserve the original DINOv3 initialization and checkpoint layout.
            from .encoder.dinov3cnn_encoder import DINOv3Encoder
            from transformers import DINOv3ConvNextConfig as DINOv3Config
            from transformers import DINOv3ViTImageProcessorFast
            from transformers import AutoImageProcessor, AutoConfig
            if backbone_config:
                config = DINOv3Config(**backbone_config)
            else:
                config = AutoConfig.from_pretrained(f"{backbone_path}")

            self.backbone = DINOv3Encoder(config)
            self.backbone_projection = nn.Identity()
            load_pretrain = processor_config.pop('load_pretrain', False)
            if load_pretrain:
                do_resize = processor_config.pop('do_resize', False)
                do_rescale = processor_config.pop('do_rescale', False)
                processor = AutoImageProcessor.from_pretrained(
                    backbone_path, do_resize=do_resize, do_rescale=do_rescale,
                )
                processor_config.update(processor.to_dict())
            else:
                processor = DINOv3ViTImageProcessorFast(**processor_config)
        elif self.backbone_target == 'efficientvit':
            from .encoder.efficientvit_encoder import EfficientViTEncoder
            from transformers import DINOv3ViTImageProcessorFast

            if efficientvit_fusion_mode not in ("projected", "native"):
                raise ValueError(
                    "efficientvit_fusion_mode must be 'projected' or 'native', "
                    f"got {efficientvit_fusion_mode!r}"
                )
            self.efficientvit_fusion_mode = efficientvit_fusion_mode
            self.backbone = EfficientViTEncoder(
                **backbone_config, frozen=not self.backbone_trainable,
            )
            if self.efficientvit_fusion_mode == "native":
                self.dim_2d = self.backbone.out_channels
                self.backbone_projection = nn.Identity()
            else:
                # Preserve old tensor shapes, state keys and optimizer ordering.
                self.backbone_projection = nn.Linear(self.backbone.out_channels, self.dim_2d)
            # Use the same geometric preprocessing and ImageNet normalization
            # as DINOv3, with no square crop to the classifier's nominal size.
            processor_config.pop('load_pretrain', None)
            processor_config.setdefault('do_resize', False)
            processor_config.setdefault('do_rescale', False)
            processor = DINOv3ViTImageProcessorFast(**processor_config)
            processor_config.update(processor.to_dict())
        else:
            raise ValueError(f"Unsupported visual backbone: {backbone_target!r}")
        self.img_processing = lambda x: processor(
            preprocess_images(x, interpolate=interpolate, patch_size=self.backbone.patch_size_log),
            return_tensors="pt",
        )['pixel_values']

        if load_backbone_pretrain:
            self.backbone.load_pretrained_model(
                backbone_path,
                freeze=not self.backbone_trainable,
            )

        self.only_first_view = only_first_view
        assert only_2d, "only_2d=True (2D token flow) is the only supported mode"
        self.only_2d = only_2d
        dim_now = self.dim_2d

        if self.backbone_target == "efficientvit" and self.efficientvit_fusion_mode == "native":
            for name, config in (("mv_fuser_config", mv_fuser_config),
                                 ("model_adapter_config", model_adapter_config)):
                if config and config.get("dim_in", dim_now) != dim_now:
                    raise ValueError(
                        f"{name}.dim_in must match native EfficientViT width {dim_now}"
                    )

        # initialize multiview fuser
        if mv_fuser_config:
            mv_fuser_config = {
                'dim_in': dim_now,
                'mlp_ratio': mlp_ratio,
                'drop_p': drop_p,
                'ffn_layer_num': ffn_layer_num,
                **mv_fuser_config
            }
            self.mv_fuser = MultiViewFuser(**mv_fuser_config)
        else:
            self.mv_fuser = None

        # initialize model adapter (ResNetAdapter only)
        self.shape_out = None
        self.model_adapter = None
        if model_adapter_config:
            model_adapter_config = {
                'dim_in': dim_now,
                **model_adapter_config
            }
            assert model_adapter_config.get('target', 'ResNetAdapter') == 'ResNetAdapter', (
                "model_adapter must be ResNetAdapter"
            )
            self.model_adapter = ResNetAdapter(
                dim_in=model_adapter_config['dim_in'],
                dim_out=model_adapter_config['dim_out'],
                shape_out=model_adapter_config['shape_out'],
                conv_num=model_adapter_config.get('conv_num'),
                **{k: v for k, v in model_adapter_config.items()
                   if k not in ('dim_in', 'dim_out', 'shape_out', 'conv_num', 'target')}
            )
            dim_now = model_adapter_config['dim_out']
            self.shape_out = model_adapter_config['shape_out']
        self.dim_out = dim_now
        if self.backbone_target == "efficientvit":
            logger.info(
                "EfficientViT fusion mode=%s, native width=%d, fusion width=%d, output width=%d",
                self.efficientvit_fusion_mode, self.backbone.out_channels, self.dim_2d, self.dim_out,
            )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        trainable_ratio = (trainable_params / total_params) * 100 if total_params > 0 else 0.0
        params_state = '\n'.join(['  ' + t for t in param_count(self).split('\n')])
        log_msg = (
            f"EmVisRM Parameters:\n"
            f"  Total: {format_number(total_params)} ({total_params:,})\n"
            f"  Trainable: {format_number(trainable_params)} ({trainable_params:,})\n"
            f"  Frozen: {format_number(frozen_params)} ({frozen_params:,})\n"
            f"  Trainable Ratio: {trainable_ratio:.2f}%\n"
            f"{params_state}"
        )
        logger.info(log_msg)

    def image_forward(self, origin_images: Tensor):
        # images: [B * S, V, 3, H, W] RGB tensor
        BS, V, _, H_origin, W_origin = origin_images.size()
        images = self.img_processing(origin_images.view(-1, 3, H_origin, W_origin))
        _, _, H, W = images.size()
        images = images.view(BS, V, 3, H, W)

        token_dict = self.backbone(images)
        token_dict['image_tokens'] = self.backbone_projection(token_dict['image_tokens'])
        return self.token_forward(token_dict)

    def token_forward(self, token_dict: Dict):
        # 2D-only token flow: image_tokens -> mv_fuser -> ResNetAdapter
        time0 = time.time()
        scene_tokens = token_dict['image_tokens']       # [B * S, V, P_image, D]
        scene_pos = token_dict['image_tokens_pos']      # [B * S, V, P_image, 2]

        if self.mv_fuser is not None:
            scene_tokens = self.mv_fuser(scene_tokens)

        scene_features = scene_tokens
        if self.model_adapter is not None:
            # Flatten to [B*V, P, C] (adapter takes 3D tokens), restore the view dim afterwards
            B, V, P, C = scene_tokens.shape
            flat = self.model_adapter(
                scene_tokens.reshape(B * V, P, C),
                scene_pos.reshape(B * V, P, 2),
            )
            scene_features = flat.reshape(B, V, *flat.shape[-2:])
        time1 = time.time()
        logger.debug(f"Token forward time: {time1 - time0:.5f} seconds")
        return scene_features

    def forward(self, images: Tensor = None, image_tokens: Tensor = None, image_pos: Tensor = None, **kwargs):
        if images is not None:
            if len(images.shape) > 5:  # images: [B, S, V, 3, H, W]
                images = images.view((-1, *images.shape[-4:]))  # images: [B * S, V, 3, H, W]
            return self.image_forward(images)
        # Tokens supplied externally (open-loop / ONNX path)
        return self.token_forward({'image_tokens': image_tokens, 'image_tokens_pos': image_pos})
