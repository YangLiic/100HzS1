"""Independent width/compression recipes and true shared-stage4 contracts."""
from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
vision = import_module('100hzs1.models.policy.flow_matching.emvis.embodied_vision')
encoder = import_module('100hzs1.models.policy.flow_matching.emvis.encoder.efficientvit_encoder')
adapter = import_module('100hzs1.models.policy.flow_matching.emvis.adapter')
POLICY = 'backbones/efficientvit_b0/tau_token/100hzs1_beat_block_hammer_efficientvit_b0_'
NAMES = {'A': POLICY+'postfusion_2x2_fm_d1024_3v_8gpu_L4',
         'B': POLICY+'stage4repeat_2x3_fm_d2048_3v_8gpu_L4',
         'C': POLICY+'poststage4_2x2_fm_d2048_3v_8gpu_L4'}


def recipe(arm):
    OmegaConf.register_new_resolver('eval', eval, replace=True)
    with initialize_config_dir(version_base=None, config_dir=str(ROOT/'configs/100hzs1_train')):
        return OmegaConf.to_container(compose(config_name=NAMES[arm]), resolve=True)


def scene(arm):
    cfg = deepcopy(recipe(arm)['model']['obs_encoder']['emvis_config'])
    cfg['load_backbone_pretrain'] = False
    return vision.EmVisRM(**cfg)


class VisualAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_two_recipes_preserve_common_training_recipe(self):
        a, b = recipe('A'), recipe('B')
        for key in ['train_dataset', 'seed', 'norm', 'num_train_epochs', 'train_batch_size',
                    'optimizer', 'horizon', 'n_obs_steps', 'n_image_obs_steps']:
            self.assertEqual(a[key], b[key], key)
        self.assertEqual(a['num_train_epochs'], 200)
        self.assertEqual(a['train_batch_size'], 128)
        self.assertEqual(a['model']['meanflow'], b['model']['meanflow'])
        for arm, cfg, width, out_width, grid, repeats in [
                ('A', a, 1024, 512, [2,2], 0), ('B', b, 2048, 1024, [2,3], 2)]:
            m = cfg['model'];v = m['obs_encoder']['emvis_config']
            self.assertEqual(m['hidden_size'], width)
            self.assertEqual(m['num_heads'], width//64)
            self.assertEqual(m['scene_adaptor'], 'mlp2x_gelu')
            self.assertEqual(m['state_adaptor'], 'mlp3x_gelu')
            self.assertEqual(m['time_injection'], 'tau_token')
            self.assertEqual(m['flow_type'], 'fm')
            self.assertEqual(cfg['inference']['num_inference_timesteps'], 4)
            self.assertEqual(m['noise_scheduler']['num_train_timesteps'], 1000)
            self.assertEqual(v['model_adapter_config']['shape_out'], grid)
            self.assertEqual(v['model_adapter_config']['dim_out'], out_width)
            self.assertEqual(v['backbone_config']['stage4_repeats'], repeats)
            self.assertFalse(v['backbone_trainable'])
            self.assertFalse(v['compress_before_fusion'])
            self.assertEqual(v['output_projection_type'], 'linear')
        self.assertNotEqual(a['output_dir'], b['output_dir'])

    def test_shared_stage4_changes_grid_and_trains_only_bridge(self):
        m = encoder.EfficientViTEncoder('efficientvit_b0.r224_in1k', stage4_repeats=2).train()
        calls = []
        handle = m.model.stages[-1].register_forward_hook(
            lambda module,args,out: calls.append((tuple(args[0].shape),tuple(out.shape))))
        bn = next(x for x in m.model.modules() if isinstance(x, nn.BatchNorm2d))
        mean = bn.running_mean.clone()
        try:
            output = m(torch.rand(1, 3, 3, 256, 320))
            self.assertEqual(output['image_tokens'].shape, (1,3,6,128))
            self.assertEqual(output['image_tokens_pos'][0,0,-1].tolist(), [1,2])
            self.assertEqual(calls, [((3,64,16,20),(3,128,8,10)),
                                     ((3,64,8,10),(3,128,4,5)),
                                     ((3,64,4,5),(3,128,2,3))])
            output['image_tokens'].square().mean().backward()
            grad = m.stage4_repeat_projection.weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in m.model.parameters()))
            self.assertFalse(bn.training)
            torch.testing.assert_close(mean, bn.running_mean, rtol=0, atol=0)
            names = [name for name,_ in m.named_parameters()]
            self.assertEqual([n for n in names if not n.startswith('model.')],
                             ['stage4_repeat_projection.weight'])
        finally:
            handle.remove()

    def test_fusion_receives_original_grid_for_a_and_compressed_grid_for_b(self):
        for arm, fusion_tokens, output_tokens, width in [('A',80,4,512),('B',6,6,1024)]:
            with self.subTest(arm=arm):
                m = scene(arm).eval();seen=[]
                handle=m.mv_fuser.register_forward_pre_hook(lambda module,args:seen.append(tuple(args[0].shape)))
                try:
                    with torch.no_grad():out=m(torch.rand(1,3,3,240,320))
                finally:
                    handle.remove()
                self.assertEqual(seen,[(1,3,fusion_tokens,128)])
                self.assertEqual(out.shape,(1,3,output_tokens,width))
                if arm=='B':
                    self.assertIsInstance(m.model_adapter,adapter.TokenProjection)
                    self.assertEqual((m.model_adapter.proj.in_features,m.model_adapter.proj.out_features),(128,1024))
                    self.assertFalse(any(isinstance(x,(adapter.ResBlock,adapter.AdaptivePatchMerging))
                                         for x in m.model_adapter.modules()))

    def test_new_visual_config_strict_json_reload_preserves_shared_repeat_output(self):
        cfg=deepcopy(recipe('B')['model']['obs_encoder']['emvis_config'])
        cfg['load_backbone_pretrain']=False
        first=vision.VisualEncoder(emvis_config=cfg,rgb_key=['left_cam','right_cam','head_cam']).eval()
        restored=vision.VisualEncoder(emvis_config=json.loads(json.dumps(cfg)),rgb_key=first.rgb_key).eval()
        restored.load_state_dict(first.state_dict(),strict=True)
        self.assertEqual(restored.output_shape_meta(),torch.Size([3,2,3,1024]))
        obs={key:torch.rand(1,3,240,320) for key in first.rgb_key}
        with torch.no_grad():torch.testing.assert_close(restored(obs),first(obs),rtol=0,atol=0)

    def test_postfusion_repeat_replaces_resblocks_but_preserves_pool_and_gradients(self):
        m=scene('C').train();events=[];shapes=[]
        hooks=[m.mv_fuser.register_forward_pre_hook(lambda mod,args:events.append(('fusion',tuple(args[0].shape)))),
               m.backbone.model.stages[-1].register_forward_hook(lambda mod,args,out:events.append(('stage4',tuple(out.shape)))),
               m.model_adapter.patch_merging.register_forward_pre_hook(lambda mod,args:shapes.append(tuple(args[0].shape)))]
        bn_stats={n:b.running_mean.clone() for n,b in m.backbone.model.named_modules() if isinstance(b,nn.BatchNorm2d)}
        try:
            out=m(torch.rand(1,3,3,240,320))
            self.assertEqual(events,[('stage4',(3,128,8,10)),('fusion',(1,3,80,128)),
                                     ('stage4',(3,128,4,5)),('stage4',(3,128,2,3))])
            self.assertEqual(shapes,[(3,128,2,3)])
            self.assertEqual(out.shape,(1,3,4,1024))
            self.assertIsInstance(m.model_adapter,adapter.ResNetAdapter)
            self.assertEqual(len(m.model_adapter.downsample_blocks),0)
            self.assertIs(m.model_adapter.patch_merging.adapter,nn.AdaptiveAvgPool2d)
            self.assertEqual((m.model_adapter.patch_merging.proj.in_features,m.model_adapter.patch_merging.proj.out_features),(128,1024))
            self.assertFalse(any(isinstance(mod,adapter.ResBlock) for mod in m.modules()))
            out.square().mean().backward()
            for param in [m.backbone.stage4_repeat_projection.weight,next(m.mv_fuser.parameters())]:
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())
                self.assertGreater(param.grad.abs().sum().item(),0)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in m.backbone.model.parameters()))
            for n,b in m.backbone.model.named_modules():
                if isinstance(b,nn.BatchNorm2d):
                    self.assertFalse(b.training);torch.testing.assert_close(b.running_mean,bn_stats[n],rtol=0,atol=0)
            self.assertEqual([n for n in m.state_dict() if 'stage4_repeat_projection' in n],['backbone.stage4_repeat_projection.weight'])
            self.assertEqual(sum(p.numel() for p in m.backbone.stage4_repeat_projection.parameters()),8192)
            # Exact pooling/projection contract, including the overlapping width bins.
            features=torch.randn(3,128,2,3)
            expected=m.model_adapter.patch_merging.proj(nn.functional.adaptive_avg_pool2d(features,(2,2)).permute(0,2,3,1))
            torch.testing.assert_close(m.model_adapter.patch_merging(features,(2,2)),expected,rtol=0,atol=0)
        finally:
            for hook in hooks:hook.remove()

    def test_postfusion_json_reload_and_invalid_double_compression(self):
        cfg=deepcopy(recipe('C')['model']['obs_encoder']['emvis_config']);cfg['load_backbone_pretrain']=False
        first=vision.VisualEncoder(emvis_config=cfg,rgb_key=['left_cam','right_cam','head_cam']).eval()
        second=vision.VisualEncoder(emvis_config=json.loads(json.dumps(cfg)),rgb_key=first.rgb_key).eval()
        second.load_state_dict(first.state_dict(),strict=True)
        self.assertEqual(second.output_shape_meta(),torch.Size([3,2,2,1024]))
        obs={key:torch.rand(1,3,240,320) for key in first.rgb_key}
        with torch.no_grad():torch.testing.assert_close(second(obs),first(obs),rtol=0,atol=0)
        cfg['model_adapter_config']['conv_num']=2
        with self.assertRaisesRegex(ValueError,'pool-only'):vision.EmVisRM(**cfg)
        for position,repeats in [('bad',2),('post_fusion',0)]:
            with self.assertRaises(ValueError):encoder.EfficientViTEncoder('efficientvit_b0.r224_in1k',stage4_repeats=repeats,stage4_repeat_position=position)

    def test_legacy_encoder_has_no_extra_state_and_rejects_bad_repeat_counts(self):
        m=encoder.EfficientViTEncoder('efficientvit_b0.r224_in1k')
        self.assertIsNone(m.stage4_repeat_projection)
        self.assertFalse(any('stage4_repeat' in k for k in m.state_dict()))
        for invalid in [-1,True,1.5]:
            with self.assertRaisesRegex(ValueError,'nonnegative integer'):
                encoder.EfficientViTEncoder('efficientvit_b0.r224_in1k',stage4_repeats=invalid)


if __name__=='__main__':unittest.main()
