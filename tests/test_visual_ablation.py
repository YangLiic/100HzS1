"""Named B0 visual presets, fixed controls and post-fusion pooling."""
from copy import deepcopy
from unittest.mock import patch
from importlib import import_module
import json
from pathlib import Path
import unittest
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
vision=import_module('100hzs1.models.policy.flow_matching.emvis.embodied_vision')
encoder=import_module('100hzs1.models.policy.flow_matching.emvis.encoder.efficientvit_encoder')
policy=import_module('100hzs1.models.policy.flow_matching.dit_fm_policy')
variants=import_module('100hzs1.models.policy.flow_matching.visual_variants')
PREFIX='backbones/efficientvit_b0/tau_token/'

def recipe(variant, pooling=True):
    OmegaConf.register_new_resolver('eval',eval,replace=True)
    with initialize_config_dir(version_base=None,config_dir=str(ROOT/'configs/100hzs1_train')):
        return OmegaConf.to_container(compose(config_name=PREFIX+'b0_'+variant+'_fm4',overrides=['model.visual_pooling='+str(pooling).lower()]),resolve=True)

def scene(variant, pooling=True):
    cfg=deepcopy(recipe(variant,pooling)['model']['obs_encoder']['emvis_config'])
    cfg['load_backbone_pretrain']=False
    return vision.EmVisRM(**cfg)

class VisualAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_named_presets_and_common_training_settings(self):
        files={p.name for p in (ROOT/'configs/100hzs1_train/backbones/efficientvit_b0').rglob('*.yaml')}
        self.assertEqual(files,{'b0_'+v+'_fm4.yaml' for v in variants.VISUAL_VARIANTS})
        configs={v:recipe(v) for v in variants.VISUAL_VARIANTS};base=configs['baseline']
        for name,cfg in configs.items():
            for key in ['norm','seed','num_train_epochs','train_batch_size','optimizer','horizon',
                        'n_obs_steps','n_image_obs_steps','train_dataset','inference']:
                self.assertEqual(cfg[key],base[key],(name,key))
            m=cfg['model'];v=m['obs_encoder']['emvis_config'];s=name in variants.STAGE4_VARIANTS;r=name in variants.RMS_VARIANTS
            self.assertEqual((m['hidden_size'],m['num_heads'],m['depth']),(1024,16,8) if r or name in variants.PLAIN_1024_VARIANTS or name=='stage4_resmlp_no_norm' else (2048,32,8))
            self.assertEqual(m['visual_variant'],name);self.assertEqual(v['visual_variant'],name)
            self.assertEqual(m['scene_adaptor'],'residual_mlp2x_gelu' if r else 'residual_mlp2x_gelu_no_norm' if name=='stage4_resmlp_no_norm' else 'mlp2x_gelu')
            self.assertEqual(m['state_adaptor'],'mlp3x_gelu');self.assertEqual(m['flow_type'],'fm')
            self.assertEqual(m['noise_scheduler']['num_train_timesteps'],1000)
            self.assertEqual(cfg['inference']['num_inference_timesteps'],4)
            self.assertEqual(v['model_adapter_config']['conv_num'],None if s else 2)
            self.assertEqual(v['model_adapter_config']['dim_out'],512 if name in variants.PLAIN_1024_VARIANTS else 1024)
            self.assertEqual(v['model_adapter_config']['shape_out'],[2,2])
            self.assertEqual(set(v['backbone_config']),{'model_name'})
            for key in ['ffn_hidden_size','scene_adaptor_hidden_size']:self.assertNotIn(key,m)
            for key in variants.RETIRED_VISION_OPTIONS:self.assertNotIn(key,v)
            policy.S1DitFlowMatchingConfig(**deepcopy(m))
        self.assertEqual(len({c['output_dir'] for c in configs.values()}),len(variants.VISUAL_VARIANTS))

    def test_only_variant_and_documented_pool_defaults_differ_in_live_yaml_overrides(self):
        import yaml
        root=ROOT/'configs/100hzs1_train/backbones/efficientvit_b0/tau_token'
        for name in ['stage4','rms','stage4_rms']:
            raw=yaml.safe_load((root/('b0_'+name+'_fm4.yaml')).read_text())
            expected={'visual_variant':name}
            if name in variants.STAGE4_VARIANTS:expected['visual_pooling']=False
            if name == 'stage4_rms':
                expected['obs_encoder']={'emvis_config':{'mv_fuser_config':{'norm_type':'rmsnorm'}}}
            self.assertEqual(raw['model'],expected)

    def test_plain_mlp1024_preset_retains_shared_compression_and_rejects_rms(self):
        cfg=recipe('stage4_mlp1024',False)['model']
        self.assertEqual(cfg['obs_encoder']['emvis_config']['model_adapter_config']['dim_out'],512)
        self.assertEqual((cfg['hidden_size'],cfg['num_heads'],cfg['scene_adaptor']),(1024,16,'mlp2x_gelu'))
        policy.S1DitFlowMatchingConfig(**deepcopy(cfg))
        mlp=policy.S1DitFlowMatching.build_condition_adapter(None,'mlp2x_gelu',512,1024)
        self.assertIsInstance(mlp,nn.Sequential)
        self.assertEqual([type(x) for x in mlp],[nn.Linear,nn.GELU,nn.Linear])
        self.assertEqual([(x.in_features,x.out_features) for x in mlp if isinstance(x,nn.Linear)],[(512,1024),(1024,1024)])
        for key,value in [('hidden_size',2048),('scene_adaptor','residual_mlp2x_gelu')]:
            bad=deepcopy(cfg);bad[key]=value
            with self.assertRaises(ValueError):policy.S1DitFlowMatchingConfig(**bad)
        bad=deepcopy(cfg);bad['obs_encoder']['emvis_config']['model_adapter_config']['dim_out']=1024
        with self.assertRaises(ValueError):policy.S1DitFlowMatchingConfig(**bad)

    def test_shared_stage4_is_reused_twice_with_one_trainable_bridge(self):
        m=encoder.EfficientViTEncoder('efficientvit_b0.r224_in1k',visual_variant='stage4').train()
        calls=[];stage=m.model.stages[-1]
        h=stage.register_forward_hook(lambda mod,args,out:calls.append((tuple(args[0].shape),tuple(out.shape))))
        before={n:x.running_mean.clone() for n,x in m.model.named_modules() if isinstance(x,nn.BatchNorm2d)}
        try:
            out=m(torch.rand(1,3,3,256,320))['image_tokens']
            self.assertEqual(list(out.shape),[1,3,6,128])
            self.assertEqual(calls,[((3,64,16,20),(3,128,8,10)),((3,64,8,10),(3,128,4,5)),((3,64,4,5),(3,128,2,3))])
            out.square().mean().backward()
            self.assertGreater(m.stage4_repeat_projection.weight.grad.abs().sum().item(),0)
            self.assertEqual(m.stage4_repeat_projection.weight.numel(),8192)
            self.assertEqual([n for n,_ in m.named_parameters() if not n.startswith('model.')],['stage4_repeat_projection.weight'])
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in m.model.parameters()))
            for n,x in m.model.named_modules():
                if isinstance(x,nn.BatchNorm2d):
                    self.assertFalse(x.training);torch.testing.assert_close(x.running_mean,before[n],rtol=0,atol=0)
            self.assertIs(stage,m.model.stages[-1]);self.assertFalse(hasattr(m,'stage4_pool'))
        finally:h.remove()

    def test_fusion_then_original_pool_for_all_variants(self):
        for name in variants.VISUAL_VARIANTS:
            with self.subTest(variant=name):
                m=scene(name).train();events=[];pool_io=[]
                def pool_hook(mod,args,out):
                    events.append('pool');pool_io.append(args[0].detach().clone())
                hooks=[m.mv_fuser.register_forward_pre_hook(lambda mod,args:events.append(('fusion',tuple(args[0].shape)))),
                       m.model_adapter.patch_merging.register_forward_hook(pool_hook)]
                try:out=m(torch.rand(1,3,3,240,320))
                finally:
                    for h in hooks:h.remove()
                count=6 if name in variants.STAGE4_VARIANTS else 80
                self.assertEqual(events,[('fusion',(1,3,count,128)),'pool'])
                self.assertEqual(list(out.shape),[1,3,4,m.dim_out])
                self.assertEqual(tuple(pool_io[0].shape), (3,128,2,3))
                self.assertIs(m.model_adapter.patch_merging.adapter,nn.AdaptiveAvgPool2d)
                self.assertEqual(len(m.model_adapter.downsample_blocks),0 if name in variants.STAGE4_VARIANTS else 2)
                out.square().mean().backward()
                for p in [next(m.mv_fuser.parameters()),m.model_adapter.patch_merging.proj.weight]:
                    self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(p.grad.abs().sum().item(),0)
                if name in variants.STAGE4_VARIANTS:self.assertGreater(m.backbone.stage4_repeat_projection.weight.grad.abs().sum().item(),0)

    def test_original_average_pool_and_linear_formula_is_retained(self):
        m=scene('stage4');x=torch.randn(3,128,2,3);a=m.model_adapter.patch_merging
        expected=a.proj(nn.functional.adaptive_avg_pool2d(x,(2,2)).permute(0,2,3,1))
        torch.testing.assert_close(a(x,(2,2)),expected,rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in nn.AdaptiveAvgPool2d((2,2)).parameters()),0)

    def test_json_strict_visual_reload_for_variants(self):
        for name in variants.VISUAL_VARIANTS:
            cfg=deepcopy(recipe(name)['model']['obs_encoder']['emvis_config']);cfg['load_backbone_pretrain']=False
            a=vision.VisualEncoder(emvis_config=cfg,rgb_key=['left_cam','right_cam','head_cam']).eval()
            b=vision.VisualEncoder(emvis_config=json.loads(json.dumps(cfg)),rgb_key=a.rgb_key).eval()
            b.load_state_dict(a.state_dict(),strict=True)
            obs={key:torch.rand(1,3,240,320) for key in a.rgb_key}
            with torch.no_grad():torch.testing.assert_close(a(obs),b(obs),rtol=0,atol=0)
            self.assertEqual(list(b.output_shape_meta()),[3,2,2,cfg['model_adapter_config']['dim_out']])

    def test_pooling_can_be_disabled_without_calling_any_pool(self):
        for name in variants.VISUAL_VARIANTS:
            with self.subTest(variant=name):
                cfg=recipe(name,False);m=scene(name,False).train();a=m.model_adapter.patch_merging
                self.assertFalse(cfg['model']['visual_pooling']);self.assertFalse(a.use_pooling)
                policy.S1DitFlowMatchingConfig(**deepcopy(cfg['model']))
                with patch.object(a,'adapter',side_effect=AssertionError('pooling must not execute')):
                    out=m(torch.rand(1,3,3,240,320))
                self.assertEqual(list(out.shape),[1,3,6,m.dim_out])
                out.square().mean().backward()
                self.assertGreater(a.proj.weight.grad.abs().sum().item(),0)
                if name in variants.STAGE4_VARIANTS:
                    self.assertGreater(m.backbone.stage4_repeat_projection.weight.grad.abs().sum().item(),0)
                x=torch.randn(3,128,2,3)
                torch.testing.assert_close(a(x,(2,3)),a.proj(x.permute(0,2,3,1)),rtol=0,atol=0)
                with self.assertRaisesRegex(ValueError,'natural grid'):a(x,(2,2))

    def test_pooling_metadata_roundtrips_without_new_visual_parameters(self):
        for name in ['stage4','stage4_rms']:
            a=scene(name,True).eval();b=scene(name,False).eval()
            b.load_state_dict(a.state_dict(),strict=True)
            self.assertEqual(set(a.state_dict()),set(b.state_dict()))
            cfg=deepcopy(recipe(name,False)['model']['obs_encoder']['emvis_config']);cfg['load_backbone_pretrain']=False
            v=vision.VisualEncoder(emvis_config=json.loads(json.dumps(cfg)),rgb_key=['left_cam','right_cam','head_cam']).eval()
            self.assertEqual(list(v.output_shape_meta()),[3,2,3,1024])
            v.scene_encoder.load_state_dict(b.state_dict(),strict=True)
            for bad in ['false',None,0]:
                raw=deepcopy(recipe(name)['model']);raw['visual_pooling']=bad
                with self.assertRaisesRegex(ValueError,'must be a bool'):policy.S1DitFlowMatchingConfig(**raw)
            old=deepcopy(recipe(name)['model']);old.pop('visual_pooling');old['obs_encoder']['emvis_config']['model_adapter_config'].pop('use_pooling')
            self.assertTrue(policy.S1DitFlowMatchingConfig(**old).visual_pooling)

    def test_arbitrary_or_contradictory_architecture_overrides_are_rejected(self):
        base=recipe('baseline')['model']
        for change in [{'visual_variant':'other'},{'hidden_size':512},{'num_heads':16},{'scene_adaptor':'identity'},
                       {'scene_adaptor_hidden_size':2048}]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                policy.S1DitFlowMatchingConfig(**dict(deepcopy(base),**change))
        for key in variants.RETIRED_BACKBONE_OPTIONS:
            cfg=deepcopy(base);cfg['obs_encoder']['emvis_config']['backbone_config'][key]=0
            with self.assertRaisesRegex(ValueError,'removed'):policy.S1DitFlowMatchingConfig(**cfg)
        for key,value in [('compress_before_fusion',True),('output_projection_type','mlp2x_gelu')]:
            cfg=deepcopy(base);cfg['obs_encoder']['emvis_config'][key]=value
            with self.assertRaisesRegex(ValueError,'removed'):policy.S1DitFlowMatchingConfig(**cfg)
        cfg=deepcopy(base);cfg['obs_encoder']['emvis_config']['visual_variant']='stage4'
        with self.assertRaisesRegex(ValueError,'must agree'):policy.S1DitFlowMatchingConfig(**cfg)
        for key in variants.RETIRED_BACKBONE_OPTIONS:
            with self.assertRaises(TypeError):encoder.EfficientViTEncoder('efficientvit_b0.r224_in1k',**{key:2})

if __name__=='__main__':unittest.main()
