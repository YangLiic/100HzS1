"""Residual visual adaptor with fixed equal-width DiT FFNs."""
from copy import deepcopy
from importlib import import_module
import json
from pathlib import Path
import unittest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch import nn

policy = import_module('100hzs1.models.policy.flow_matching.dit_fm_policy')
adapters = import_module('100hzs1.models.policy.flow_matching.module.condition_adapter')
head = import_module('100hzs1.models.policy.flow_matching.module.conditional_dit_head')
ROOT = Path(__file__).resolve().parents[1]
NAME = 'backbones/efficientvit_b0/tau_token/b0_rms_fm4'


def recipe():
    OmegaConf.register_new_resolver('eval',eval,replace=True)
    with initialize_config_dir(version_base=None,config_dir=str(ROOT/'configs/100hzs1_train')):
        return OmegaConf.to_container(compose(config_name=NAME),resolve=True)


def small_policy(injection='tau_token',flow='fm',visual_residual=True):
    m=deepcopy(recipe()['model']);m.update(hidden_size=64,num_heads=1,depth=2,time_injection=injection,flow_type=flow)
    m.pop('visual_variant',None)
    if not visual_residual:m['scene_adaptor']='mlp2x_gelu'
    v=m['obs_encoder']['emvis_config'];v['visual_variant']='baseline'
    # Generic non-B0 unit fixture keeps tiny heads; B0 presets reject arbitrary widths.
    v['backbone_config']['model_name']='efficientvit_b2.r256_in1k'
    v['load_backbone_pretrain']=False;v['model_adapter_config']['dim_out']=64
    p=policy.S1DitFlowMatching(policy.S1DitFlowMatchingConfig(**m)).float()
    p.model.dtype=p.model.t_embedder.dtype=torch.float32
    if p.model.r_embedder is not None:p.model.r_embedder.dtype=torch.float32
    # Keep gradients observable without spending optimizer steps on zero-head startup.
    nn.init.normal_(p.model.final_layer.ffn_final.fc2.weight,std=.02)
    return p


class ResidualWidthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_formal_recipe_and_legacy_widths(self):
        c=recipe();m=c['model'];v=m['obs_encoder']['emvis_config']
        self.assertEqual((m['hidden_size'],m['num_heads'],m['depth']),(1024,16,8))
        self.assertNotIn('ffn_hidden_size',m)
        self.assertEqual(m['scene_adaptor'],'residual_mlp2x_gelu');self.assertEqual(m['state_adaptor'],'mlp3x_gelu')
        self.assertEqual(v['model_adapter_config']['dim_out'],1024);self.assertEqual(v['model_adapter_config']['shape_out'],[2,3])
        self.assertEqual(v['model_adapter_config']['conv_num'],2);self.assertEqual(m['visual_variant'],'rms')
        self.assertEqual(m['flow_type'],'fm');self.assertEqual(c['inference']['num_inference_timesteps'],4)
        self.assertEqual((c['norm'],c['seed'],c['num_train_epochs'],c['train_batch_size']),('groot',42,200,128))
        self.assertIsNone(c['resume_from_checkpoint'])

    def test_residual_norm_is_only_on_branch(self):
        a=adapters.ResidualMLPAdapter(64,64);x=torch.randn(2,12,64,requires_grad=True)
        expected=x+a.fc2(a.act(a.fc1(a.norm(x))))
        torch.testing.assert_close(a(x),expected,rtol=0,atol=0)
        a(x).square().mean().backward()
        for p in a.parameters():self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0)
        nn.init.zeros_(a.fc2.weight);nn.init.zeros_(a.fc2.bias)
        torch.testing.assert_close(a(x),x,rtol=0,atol=0)
        for dims in [(32,64)]:
            with self.assertRaises(ValueError):adapters.ResidualMLPAdapter(*dims)

    def test_both_time_modes_and_flows_gradients_cached_equivalence_and_strict_reload(self):
        for injection,flow in [('tau_token','fm'),('tau_AdaRMS','fm'),('tau_token','meanflow'),('tau_AdaRMS','meanflow')]:
            with self.subTest(injection=injection,flow=flow):
                p=small_policy(injection,flow);D=64;t=torch.ones(1);r=torch.zeros(1) if flow=='meanflow' else None
                self.assertEqual([(b.ffn.fc1.in_features,b.ffn.fc1.out_features,b.ffn.fc2.out_features) for b in p.model.blocks],[(64,64,64)]*2)
                self.assertEqual([(x.in_features,x.out_features) for x in p.state_adaptor if isinstance(x,nn.Linear)],[(14,64),(64,64),(64,64)])
                self.assertEqual(p.model.t_embedder.mlp[-1].out_features,D)
                if flow=='meanflow':self.assertEqual(p.model.r_embedder.mlp[-1].out_features,D)
                else:self.assertIsNone(p.model.r_embedder)
                source=torch.randn(1,18,D);conditions=p.scene_adaptor(source);x=p.state_adaptor(torch.randn(1,24,14))
                p.train();out=p.model(x,t,conditions,r=r);out.square().mean().backward()
                checked=[p.scene_adaptor.norm.weight,p.scene_adaptor.fc1.weight,p.scene_adaptor.fc2.weight,p.model.blocks[0].ffn.fc1.weight,p.model.t_embedder.mlp[0].weight]
                checked += [x.weight for x in p.state_adaptor if isinstance(x,nn.Linear)]
                if flow=='meanflow':checked.append(p.model.r_embedder.mlp[0].weight)
                for q in checked:self.assertIsNotNone(q.grad);self.assertTrue(torch.isfinite(q.grad).all());self.assertGreater(float(q.grad.abs().sum()),0)
                p.eval()
                with torch.no_grad():
                    cache=p.model.prepare_kv_cache(conditions)
                    self.assertEqual(cache[0][0].shape,(1,1,18,64))
                    cached=p.model(x,t,conditions,kv_cache=cache,r=r);plain=p.model(x,t,conditions,r=r)
                    torch.testing.assert_close(cached,plain,rtol=1e-5,atol=1e-6)
                restored=policy.S1DitFlowMatching(policy.S1DitFlowMatchingConfig(**json.loads(p.config.to_json_string()))).float().eval()
                restored.model.dtype=restored.model.t_embedder.dtype=torch.float32
                if restored.model.r_embedder is not None:restored.model.r_embedder.dtype=torch.float32
                restored.load_state_dict(p.state_dict(),strict=True)
                with torch.no_grad():torch.testing.assert_close(restored.model(x,t,conditions,r=r),plain,rtol=0,atol=0)

    def test_old_configs_default_to_original_ffn_and_state_layout(self):
        old=small_policy(visual_residual=False)
        self.assertEqual([b.ffn.fc1.out_features for b in old.model.blocks],[64,64])
        self.assertFalse(any('scene_adaptor.norm.' in key for key in old.state_dict()))
        restored=small_policy(visual_residual=False);restored.load_state_dict(old.state_dict(),strict=True)

    def test_retired_ffn_width_cannot_be_set_by_config_or_constructor(self):
        self.assertNotIn('ffn_hidden_size',policy.S1DitFlowMatchingConfig.__dataclass_fields__)
        cfg=deepcopy(recipe()['model'])
        legacy=policy.S1DitFlowMatchingConfig(**dict(cfg,ffn_hidden_size=None))
        self.assertNotIn('ffn_hidden_size',legacy.to_dict())
        for value in [-1,0,True,1.5,1024,2048]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError,'fixed to hidden_size'):
                    policy.S1DitFlowMatchingConfig(**dict(cfg,ffn_hidden_size=value))
                with self.assertRaises(TypeError):
                    head.DiT(14,24,hidden_size=64,depth=1,num_heads=1,ffn_hidden_size=value)
                with self.assertRaises(TypeError):
                    head.Block(64,1,ffn_hidden_size=value)
                with self.assertRaises(TypeError):
                    head.AdaRMSBlock(64,1,ffn_hidden_size=value)

    def test_real_1024_and_2048_dit_blocks_keep_ffn_equal_to_hidden_width(self):
        for width in [1024,2048]:
            for cls in [head.Block,head.AdaRMSBlock]:
                with self.subTest(width=width,block=cls.__name__):
                    block=cls(width,width//64)
                    self.assertEqual((block.ffn.fc1.in_features,block.ffn.fc1.out_features,
                                      block.ffn.fc2.in_features,block.ffn.fc2.out_features),
                                     (width,width,width,width))
                    self.assertIsInstance(block.ffn.act,nn.GELU)
                    del block

    def test_wide_ffn_checkpoint_tensors_are_not_silently_loaded(self):
        first=small_policy();state=first.state_dict()
        state['model.blocks.0.ffn.fc1.weight']=torch.zeros(128,64)
        state['model.blocks.0.ffn.fc1.bias']=torch.zeros(128)
        state['model.blocks.0.ffn.fc2.weight']=torch.zeros(64,128)
        with self.assertRaisesRegex(RuntimeError,'size mismatch'):
            small_policy().load_state_dict(state,strict=True)



if __name__=='__main__':unittest.main()
