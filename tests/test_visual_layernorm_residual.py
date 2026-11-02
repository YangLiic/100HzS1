"""LayerNorm-only visual residual ablations, preserving fusion and DiT controls."""
from copy import deepcopy
from importlib import import_module
import json
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from test_visual_ablation import recipe, scene, policy

A=import_module('100hzs1.models.policy.flow_matching.module.condition_adapter')

class LayerNormResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_formula_centering_skip_and_gradients(self):
        a=A.LayerNormResidualMLPAdapter(32,32)
        x=(torch.randn(2,18,32)+4).requires_grad_()
        normalized=F.layer_norm(x,(32,),a.norm.weight,a.norm.bias,1e-6)
        expected=x+a.fc2(a.act(a.fc1(normalized)))
        torch.testing.assert_close(a(x),expected,rtol=0,atol=0)
        a(x).square().mean().backward()
        for name,q in a.named_parameters():
            self.assertIsNotNone(q.grad,name)
            self.assertTrue(torch.isfinite(q.grad).all(),name)
            self.assertGreater(q.grad.abs().sum().item(),0,name)
        nn.init.zeros_(a.fc2.weight);nn.init.zeros_(a.fc2.bias)
        torch.testing.assert_close(a(x),x,rtol=0,atol=0)

    def test_only_norm_differs_from_rms_with_matched_initial_linears(self):
        torch.manual_seed(42);r=A.ResidualMLPAdapter(32,32)
        torch.manual_seed(42);l=A.LayerNormResidualMLPAdapter(32,32)
        for key in ['fc1.weight','fc1.bias','fc2.weight','fc2.bias']:
            torch.testing.assert_close(r.state_dict()[key],l.state_dict()[key],rtol=0,atol=0)
        self.assertEqual(r.norm.eps,l.norm.eps)
        self.assertEqual(set(l.state_dict())-set(r.state_dict()),{'norm.bias'})
        self.assertEqual(sum(q.numel() for q in l.parameters())-sum(q.numel() for q in r.parameters()),32)
        x=torch.randn(1,18,32)+5
        self.assertFalse(torch.allclose(r(x),l(x)))

    def test_named_recipes_keep_fusion_layernorm_and_expected_compression(self):
        for name in ['ln','stage4_ln']:
            with self.subTest(variant=name):
                c=recipe(name,False);m=c['model'];v=m['obs_encoder']['emvis_config']
                self.assertEqual((m['hidden_size'],m['num_heads'],m['depth']),(1024,16,8))
                self.assertEqual(m['scene_adaptor'],'residual_mlp2x_gelu_layernorm')
                self.assertEqual(v['mv_fuser_config']['norm_type'],'layernorm')
                self.assertEqual(v['model_adapter_config']['dim_out'],1024)
                self.assertEqual(v['model_adapter_config']['conv_num'],None if name=='stage4_ln' else 2)
                e=scene(name,False).eval()
                self.assertEqual(sum(isinstance(q,nn.LayerNorm) for q in e.mv_fuser.modules()),5)
                self.assertEqual(sum(isinstance(q,nn.RMSNorm) for q in e.mv_fuser.modules()),0)
                self.assertEqual(e(torch.rand(1,3,3,240,320)).shape,(1,3,6,1024))
                p=policy.S1DitFlowMatchingConfig(**deepcopy(m))
                restored=policy.S1DitFlowMatchingConfig(**json.loads(p.to_json_string()))
                self.assertEqual(restored.visual_variant,name)
                a=policy.S1DitFlowMatching.build_condition_adapter(None,m['scene_adaptor'],1024,1024)
                self.assertIsInstance(a.norm,nn.LayerNorm)
                self.assertEqual((a.fc1.in_features,a.fc1.out_features,a.fc2.out_features),(1024,2048,1024))

    def test_strict_adapter_reload_and_old_rms_layout_are_preserved(self):
        a=A.LayerNormResidualMLPAdapter(32,32);b=A.LayerNormResidualMLPAdapter(32,32)
        b.load_state_dict(a.state_dict(),strict=True);x=torch.randn(1,18,32)
        torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
        self.assertNotIn('norm.bias',A.ResidualMLPAdapter(32,32).state_dict())
        self.assertFalse(any(k.startswith('norm.') for k in A.ResidualMLPAdapter(32,32,use_rmsnorm=False).state_dict()))
        with self.assertRaises(ValueError):A.LayerNormResidualMLPAdapter(32,64)

if __name__=='__main__':unittest.main()
