"""New defaults and the explicitly requested no-RMS visual residual ablation."""
from copy import deepcopy
from importlib import import_module
import unittest
import torch
from torch import nn
from test_visual_ablation import recipe
V=import_module('100hzs1.models.policy.flow_matching.visual_variants')
P=import_module('100hzs1.models.policy.flow_matching.dit_fm_policy')
A=import_module('100hzs1.models.policy.flow_matching.module.condition_adapter')
class GrootSixTests(unittest.TestCase):
    def test_live_defaults_and_explicit_overrides(self):
        from hydra import initialize_config_dir,compose
        from omegaconf import OmegaConf
        from pathlib import Path
        OmegaConf.register_new_resolver('eval',eval,replace=True)
        for name in V.VISUAL_VARIANTS:
            with initialize_config_dir(version_base=None,config_dir=str(Path(__file__).resolve().parents[1]/'configs/100hzs1_train')):
                c=OmegaConf.to_container(compose(config_name='backbones/efficientvit_b0/tau_token/b0_'+name+'_fm4'),resolve=True)
            self.assertEqual(c['norm'],'groot');self.assertEqual(c['model']['norm'],'groot')
            self.assertFalse(c['model']['visual_pooling'])
            self.assertEqual(c['model']['time_injection'],'tau_token')
            self.assertEqual(c['model']['obs_encoder']['emvis_config']['model_adapter_config']['shape_out'],[2,3])
            P.S1DitFlowMatchingConfig(**deepcopy(c['model']))
        self.assertEqual(P.S1DitFlowMatchingConfig(state_key='state').norm,'groot')
        c=recipe('baseline',True)['model'];c['norm']='minmax'
        self.assertTrue(P.S1DitFlowMatchingConfig(**c).visual_pooling)
    def test_no_norm_retains_identical_residual_formula_and_state_keys(self):
        a=A.ResidualMLPAdapter(32,32,use_rmsnorm=False)
        self.assertIsInstance(a.norm,nn.Identity)
        self.assertFalse(any('norm' in k for k in a.state_dict()))
        x=torch.randn(2,18,32,requires_grad=True)
        torch.testing.assert_close(a(x),x+a.fc2(a.act(a.fc1(x))),rtol=0,atol=0)
        a(x).square().mean().backward()
        self.assertTrue(all(q.grad is not None and torch.isfinite(q.grad).all() for q in a.parameters()))
        b=A.ResidualMLPAdapter(32,32,use_rmsnorm=False);b.load_state_dict(a.state_dict(),strict=True)
        torch.testing.assert_close(a(x),b(x),rtol=0,atol=0)
        self.assertIsInstance(A.ResidualMLPAdapter(32,32).norm,import_module('timm.models.vision_transformer').RmsNorm)
if __name__=='__main__':unittest.main()
