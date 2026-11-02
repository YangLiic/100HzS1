"""Isolate visual RMSNorm while preserving the residual MLP and policy widths."""
from copy import deepcopy
from importlib import import_module
from pathlib import Path
import unittest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
adapters=import_module('100hzs1.models.policy.flow_matching.module.condition_adapter')
policy=import_module('100hzs1.models.policy.flow_matching.dit_fm_policy')

def recipe(name):
    OmegaConf.register_new_resolver('eval',eval,replace=True)
    with initialize_config_dir(version_base=None,config_dir=str(ROOT/'configs/100hzs1_train')):
        return OmegaConf.to_container(compose(config_name=name,overrides=['output_dir=/data/vepfs/licheng.yang/100HzS1/runs/config_test','base_dir=/data/vepfs/licheng.yang/100HzS1/runs/config_test']),resolve=True)

class NoNormVisualAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_only_resolved_scene_adapter_changes(self):
        base=recipe('backbones/efficientvit_b0/tau_token/b0_stage4_rms_fm4')
        new=recipe('ablations/b0_stage4_resmlp_no_norm_fm4')
        self.assertEqual(new['model']['scene_adaptor'],'residual_mlp2x_gelu_no_norm')
        validated=policy.S1DitFlowMatchingConfig(**deepcopy(new['model']))
        self.assertEqual(validated.hidden_size,1024)
        self.assertFalse(validated.visual_pooling)
        self.assertEqual(new['model']['obs_encoder']['emvis_config']['model_adapter_config']['shape_out'],[2,3])
        new['model']['scene_adaptor']=base['model']['scene_adaptor']
        self.assertEqual(base,new)

    def test_same_initial_weights_exactly_one_norm_removed(self):
        torch.manual_seed(42);base=adapters.ResidualMLPAdapter(1024,1024)
        torch.manual_seed(42);new=adapters.ResidualMLPAdapter(1024,1024,use_rms_norm=False)
        self.assertIsInstance(new.norm,nn.Identity)
        self.assertEqual(set(base.state_dict())-set(new.state_dict()),{'norm.weight'})
        for name,value in new.state_dict().items():
            torch.testing.assert_close(value,base.state_dict()[name],rtol=0,atol=0)
        self.assertEqual(sum(p.numel() for p in base.parameters())-sum(p.numel() for p in new.parameters()),1024)

    def test_formula_gradients_and_strict_reload(self):
        block=policy.S1DitFlowMatching.build_condition_adapter(None,'residual_mlp2x_gelu_no_norm',1024,1024)
        x=torch.randn(2,18,1024,requires_grad=True)
        torch.testing.assert_close(block(x),x+block.fc2(block.act(block.fc1(x))),rtol=0,atol=0)
        block(x).square().mean().backward()
        for name,p in block.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
            self.assertGreater(p.grad.abs().sum().item(),0,name)
        copy=adapters.ResidualMLPAdapter(1024,1024,use_rms_norm=False)
        copy.load_state_dict(block.state_dict(),strict=True)
        torch.testing.assert_close(block(x),copy(x),rtol=0,atol=0)

    def test_existing_rms_checkpoint_layout_preserved(self):
        block=policy.S1DitFlowMatching.build_condition_adapter(None,'residual_mlp2x_gelu',1024,1024)
        self.assertEqual(type(block.norm).__name__,'RmsNorm')
        expected={'norm.weight','fc1.weight','fc1.bias','fc2.weight','fc2.bias'}
        self.assertEqual(set(block.state_dict()),expected)
        adapters.ResidualMLPAdapter(1024,1024).load_state_dict(block.state_dict(),strict=True)

    def test_no_norm_override_rejected_for_non_residual_preset(self):
        raw=recipe('backbones/efficientvit_b0/tau_token/b0_stage4_fm4')['model']
        raw['scene_adaptor']='residual_mlp2x_gelu_no_norm'
        with self.assertRaisesRegex(ValueError,'fixes scene_adaptor'):
            policy.S1DitFlowMatchingConfig(**raw)

if __name__=='__main__':unittest.main()
