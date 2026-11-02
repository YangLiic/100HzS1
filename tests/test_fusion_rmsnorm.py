"""Fusion RMSNorm math, gradients, inference semantics and checkpoint metadata."""
from copy import deepcopy
from importlib import import_module
from unittest.mock import patch
import json
import unittest

import torch
from torch import nn
from torch.nn import functional as F

fusion = import_module('100hzs1.models.policy.flow_matching.emvis.mv_fuser')


class FusionRMSNormTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_five_rms_norms_and_finite_trainable_gradients(self):
        torch.manual_seed(42)
        m = fusion.MultiViewFuser(128, depth=1, num_heads=16, mlp_ratio=2,
                                 norm_type='rmsnorm').train()
        norms = [(n, x) for n, x in m.named_modules() if isinstance(x, nn.RMSNorm)]
        self.assertEqual([n for n, _ in norms], [
            'frame_blocks.0.norm1', 'frame_blocks.0.norm2',
            'global_blocks.0.norm1', 'global_blocks.0.norm2', 'norm'])
        self.assertFalse(any(isinstance(x, nn.LayerNorm) for x in m.modules()))
        x = torch.randn(2, 3, 6, 128, requires_grad=True)
        y = m(x)
        self.assertEqual(y.shape, x.shape)
        (y * torch.randn_like(y)).mean().backward()
        for name, norm in norms:
            self.assertIsNotNone(norm.weight.grad, name)
            self.assertTrue(torch.isfinite(norm.weight.grad).all(), name)
            self.assertGreater(norm.weight.grad.abs().sum().item(), 0, name)
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_residual_attention_and_ffn_match_explicit_rms_math(self):
        torch.manual_seed(12)
        m = fusion.RMSNormTransformerEncoderLayer(
            16, 4, dim_feedforward=32, dropout=0, activation='gelu',
            batch_first=True, norm_first=True).double().train()
        # A nonzero channel mean distinguishes RMSNorm from LayerNorm.
        x = torch.randn(2, 6, 16, dtype=torch.float64) + 3
        n1 = F.rms_norm(x, (16,), m.norm1.weight, m.norm1.eps)
        a = m.self_attn(n1, n1, n1, need_weights=False)[0]
        h = x + a
        n2 = F.rms_norm(h, (16,), m.norm2.weight, m.norm2.eps)
        expected = h + F.linear(F.gelu(F.linear(n2, m.linear1.weight,
                   m.linear1.bias)), m.linear2.weight, m.linear2.bias)
        torch.testing.assert_close(m(x), expected, rtol=1e-12, atol=1e-12)

    def test_eval_never_enters_layernorm_only_fused_encoder_path(self):
        m = fusion.MultiViewFuser(128, num_heads=16, mlp_ratio=2,
                                 norm_type='rmsnorm').eval()
        x = torch.randn(1, 3, 6, 128) + 2
        # Eval + inference_mode + even heads is eligible for the parent fast path.
        with patch.object(torch, '_transformer_encoder_layer_fwd',
                          side_effect=AssertionError('LayerNorm fast path executed')):
            with torch.inference_mode():
                inference = m(x)
        m.train()
        with torch.no_grad():
            training = m(x)
        torch.testing.assert_close(inference, training, rtol=1e-5, atol=1e-6)

    def test_missing_norm_metadata_keeps_old_layernorm_state_and_outputs(self):
        old = fusion.MultiViewFuser(128, num_heads=16, mlp_ratio=2).eval()
        explicit = fusion.MultiViewFuser(128, num_heads=16, mlp_ratio=2,
                                        norm_type='layernorm').eval()
        explicit.load_state_dict(old.state_dict(), strict=True)
        self.assertEqual(sum(isinstance(x, nn.LayerNorm) for x in old.modules()), 5)
        self.assertEqual(sum(isinstance(x, nn.RMSNorm) for x in old.modules()), 0)
        self.assertIs(type(old.frame_blocks[0]), nn.TransformerEncoderLayer)
        x = torch.randn(1, 3, 6, 128)
        with torch.inference_mode():
            torch.testing.assert_close(old(x), explicit(x), rtol=0, atol=0)
        rms = fusion.MultiViewFuser(128, num_heads=16, mlp_ratio=2, norm_type='rmsnorm')
        self.assertEqual(sum(p.numel() for p in old.parameters()) -
                         sum(p.numel() for p in rms.parameters()), 640)
        with self.assertRaises(RuntimeError):
            rms.load_state_dict(old.state_dict(), strict=True)

    def test_rms_config_roundtrip_and_strict_visual_reload(self):
        from test_visual_ablation import recipe, vision
        raw = recipe('stage4_rms', False)['model']['obs_encoder']['emvis_config']
        self.assertEqual(raw['mv_fuser_config']['norm_type'], 'rmsnorm')
        raw['load_backbone_pretrain'] = False
        a = vision.VisualEncoder(emvis_config=deepcopy(raw),
                                 rgb_key=['left_cam', 'right_cam', 'head_cam']).eval()
        b = vision.VisualEncoder(emvis_config=json.loads(json.dumps(raw)), rgb_key=a.rgb_key).eval()
        b.load_state_dict(a.state_dict(), strict=True)
        obs = {key: torch.rand(1, 3, 240, 320) for key in a.rgb_key}
        with torch.inference_mode():
            torch.testing.assert_close(a(obs), b(obs), rtol=0, atol=0)
        self.assertEqual(a(obs).shape, (1, 18, 1024))
        # Only the five fusion affine biases disappear; token count/width stay fixed.
        self.assertFalse(any(n.endswith('.bias') and 'norm' in n
                             for n in a.scene_encoder.mv_fuser.state_dict()))


if __name__ == '__main__':
    unittest.main()
