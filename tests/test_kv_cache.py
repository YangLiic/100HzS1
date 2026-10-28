"""Inference K/V cache equivalence, lifetime, training and checkpoint compatibility."""
from contextlib import nullcontext
import subprocess
import types
from pathlib import Path
import unittest

import numpy as np
import torch
from torch import nn
from timm.models.vision_transformer import RmsNorm

from vfa.models.policy.vfa.dit_fm_policy import VFADitFlowMatching
from vfa.models.policy.vfa.module.blocks import CrossAttention
from vfa.models.policy.vfa.module.conditional_dit_head import DiT


def small_dit(self_attn_first=True):
    model = DiT(output_dim=14, horizon=6, hidden_size=64, depth=2, num_heads=4,
                self_attn_first=self_attn_first, scene_cond_len=12,
                scene_pos_embed_config=[("image", (1, 3, 4))], dtype=torch.float32)
    # A zero initialized output head would hide cache mistakes.
    nn.init.normal_(model.final_layer.ffn_final.fc2.weight, std=.02)
    return model


class Sampler(nn.Module):
    conditional_sample = VFADitFlowMatching.conditional_sample

    def __init__(self):
        super().__init__()
        self.model = small_dit()
        self.state_adaptor = nn.Linear(14, 64)
        self.horizon, self.action_dim = 6, 14
        self.num_inference_timesteps, self.num_timestep_buckets = 10, 1000

    def _zero_static_joints(self, value):
        return value


class KVCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_attention_cached_matches_uncached_with_masks_and_precision(self):
        for fused in (False, True):
            for bf16 in (False, True):
                with self.subTest(fused=fused, bf16=bf16):
                    model = CrossAttention(64, num_heads=4, qkv_bias=True,
                                           qk_norm=True, norm_layer=RmsNorm).eval()
                    model.fused_attn = fused
                    c, x = torch.randn(2, 12, 64), torch.randn(2, 7, 64)
                    mask = torch.ones(2, 12, dtype=torch.bool)
                    mask[:, -3:] = False
                    context = torch.autocast("cpu", dtype=torch.bfloat16) if bf16 else nullcontext()
                    with torch.no_grad(), context:
                        cache = model.prepare_kv(c)
                        for attention_mask in (None, mask):
                            expected = model(x, c, attention_mask)
                            actual = model(x, c, attention_mask, kv_cache=cache)
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_uncached_training_outputs_and_gradients_match_original(self):
        root = Path(__file__).resolve().parents[1]
        source = subprocess.check_output(
            ["git", "show", "8a7a28a:src/vfa/models/policy/vfa/module/blocks.py"],
            cwd=root, text=True)
        legacy = types.ModuleType("legacy_blocks")
        exec(compile(source, "<original_blocks>", "exec"), legacy.__dict__)
        original = legacy.CrossAttention(64, num_heads=4, qkv_bias=True,
                                          qk_norm=True, norm_layer=RmsNorm).train()
        updated = CrossAttention(64, num_heads=4, qkv_bias=True,
                                 qk_norm=True, norm_layer=RmsNorm).train()
        self.assertEqual(list(original.state_dict()), list(updated.state_dict()))
        updated.load_state_dict(original.state_dict(), strict=True)
        x = torch.randn(2, 7, 64, requires_grad=True)
        c = torch.randn(2, 12, 64, requires_grad=True)
        x2, c2 = x.detach().clone().requires_grad_(), c.detach().clone().requires_grad_()
        expected, actual = original(x, c), updated(x2, c2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.square().mean().backward()
        actual.square().mean().backward()
        torch.testing.assert_close(x2.grad, x.grad, rtol=0, atol=0)
        torch.testing.assert_close(c2.grad, c.grad, rtol=0, atol=0)
        for (name, old), (new_name, new) in zip(original.named_parameters(), updated.named_parameters()):
            self.assertEqual(name, new_name)
            torch.testing.assert_close(new.grad, old.grad, rtol=0, atol=0)

    def test_dit_cache_matches_for_both_attention_orders(self):
        for self_attn_first in (True, False):
            with self.subTest(self_attn_first=self_attn_first):
                model = small_dit(self_attn_first).eval()
                c = torch.randn(2, 12, 64)
                mask = torch.ones(2, 12, dtype=torch.bool)
                mask[:, -2:] = False
                keys = list(model.state_dict())
                with torch.no_grad():
                    cache = model.prepare_kv_cache(c)
                    for step in range(3):
                        x, t = torch.randn(2, 6, 64), torch.full((2,), step)
                        expected = model(x, t, c, mask)
                        actual = model(x, t, c, mask, kv_cache=cache)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(keys, list(model.state_dict()))

    def test_sampler_cache_counts_flow_and_new_condition(self):
        model = Sampler().eval()
        c = torch.randn(1, 12, 64)
        state = torch.randn(1, 6, 14)
        mask = torch.ones(1, 6, 14, dtype=torch.bool)
        mask[:, 0] = False
        counts, norm_counts = [0, 0], [0, 0]
        hooks = []
        for index, block in enumerate(model.model.blocks):
            def count(module, args, result, i=index): counts[i] += 1
            def count_norm(module, args, result, i=index): norm_counts[i] += 1
            hooks.append(block.cross_attn.kv.register_forward_hook(count))
            hooks.append(block.cross_attn.k_norm.register_forward_hook(count_norm))
        try:
            for condition in (c, c + .25):
                counts[:] = [0, 0]
                norm_counts[:] = [0, 0]
                with torch.no_grad():
                    torch.manual_seed(42)
                    cached, cached_flow = model.conditional_sample(condition, state, mask, True)
                    self.assertEqual(counts, [1, 1])
                    self.assertEqual(norm_counts, [1, 1])
                    counts[:] = [0, 0]
                    norm_counts[:] = [0, 0]
                    torch.manual_seed(42)
                    expected, expected_flow = model.conditional_sample(
                        condition, state, mask, True, use_kv_cache=False)
                    self.assertEqual(counts, [10, 10])
                    self.assertEqual(norm_counts, [10, 10])
                torch.testing.assert_close(cached, expected, rtol=0, atol=0)
                for a, b in zip(cached_flow, expected_flow): np.testing.assert_array_equal(a, b)
        finally:
            for hook in hooks: hook.remove()

    def test_sampler_skips_cache_in_training_and_grad_enabled_eval(self):
        model = Sampler()
        c, state = torch.randn(1, 12, 64), torch.randn(1, 6, 14)
        mask = torch.ones(1, 6, 14, dtype=torch.bool)
        mask[:, 0] = False
        def forbidden(*args): raise AssertionError("cache reached a training/grad-enabled path")
        model.model.prepare_kv_cache = forbidden
        with torch.no_grad():
            model.train().conditional_sample(c, state, mask)
        result, _ = model.eval().conditional_sample(c, state, mask)
        self.assertTrue(result.requires_grad)

    def test_cache_rejects_training_grad_enabled_and_incomplete_cache(self):
        model = small_dit()
        c, x, t = torch.randn(1, 12, 64), torch.randn(1, 6, 64), torch.tensor([0])
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, "eval mode"):
            model.prepare_kv_cache(c)
        model.eval()
        with self.assertRaisesRegex(RuntimeError, "gradients disabled"):
            model.prepare_kv_cache(c)
        with torch.no_grad():
            cache = model.prepare_kv_cache(c)
            with self.assertRaisesRegex(ValueError, "one entry"):
                model(x, t, c, kv_cache=cache[:-1])
        with self.assertRaisesRegex(RuntimeError, "gradients disabled"):
            model(x, t, c, kv_cache=cache)


if __name__ == "__main__":
    unittest.main()
