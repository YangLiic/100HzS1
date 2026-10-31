"""MeanFlow identity, masking, gradients, sampling and checkpoint runtime contract."""
from importlib import import_module
from types import SimpleNamespace
import unittest
import torch
from torch import nn

mf = import_module('100hzs1.models.policy.flow_matching.meanflow')
Policy = import_module('100hzs1.models.policy.flow_matching.dit_fm_policy').S1DitFlowMatching
Config = import_module('100hzs1.models.policy.flow_matching.dit_fm_policy').S1DitFlowMatchingConfig
DiT = import_module('100hzs1.models.policy.flow_matching.module.conditional_dit_head').DiT
runtime = import_module('100hzs1.common.inference')


def small_dit(mode='meanflow'):
    model = DiT(3, 4, hidden_size=32, depth=2, num_heads=4, scene_cond_len=2,
                scene_pos_embed_config=[('image', (1, 1, 2))], dtype=torch.float32,
                flow_type=mode)
    nn.init.normal_(model.final_layer.ffn_final.fc2.weight, std=.05)
    return model


class MeanFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_config_default_and_pure_scope(self):
        self.assertEqual(Config(action_key='state').flow_type, 'fm')
        self.assertEqual(Config(action_key='state').state_key, 'state')
        self.assertEqual(Config(state_key='state', flow_type='meanflow').meanflow['equal_time_ratio'], 0.0)
        for value in ({'guidance_scale': 2}, {'equal_time_ratio': 1.1},
                      {'loss_eps': 0}, {'time_std': 0}, {'dispersive_weight': .5},
                      {'time_embed_scale': 0}, {'time_embed_scale': float('nan')}):
            with self.assertRaises(ValueError): mf.validate_options(value)
        with self.assertRaises(ValueError): Config(flow_type='unknown')

    def test_default_pairs_have_no_forced_equal_times(self):
        torch.manual_seed(123)
        t, r = mf.sample_time_pairs(128, 'cpu', mf.validate_options())
        self.assertEqual(int((t == r).sum()), 0)

    def test_pairs_exact_half_and_seeded(self):
        options = mf.validate_options({'equal_time_ratio': 0.5})
        torch.manual_seed(123)
        t, r = mf.sample_time_pairs(16, 'cpu', options)
        self.assertEqual(int((t == r).sum()), 8)
        self.assertTrue(((t >= r) & (r >= 0) & (t <= 1)).all())
        torch.manual_seed(123)
        t2, r2 = mf.sample_time_pairs(16, 'cpu', options)
        self.assertTrue(torch.equal(t, t2) and torch.equal(r, r2))

    def test_identity_target_detach_and_masked_gradient(self):
        scale = nn.Parameter(torch.tensor(1.3))
        z = torch.randn(2, 4, 3)
        t, r = torch.tensor([.8, .6]), torch.tensor([.2, .6])
        v = torch.randn_like(z)
        mask = torch.ones_like(z, dtype=torch.bool); mask[:, 0] = False; mask[:, :, -1] = False
        v = v * mask
        fn = lambda z, t, r: scale * z + t[:, None, None].square() + 2 * r[:, None, None]
        result = mf.objective(fn, z, t, r, v, mask, mf.validate_options({'loss_type': 'mse'}), nn.Identity())
        derivative = scale.detach() * v + 2 * t[:, None, None]
        target = v - (t-r)[:, None, None] * derivative
        error = fn(z, t, r).detach() - target
        denominator = mask.flatten(1).sum(1)
        expected = ((error.square() * mask).flatten(1).sum(1) / denominator).mean()
        torch.testing.assert_close(result['loss'].detach(), expected)
        result['loss'].backward()
        expected_grad = ((2 * error * z * mask).flatten(1).sum(1) / denominator).mean()
        torch.testing.assert_close(scale.grad, expected_grad)
        sample_mse = (error.square() * mask).flatten(1).sum(1) / denominator
        torch.testing.assert_close(result['meanflow_mse_interval'], sample_mse[0])
        torch.testing.assert_close(result['meanflow_mse_equal_time'], sample_mse[1])

    def test_actual_dit_jvp_finite_difference_and_trainable_conditions(self):
        torch.manual_seed(5)
        head = small_dit()
        adapter = nn.Linear(3, 32)
        scene_adapter = nn.Linear(8, 32)
        scene = scene_adapter(torch.randn(2, 2, 8))
        z, v = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
        t, r = torch.tensor([.8, .7]), torch.tensor([.2, .3])
        mask = torch.ones_like(z, dtype=torch.bool); mask[:, 0] = False
        v = v * mask
        state = z[:, :1].clone()
        def fn(x, t, r):
            x = torch.cat((state, x[:, 1:]), dim=1)
            return head(adapter(x), t, scene, r=r)
        flags = [m.fused_attn for m in head.modules() if hasattr(m, 'fused_attn')]
        with torch.no_grad(), mf.math_attention(head):
            _, derivative = torch.func.jvp(fn, (z, t, r), (v, torch.ones_like(t), torch.zeros_like(r)))
            eps = .001
            finite_difference = (fn(z+eps*v, t+eps, r)-fn(z-eps*v, t-eps, r)) / (2*eps)
        torch.testing.assert_close(derivative, finite_difference, rtol=.02, atol=.002)
        self.assertEqual(flags, [m.fused_attn for m in head.modules() if hasattr(m, 'fused_attn')])
        result = mf.objective(fn, z, t, r, v, mask, mf.validate_options(), head)
        result['loss'].backward()
        for value in (adapter.weight.grad, scene_adapter.weight.grad, head.r_embedder.mlp[0].weight.grad):
            self.assertIsNotNone(value)
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreater(float(value.abs().sum()), 0)

    def test_attention_flags_restore_on_exception(self):
        head = small_dit()
        flags = [m.fused_attn for m in head.modules() if hasattr(m, 'fused_attn')]
        with self.assertRaises(RuntimeError):
            with mf.math_attention(head): raise RuntimeError('test')
        self.assertEqual(flags, [m.fused_attn for m in head.modules() if hasattr(m, 'fused_attn')])

    def test_fm_has_no_additional_parameters(self):
        fm, mean = small_dit('fm'), small_dit()
        self.assertFalse(any('r_embedder' in k for k in fm.state_dict()))
        extra = set(mean.state_dict()) - set(fm.state_dict())
        self.assertEqual(extra, {'r_embedder.mlp.0.weight', 'r_embedder.mlp.0.bias',
                                 'r_embedder.mlp.2.weight', 'r_embedder.mlp.2.bias'})
        self.assertEqual(fm.x_pos_embed.shape, mean.x_pos_embed.shape)
        with self.assertRaises(ValueError): mean(torch.randn(2, 4, 32), torch.ones(2), torch.randn(2, 2, 32))

    def test_sampling_direction_prefix_static_mask_and_nfe(self):
        class Oracle(nn.Module):
            def forward(self, x, t, scene, r=None, kv_cache=None):
                self.calls += 1
                return x / (t[:, None, None] / 1000)
        class Sampler(nn.Module):
            conditional_sample = Policy.conditional_sample
            _sample_meanflow = Policy._sample_meanflow
            def _zero_static_joints(self, x): return x.masked_fill(self.static_joint_mask.view(1, 1, -1), 0)
        for steps in (1, 2, 4):
            sampler = Sampler().eval()
            sampler.flow_type = 'meanflow'; sampler.horizon = 4; sampler.action_dim = 3
            sampler.num_inference_timesteps = steps; sampler.num_timestep_buckets = 1000
            sampler.state_adaptor = nn.Identity(); sampler.model = Oracle(); sampler.model.calls = 0
            sampler.static_joint_mask = torch.tensor([False, False, True])
            state = torch.ones(2, 4, 3) * 2
            mask = torch.ones_like(state, dtype=torch.bool); mask[:, 0] = False
            with torch.no_grad():
                result, flow = sampler.conditional_sample(torch.empty(2, 2, 3), state, mask, True,
                                                           use_kv_cache=False, noise_trajectory=torch.ones_like(state))
            torch.testing.assert_close(result[:, 1:], torch.zeros_like(result[:, 1:]))
            torch.testing.assert_close(result[:, 0, :2], state[:, 0, :2])
            self.assertEqual(sampler.model.calls, steps)
            self.assertEqual(len(flow), steps+1)
            self.assertTrue((result[:, :, -1] == 0).all())

    def test_meanflow_cache_matches_and_equal_time_is_local_target(self):
        class Sampler(nn.Module):
            conditional_sample = Policy.conditional_sample
            _sample_meanflow = Policy._sample_meanflow
            def _zero_static_joints(self, x): return x
        sampler = Sampler().eval()
        sampler.flow_type = 'meanflow'; sampler.horizon = 4; sampler.action_dim = 3
        sampler.num_inference_timesteps = 4; sampler.num_timestep_buckets = 1000
        sampler.state_adaptor = nn.Linear(3, 32); sampler.model = small_dit().eval()
        scene, state = torch.randn(2, 2, 32), torch.randn(2, 4, 3)
        mask = torch.ones_like(state, dtype=torch.bool);mask[:, 0] = False
        with torch.no_grad():
            torch.manual_seed(23);a, _ = sampler.conditional_sample(scene, state, mask, use_kv_cache=True)
            torch.manual_seed(23);b, _ = sampler.conditional_sample(scene, state, mask, use_kv_cache=False)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        t = torch.tensor([.7, .8]);v = torch.randn_like(state)
        scale = nn.Parameter(torch.tensor(1.0))
        fn = lambda z, t, r: scale * z + t[:, None, None]
        result = mf.objective(fn, state, t, t, v, mask, mf.validate_options({'loss_type': 'mse'}), nn.Identity())
        expected = ((fn(state, t, t)-v).square()*mask).sum()/mask.sum()
        torch.testing.assert_close(result['loss'], expected)
        self.assertEqual(float(result['meanflow_correction_rms']), 0)
        self.assertEqual(float(result['meanflow_mse_interval']), 0)
        torch.testing.assert_close(result['meanflow_mse_equal_time'], expected.detach())

    def test_saved_time_scale_changes_sampler_inputs_without_reinterpreting_v1(self):
        class Oracle(nn.Module):
            def forward(self, x, t, scene, r=None, kv_cache=None):
                self.times.append((t.clone(), r.clone()))
                return x / (t[:, None, None] / self.time_scale)
        class Sampler(nn.Module):
            conditional_sample = Policy.conditional_sample
            _sample_meanflow = Policy._sample_meanflow
            def _zero_static_joints(self, x): return x
        for options, expected_scale in ((None, 1000), ({}, 1000),
                                         (mf.validate_options(), 1000),
                                         (mf.validate_options({'time_embed_scale': 1.0}), 1)):
            sampler = Sampler().eval()
            sampler.flow_type = 'meanflow'; sampler.horizon = 4; sampler.action_dim = 3
            sampler.num_inference_timesteps = 4; sampler.num_timestep_buckets = 1000
            sampler.meanflow_options = options
            sampler.state_adaptor = nn.Identity(); sampler.model = Oracle()
            sampler.model.times = []; sampler.model.time_scale = expected_scale
            mask = torch.ones(2,4,3,dtype=torch.bool); mask[:,0] = False
            state = torch.ones(2,4,3)*2
            with torch.no_grad():
                result, _ = sampler.conditional_sample(torch.empty(2,2,3),state,mask,
                    use_kv_cache=False,noise_trajectory=torch.ones_like(state))
            torch.testing.assert_close(result[:,1:],torch.zeros_like(result[:,1:]))
            torch.testing.assert_close(result[:,0],state[:,0])
            for i,(t,r) in enumerate(sampler.model.times):
                torch.testing.assert_close(t,torch.full((2,),expected_scale*(1-i/4)))
                torch.testing.assert_close(r,torch.full((2,),expected_scale*(1-(i+1)/4)))

    def test_checkpoint_runtime_precedence(self):
        config = {'flow_type': 'meanflow', 'inference': {'num_inference_timesteps': 1}}
        self.assertEqual(runtime.load_inference_options(checkpoint_config=config)['num_inference_timesteps'], 1)
        self.assertEqual(runtime.load_inference_options(checkpoint_config={'flow_type': 'meanflow'})['num_inference_timesteps'], 1)
        self.assertEqual(runtime.load_inference_options(checkpoint_config={})['num_inference_timesteps'], 10)
        self.assertEqual(runtime.load_inference_options(overrides={'num_inference_timesteps': 4}, checkpoint_config=config)['num_inference_timesteps'], 4)

if __name__ == '__main__': unittest.main()
