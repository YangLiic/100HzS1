"""Time injection, MeanFlow derivatives, caching and compatibility contracts."""
from importlib import import_module
from functools import partial
import unittest
from unittest.mock import patch

import torch
from torch import nn

policy_module = import_module('100hzs1.models.policy.flow_matching.dit_fm_policy')
head_module = import_module('100hzs1.models.policy.flow_matching.module.conditional_dit_head')
mf = import_module('100hzs1.models.policy.flow_matching.meanflow')
DiT = head_module.DiT


class TinyVisualEncoder(nn.Module):
    rgb_key = ['head_cam']

    def __init__(self, **kwargs):
        super().__init__()

    def output_shape_meta(self):
        return (1, 1, 2, 8)

    def forward(self, obs, **kwargs):
        image = obs['head_cam']
        return image.mean(dim=(1, 2, 3)).view(-1, 1, 1).expand(-1, 2, 8)


def head(mode='tau_AdaRMS', flow='meanflow', prefix=1, self_first=True):
    model = DiT(3, 4, hidden_size=32, depth=2, num_heads=4, scene_cond_len=2,
                scene_pos_embed_config=[('image', (1, 1, 2))], dtype=torch.float32,
                flow_type=flow, time_injection=mode, n_obs_steps=prefix,
                self_attn_first=self_first)
    nn.init.normal_(model.final_layer.ffn_final.fc2.weight, std=.05)
    return model


class TimeInjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_config_defaults_and_time_scale(self):
        Config = policy_module.S1DitFlowMatchingConfig
        legacy = Config(state_key='state')
        self.assertEqual(legacy.time_injection, 'tau_token')
        self.assertEqual(legacy.flow_type, 'fm')
        ada = Config(state_key='state', flow_type='meanflow', time_injection='tau_AdaRMS')
        self.assertEqual(ada.meanflow['time_embed_scale'], 1.0)
        self.assertEqual(Config.from_customed_dict(ada.to_dict()).time_injection, 'tau_AdaRMS')
        old = Config(state_key='state', flow_type='meanflow')
        self.assertIsNone(old.meanflow['time_embed_scale'])
        with self.assertRaises(ValueError):
            Config(time_injection='invalid')
        with self.assertRaises(ValueError):
            Config(flow_type='meanflow', time_injection='tau_AdaRMS',
                   meanflow={'time_embed_scale': 1000})

    def test_legacy_token_default_has_identical_weights_and_output(self):
        args = dict(output_dim=3, horizon=4, hidden_size=32, depth=2, num_heads=4,
                    scene_cond_len=2, scene_pos_embed_config=[('image', (1, 1, 2))],
                    dtype=torch.float32, flow_type='fm')
        torch.manual_seed(42); old = DiT(**args)
        torch.manual_seed(42); explicit = DiT(**args, time_injection='tau_token')
        nn.init.normal_(old.final_layer.ffn_final.fc2.weight)
        explicit.load_state_dict(old.state_dict(), strict=True)
        x, t, scene = torch.randn(2, 4, 32), torch.tensor([120, 900]), torch.randn(2, 2, 32)
        torch.testing.assert_close(old(x, t, scene), explicit(x, t, scene), rtol=0, atol=0)
        self.assertFalse(any('modulation' in key or 'fusion' in key for key in old.state_dict()))
        with self.assertRaises(RuntimeError):
            head(flow='fm').load_state_dict(old.state_dict(), strict=True)

    def test_only_actions_fuse_and_layers_have_separate_gates(self):
        model = head(prefix=2)
        captured = []
        hook = model.blocks[0].register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        x, scene = torch.randn(2, 4, 32), torch.randn(2, 2, 32)
        out = model(x, torch.tensor([.8, .7]), scene, r=torch.tensor([.2, .3]))
        hook.remove()
        self.assertEqual(out.shape, (2, 4, 3))
        self.assertEqual(model.x_pos_embed.shape, (1, 4, 32))
        torch.testing.assert_close(captured[0][:, :2], x[:, :2] + model.x_pos_embed[:, :2])
        self.assertFalse(torch.equal(captured[0][:, 2:], x[:, 2:] + model.x_pos_embed[:, 2:]))
        self.assertIsNot(model.blocks[0].modulations[0][1].weight,
                         model.blocks[1].modulations[0][1].weight)
        block = model.blocks[0]
        for module in block.modulations:
            module[1].bias.data[64:].zero_()
        condition = torch.randn(2, 32)
        torch.testing.assert_close(block(x, scene, time_condition=condition), x, rtol=0, atol=0)

    def test_meanflow_jvp_matches_finite_difference_and_gradients(self):
        torch.manual_seed(7)
        model = head()
        for block in model.blocks:
            for module in block.modulations:
                nn.init.normal_(module[1].weight, std=.02)
        nn.init.normal_(model.final_modulation[1].weight, std=.02)
        adapter, scene_adapter = nn.Linear(3, 32), nn.Linear(6, 32)
        scene = scene_adapter(torch.randn(2, 2, 6))
        z, velocity = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
        mask = torch.ones_like(z, dtype=torch.bool); mask[:, 0] = False
        velocity = velocity * mask
        state = z[:, :1].clone()
        t, r = torch.tensor([.8, .7]), torch.tensor([.2, .3])
        def predict(value, time, target):
            value = torch.cat((state, value[:, 1:]), dim=1)
            return model(adapter(value), time, scene, r=target)
        with torch.no_grad(), mf.math_attention(model):
            _, jvp = torch.func.jvp(predict, (z, t, r),
                                    (velocity, torch.ones_like(t), torch.zeros_like(r)))
            eps = .001
            finite = (predict(z + eps*velocity, t + eps, r)
                      - predict(z - eps*velocity, t - eps, r)) / (2*eps)
        torch.testing.assert_close(jvp, finite, rtol=.03, atol=.003)
        result = mf.objective(predict, z, t, r, velocity, mask, mf.validate_options(), model)
        result['loss'].backward()
        parameters = [model.t_embedder.mlp[0].weight, model.r_embedder.mlp[0].weight,
                      model.action_time_fusion[0].weight, model.blocks[0].modulations[0][1].weight,
                      model.final_modulation[1].weight, adapter.weight, scene_adapter.weight]
        for parameter in parameters:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_kv_cache_roundtrip_and_fm(self):
        for flow in ('fm', 'meanflow'):
            for self_first in (True, False):
                model = head(flow=flow, self_first=self_first).eval()
                x, scene = torch.randn(2, 4, 32), torch.randn(2, 2, 32)
                t = torch.tensor([.7, .8]) if flow == 'meanflow' else torch.tensor([700, 800])
                r = torch.tensor([.2, .3]) if flow == 'meanflow' else None
                with torch.no_grad():
                    cache = model.prepare_kv_cache(scene)
                    a = model(x, t, scene, r=r)
                    b = model(x, t, None, r=r, kv_cache=cache)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                restored = head(flow=flow, self_first=self_first).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(a, restored(x, t, scene, r=r), rtol=0, atol=0)

    def test_bfloat16_jvp_trains_every_residual_gate(self):
        torch.manual_seed(42)
        model = head().to(torch.bfloat16)
        model.t_embedder.dtype = model.r_embedder.dtype = torch.bfloat16
        adapter = nn.Linear(3, 32).to(torch.bfloat16)
        scene = torch.randn(2, 2, 32, dtype=torch.bfloat16)
        z = torch.randn(2, 4, 3, dtype=torch.bfloat16)
        velocity = torch.randn_like(z); velocity[:, 0] = 0
        mask = torch.ones_like(z, dtype=torch.bool); mask[:, 0] = False
        state = z[:, :1].clone()
        t, r = torch.tensor([.8, .7]), torch.tensor([.2, .3])
        def predict(value, time, target):
            return model(adapter(torch.cat((state, value[:, 1:]), dim=1)), time, scene, r=target)
        result = mf.objective(predict, z, t, r, velocity, mask, mf.validate_options(), model)
        self.assertTrue(torch.isfinite(result['loss']))
        result['loss'].backward()
        for block in model.blocks:
            for modulation in block.modulations:
                linear = modulation[1]
                gate_gradient = linear.weight.grad[2 * model.hidden_size:]
                self.assertTrue(torch.isfinite(gate_gradient).all())
                self.assertGreater(float(gate_gradient.float().abs().sum()), 0)

    def test_policy_meanflow_loss_sampling_and_export_step(self):
        values = dict(state_key='state', supervision_key='action',
                      shape_meta={'obs': {'state': {'shape': [3], 'type': 'low_dim'}},
                                  'action': {'shape': [3]}},
                      noise_scheduler=dict(num_train_timesteps=1000, num_inference_timesteps=1,
                                           noise_beta_alpha=1.5, noise_beta_beta=1., noise_s=.999),
                      obs_encoder={}, hidden_size=32, depth=2, num_heads=4,
                      horizon=4, n_action_steps=3, n_obs_steps=1,
                      scene_adaptor='linear', state_adaptor='linear',
                      flow_type='meanflow', time_injection='tau_AdaRMS',
                      meanflow={'time_embed_scale': 1.0},
                      inference=dict(num_inference_timesteps=1, use_kv_cache=False,
                                     use_cuda_graph=False, precision='fp32'))
        with patch.object(policy_module, 'VisualEncoder', TinyVisualEncoder), patch.object(
            head_module, 'DiT', partial(DiT, dtype=torch.float32)
        ):
            policy = policy_module.S1DitFlowMatching(policy_module.S1DitFlowMatchingConfig.from_dict(values))
        nn.init.normal_(policy.model.final_layer.ffn_final.fc2.weight, std=.05)
        policy.normalizer.fit({k: torch.tensor([[-2., -1., -3.], [3., 4., 5.]])
                               for k in ('state', 'action')})
        states, actions = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
        obs = {'state': states, 'head_cam': torch.rand(2, 1, 3, 4, 4)}
        losses = policy.train().compute_loss({'obs': obs, 'action': {'action': actions}})
        self.assertTrue(torch.isfinite(losses['loss']))
        losses['loss'].backward()
        self.assertGreater(float(policy.model.action_time_fusion[0].weight.grad.abs().sum()), 0)
        self.assertGreater(float(policy.model.blocks[0].modulations[0][1].weight.grad.abs().sum()), 0)
        policy.eval()
        noise = torch.randn(2, 4, 3)
        with torch.no_grad():
            for steps in (1, 4):
                policy.num_inference_timesteps = steps
                result = policy.predict_action(obs, noise_trajectory=noise, use_kv_cache=True)
                self.assertEqual(result['action'].shape, (2, 3, 3))
                self.assertTrue(torch.isfinite(result['action']).all())
            # The TensorRT step wrapper shares the same DiT action/time path.
            Step = import_module('100hzs1.inference.export_tensorrt').Step
            state = policy.normalizer['state'].normalize(states[:, :1])
            scene = torch.randn(2, 2, 32)
            t, r, dt = torch.full((2,), .75), torch.full((2,), .5), torch.full((2, 1, 1), -.25)
            wrapped = Step(policy, cache=False)(noise, state, scene, t, r, dt)
            trajectory = torch.cat((state, torch.zeros_like(noise[:, 1:])), dim=1)
            value = torch.where(policy.action_mask, noise, trajectory)
            velocity = policy.model(policy.state_adaptor(value), t, scene, r=r)
            expected = torch.where(policy.action_mask, value + dt * velocity, trajectory)
            torch.testing.assert_close(wrapped, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
