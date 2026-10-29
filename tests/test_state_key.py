"""State-condition naming and compatibility with legacy checkpoint configs."""
from functools import partial
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from torch import nn


MODULES = (
    ('100hzs1.models.policy.flow_matching', 'S1DitFlowMatching'),
    ('rtpolicy.models.policy.rtpolicy', 'RTPolicyDitFlowMatching'),
)


def config_values(key='state', supervision='action'):
    return dict(
        state_key=key, supervision_key=supervision,
        shape_meta={'obs': {'state': {'shape': [2], 'type': 'low_dim'},
                            'delta_state': {'shape': [2], 'type': 'low_dim'}},
                    'action': {'shape': [2]}},
        noise_scheduler=dict(num_train_timesteps=1000, num_inference_timesteps=2,
                             noise_beta_alpha=1.5, noise_beta_beta=1., noise_s=.999),
        obs_encoder={}, hidden_size=32, depth=1, num_heads=4,
        horizon=4, n_action_steps=3, n_obs_steps=1,
        scene_adaptor='linear', state_adaptor='linear',
        inference=dict(num_inference_timesteps=2, use_kv_cache=False,
                       use_cuda_graph=False, precision='fp32'),
    )


class TinyVisualEncoder(nn.Module):
    rgb_key = ['head_cam']

    def __init__(self, **kwargs):
        super().__init__()

    def output_shape_meta(self):
        return (1, 1, 2, 8)

    def forward(self, obs, **kwargs):
        image = obs['head_cam']
        return image.mean(dim=(1, 2, 3)).view(-1, 1, 1).expand(-1, 2, 8)


class StateKeyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_legacy_config_save_and_reload_use_canonical_name(self):
        for package, name in MODULES:
            config_class = getattr(import_module(package + '.dit_fm_policy'), name + 'Config')
            for key in ('state', 'delta_state'):
                with self.subTest(package=package, key=key):
                    values = config_values(key)
                    values['action_key'] = values.pop('state_key')
                    config = config_class.from_dict(values)
                    self.assertEqual(config.state_key, key)
                    self.assertNotIn('action_key', config.to_dict())
                    with TemporaryDirectory() as directory:
                        config.save_pretrained(directory)
                        restored = config_class.from_pretrained(directory, local_files_only=True)
                        self.assertEqual(restored.state_key, key)
                        self.assertEqual(restored.supervision_key, 'action')
                        self.assertNotIn('action_key', (Path(directory) / 'config.json').read_text())

    def test_conflicting_names_fail_instead_of_changing_condition(self):
        for package, name in MODULES:
            config_class = getattr(import_module(package + '.dit_fm_policy'), name + 'Config')
            with self.subTest(package=package):
                values = config_values()
                values['action_key'] = 'state'
                self.assertEqual(config_class(**values).state_key, 'state')
                values['action_key'] = 'delta_state'
                with self.assertRaisesRegex(ValueError, 'must agree'):
                    config_class(**values)

    def test_training_inference_and_export_match_legacy_configs(self):
        for package, name in MODULES:
            module = import_module(package + '.dit_fm_policy')
            dit_module = import_module(package + '.module.conditional_dit_head')
            config_class, policy_class = getattr(module, name + 'Config'), getattr(module, name)
            namespace = package.split('.')[0]
            export_module = import_module(namespace + '.trainer.onnx')
            for key, supervision in (('state', 'action'), ('delta_state', None)):
                with self.subTest(package=package, key=key, supervision=supervision):
                    values = config_values(key, supervision)
                    legacy = dict(values)
                    legacy['action_key'] = legacy.pop('state_key')
                    policies = []
                    with patch.object(module, 'VisualEncoder', TinyVisualEncoder), patch.object(
                        dit_module, 'DiT', partial(dit_module.DiT, dtype=torch.float32)
                    ):
                        for config in (values, legacy):
                            torch.manual_seed(17)
                            policy = policy_class(config_class.from_dict(config)).float()
                            nn.init.normal_(policy.model.final_layer.ffn_final.fc2.weight, std=.1)
                            policy.normalizer.fit({k: torch.tensor([[-2., -1.], [3., 4.]])
                                                   for k in ('state', 'delta_state', 'action')})
                            policies.append(policy)
                    self.assertEqual(policies[0].state_key, key)
                    self.assertEqual(export_module._get_state_key(policies[0]), key)
                    self.assertEqual(policies[0].supervision_key, supervision or key)
                    policies[1].load_state_dict(policies[0].state_dict(), strict=True)
                    states, actions = torch.randn(2, 4, 2), torch.randn(2, 4, 2) + 2
                    obs = {'state': states, key: states, 'head_cam': torch.rand(2, 1, 3, 4, 4)}
                    batch = {'obs': obs, 'action': {key: states, 'action': actions}}
                    losses, predictions = [], []
                    for policy in policies:
                        torch.manual_seed(23)
                        loss = policy.train().compute_loss(batch)
                        self.assertTrue(torch.isfinite(loss))
                        loss.backward()
                        self.assertTrue(torch.isfinite(policy.model.final_layer.ffn_final.fc2.weight.grad).all())
                        losses.append(loss.detach())
                        torch.manual_seed(29)
                        predictions.append(policy.eval().predict_action(obs)['action'])
                    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
                    torch.testing.assert_close(predictions[0], predictions[1], rtol=0, atol=0)
                    self.assertEqual(predictions[0].shape, (2, 3, 2))
                    if key != "state":
                        continue  # The ONNX wrapper accepts absolute state inputs.
                    wrapper = export_module.PolicyExportWrapper(policies[0])
                    torch.manual_seed(29)
                    noise = torch.randn(2, 4, 2)
                    exported, _ = wrapper(head_cam=obs['head_cam'], state=states,
                                          noise_trajectory=noise)
                    torch.testing.assert_close(exported, predictions[0], rtol=0, atol=0)
