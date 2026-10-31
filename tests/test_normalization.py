"""Normalization choices, selected-frame statistics and old-checkpoint compatibility."""
from functools import partial
from importlib import import_module
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from safetensors.torch import load_model, save_model

norms = import_module("100hzs1.common.norm")
common = import_module("100hzs1.common.normalizer")
policy_normalizer = import_module("100hzs1.models.policy.flow_matching.common.normalizer")
policy_module = import_module("100hzs1.models.policy.flow_matching.dit_fm_policy")
head_module = import_module("100hzs1.models.policy.flow_matching.module.conditional_dit_head")
loader = import_module("100hzs1.models.base_runner")
from test_state_key import config_values, TinyVisualEncoder


def statistics():
    return {k: np.array(v, dtype=np.float32) for k, v in dict(
        min=[0, 4], max=[10, 4], mean=[5, 4], std=[2, 0], q01=[1, 4], q99=[9, 4]
    ).items()}


class NormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_four_choices_and_invalid_names(self):
        self.assertEqual(norms.NORM_MODES, ("minmax", "minmax_expanded", "pi", "groot"))
        self.assertEqual(norms.validate_norm(), "minmax")
        for value in [None, {}, "invalid", "MINMAX"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                norms.validate_norm(value)
        for value in [np.empty((0, 2)), np.array([[np.nan, 0]]), np.zeros((2, 2, 2))]:
            with self.assertRaises(ValueError):
                norms.normalizer_from_data(value, "pi")

    def test_ranges_constants_inverse_and_gradients(self):
        x = torch.tensor([[[-1., 4.], [0., 4.], [5., 4.], [10., 4.], [11., 4.]]], requires_grad=True)
        for mode in norms.NORM_MODES:
            with self.subTest(mode=mode):
                n = norms.make_normalizer(statistics(), mode)
                self.assertFalse(any(p.requires_grad for p in n.parameters()))
                y = n.normalize(x)
                if mode == "minmax":
                    torch.testing.assert_close(y[0, :, 0], torch.tensor([-1.2, -1., 0., 1., 1.2]))
                elif mode == "minmax_expanded":
                    torch.testing.assert_close(y[0, :, 0], torch.tensor([-.96, -.8, 0., .8, .96]))
                elif mode == "pi":
                    torch.testing.assert_close(y[..., 1], -torch.ones_like(y[..., 1]))
                    self.assertGreater(float(y[..., 0].max()), 1.)
                else:
                    self.assertLessEqual(float(y.abs().max()), 1.)
                    torch.testing.assert_close(y[..., 1], torch.zeros_like(y[..., 1]))
                physical = n.unnormalize(y)
                if mode != "groot":
                    torch.testing.assert_close(physical, x, atol=2e-6, rtol=0)
                else:
                    torch.testing.assert_close(physical[0, :, 0], torch.tensor([1., 1., 5., 9., 9.]))
                    torch.testing.assert_close(physical[..., 1], torch.full_like(physical[..., 1], 4.))
                self.assertTrue(torch.isfinite(torch.autograd.grad(physical.sum(), x)[0]).all())

    def test_old_schema_has_identical_keys_and_bitwise_arithmetic(self):
        old = import_module("100hzs1.common.normalize_util").get_range_normalizer_from_stat(
            {k: v for k, v in statistics().items() if k not in {"q01", "q99"}})
        new = norms.make_normalizer(statistics())
        self.assertEqual(set(old.state_dict()), set(new.state_dict()))
        for key in old.state_dict():
            torch.testing.assert_close(old.state_dict()[key], new.state_dict()[key], rtol=0, atol=0)
        x = torch.randn(2, 3, 2)
        expected = x * old.params_dict["scale"] + old.params_dict["offset"]
        torch.testing.assert_close(new.normalize(x), expected, rtol=0, atol=0)
        expected_inverse = (expected - old.params_dict["offset"]) / old.params_dict["scale"]
        torch.testing.assert_close(new.unnormalize(expected), expected_inverse, rtol=0, atol=0)

    def test_cross_class_strict_safetensors_reload_retains_all_modes(self):
        x = torch.tensor([[[-10., 4.], [15., 4.]]])
        for mode in norms.NORM_MODES:
            source = common.LinearNormalizer()
            source["state"] = norms.make_normalizer(statistics(), mode)
            target = nn.Module()
            target.normalizer = policy_normalizer.LinearNormalizer()
            target.normalizer.load_state_dict(source.state_dict(), strict=True)
            with TemporaryDirectory() as directory:
                file = Path(directory) / "model.safetensors"
                save_model(target, str(file))
                restored = nn.Module()
                restored.normalizer = policy_normalizer.LinearNormalizer()
                load_model(restored, str(file), strict=True)
                torch.testing.assert_close(restored.normalizer["state"].normalize(x),
                                           source["state"].normalize(x), rtol=0, atol=0)
                torch.testing.assert_close(restored.normalizer["state"].unnormalize(x),
                                           source["state"].unnormalize(x), rtol=0, atol=0)

    def test_full_policy_loads_old_config_without_norm_and_preserves_output(self):
        values = config_values()
        values["pkg_map"] = {"AutoConfig": "100hzs1.models.S1DitFlowMatchingConfig",
                             "AutoModel": "100hzs1.models.S1DitFlowMatching"}
        with patch.object(policy_module, "VisualEncoder", TinyVisualEncoder), patch.object(
            head_module, "DiT", partial(head_module.DiT, dtype=torch.float32)
        ), TemporaryDirectory() as directory:
            config = policy_module.S1DitFlowMatchingConfig.from_dict(values)
            self.assertEqual(config.norm, "minmax")
            model = policy_module.S1DitFlowMatching(config).float().eval()
            model.normalizer.fit({k: torch.tensor([[-2., -1.], [3., 4.]]) for k in ("state", "action")})
            nn.init.normal_(model.model.final_layer.ffn_final.fc2.weight, std=.1)
            model.save_pretrained(directory)
            file = Path(directory) / "config.json"
            payload = json.loads(file.read_text())
            payload.pop("norm", None)
            file.write_text(json.dumps(payload))
            legacy = loader.load_policy(directory, device="cpu").float().eval()
            payload["norm"] = "minmax"
            file.write_text(json.dumps(payload))
            explicit = loader.load_policy(directory, device="cpu").float().eval()
            self.assertEqual(legacy.config.norm, "minmax")
            obs = {"state": torch.rand(1, 1, 2), "head_cam": torch.rand(1, 1, 3, 4, 4)}
            noise = torch.randn(1, 4, 2)
            with torch.no_grad():
                a = legacy.predict_action(obs, noise_trajectory=noise)["action"]
                b = explicit.predict_action(obs, noise_trajectory=noise)["action"]
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(set(model.state_dict()), set(legacy.state_dict()))

    def test_full_policy_config_reload_and_clipped_output(self):
        values = config_values()
        values.update(norm="groot", pkg_map={"AutoConfig": "100hzs1.models.S1DitFlowMatchingConfig",
                                           "AutoModel": "100hzs1.models.S1DitFlowMatching"})
        with patch.object(policy_module, "VisualEncoder", TinyVisualEncoder), patch.object(
            head_module, "DiT", partial(head_module.DiT, dtype=torch.float32)
        ), TemporaryDirectory() as directory:
            model = policy_module.S1DitFlowMatching(policy_module.S1DitFlowMatchingConfig.from_dict(values)).float().eval()
            normalizer = common.LinearNormalizer()
            for key in ("state", "action"):
                normalizer[key] = norms.make_normalizer(statistics(), "groot")
            model.set_normalizer(normalizer)
            model.save_pretrained(directory)
            restored = loader.load_policy(directory, device="cpu").float().eval()
            self.assertEqual(restored.config.norm, "groot")
            obs = {"state": torch.rand(1, 1, 2), "head_cam": torch.rand(1, 1, 3, 4, 4)}
            with torch.no_grad():
                action = restored.predict_action(obs, noise_trajectory=torch.full((1, 4, 2), 100.))["action"]
            self.assertGreaterEqual(float(action[..., 0].min()), 1.)
            self.assertLessEqual(float(action[..., 0].max()), 9.)
            torch.testing.assert_close(action[..., 1], torch.full_like(action[..., 1], 4.))

    def test_selected_training_frames_and_independent_feature_statistics(self):
        dataset_class = import_module("100hzs1.datasets.dataset").S1Dataset
        class FrameTable:
            def __init__(self, values):
                self.values = values
            def select_columns(self, keys):
                return FrameTable({key: self.values[key] for key in keys})
            def __getitem__(self, item):
                return self.values
        dataset = dataset_class.__new__(dataset_class)
        dataset.feature_key_map = {"observation.state": "state"}
        dataset.hf_dataset = FrameTable({
            "observation.state": np.array([[0., 0.], [1., 1.], [2., 2.]], dtype=np.float32),
            "action": np.array([[10., 10.], [20., 20.], [30., 30.]], dtype=np.float32)})
        # Only selected training rows are exposed to the normalizer.
        for mode in ("minmax_expanded", "pi", "groot"):
            state = dataset._create_feature_normalizer("observation.state", {"dtype": "float32"}, norm=mode)
            action = dataset._create_feature_normalizer("action", {"dtype": "float32"}, norm=mode)
            torch.testing.assert_close(state.params_dict["input_stats"]["max"], torch.tensor([2.25, 2.25])
                                       if mode == "minmax_expanded" else torch.tensor([2., 2.]))
            self.assertFalse(torch.equal(state.params_dict["offset"], action.params_dict["offset"]))

    def test_model_rejects_invalid_or_unsupported_mode(self):
        with self.assertRaises(ValueError):
            policy_module.S1DitFlowMatchingConfig(norm="invalid")
        with self.assertRaisesRegex(ValueError, "absolute joint"):
            policy_module.S1DitFlowMatchingConfig(state_key="delta_state", norm="pi")

    def test_trainer_routes_norm_and_keeps_checkpoint_statistics(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        apply = import_module("100hzs1.trainer.train").apply_dataset_normalizer
        dataset = SimpleNamespace(get_normalizer=Mock(return_value=common.LinearNormalizer()))
        policy = SimpleNamespace(config=SimpleNamespace(norm="minmax"), set_normalizer=Mock())
        ema = SimpleNamespace(set_normalizer=Mock())
        logger = Mock()
        apply(SimpleNamespace(), dataset, policy, ema, logger)
        dataset.get_normalizer.assert_called_once_with(norm="minmax")
        dataset.get_normalizer.reset_mock()
        apply(SimpleNamespace(norm="pi"), dataset, policy, ema, logger)
        dataset.get_normalizer.assert_called_once_with(norm="pi")
        dataset.get_normalizer.reset_mock()
        apply(SimpleNamespace(norm="groot", normalizer_source="checkpoint", checkpoint_dir="/existing"),
              dataset, policy, ema, logger)
        dataset.get_normalizer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
