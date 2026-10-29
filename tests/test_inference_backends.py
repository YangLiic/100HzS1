"""Deployment configuration, cache validity and physical observation contracts."""
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

options = import_module('100hzs1.common.inference')
api = import_module('100hzs1.inference')
native = import_module('100hzs1.inference.tensorrt')
common = import_module('100hzs1.inference.common')


def metadata():
    return dict(rgb_keys=['head_cam'], raw_image_shapes={'head_cam': [3, 2, 4]},
                n_obs_steps=3, n_image_obs_steps=1, horizon=5, n_action_steps=2,
                action_dim=3, state_key='state', supervision_key='action', flow_type='fm')


class BackendConfigurationTests(unittest.TestCase):
    def test_legacy_checkpoint_uses_eager_bf16_without_graph(self):
        cfg = {'inference': dict(use_cuda_graph=True, precision='bf16',
                                 num_inference_timesteps=4, use_kv_cache=False)}
        before = json.dumps(cfg, sort_keys=True)
        actual = options.load_inference_options(checkpoint_config=cfg)
        self.assertEqual((actual['backend'], actual['precision'], actual['use_cuda_graph']),
                         ('eager', 'bf16', False))
        self.assertEqual(actual['num_inference_timesteps'], 4)
        self.assertFalse(actual['use_kv_cache'])
        self.assertEqual(before, json.dumps(cfg, sort_keys=True))

    def test_explicit_yaml_and_call_select_backend_precision_and_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'runtime.yaml'
            p.write_text('backend: eager\nprecision: bf16\nuse_cuda_graph: true\n')
            actual = options.load_inference_options(p)
            self.assertEqual((actual['backend'], actual['precision'], actual['use_cuda_graph']),
                             ('eager', 'bf16', True))
            actual = options.load_inference_options(p, {'backend': 'TensorRT', 'precision': 'fp16', 'use_cuda_graph': False})
            self.assertEqual((actual['backend'], actual['precision'], actual['use_cuda_graph']),
                             ('tensorrt', 'fp16', False))

    def test_invalid_backend_and_precision_fail_before_loading_weights(self):
        for change in [dict(backend='auto'), dict(precision='int8'), dict(backend='tensorrt', precision='bf16'),
                       dict(tensorrt_batch_size=0), dict(tensorrt_batch_size=True),
                       dict(use_cuda_graph='false'), dict(tensorrt_engine_dir=3)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                options.load_inference_options(overrides=change)

    def test_factory_dispatches_only_the_selected_backend(self):
        eager = import_module('100hzs1.inference.eager')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{}')
            with patch.object(native, 'TensorRTPolicy') as trt, patch.object(eager, 'EagerPolicy') as torch:
                self.assertIs(api.load_inference_policy(root), torch.return_value)
                trt.assert_not_called()
                self.assertFalse(torch.call_args.args[2]['use_cuda_graph'])
                self.assertEqual(torch.call_args.args[2]['precision'], 'bf16')
            with patch.object(native, 'TensorRTPolicy') as trt, patch.object(eager, 'EagerPolicy') as torch:
                self.assertIs(api.load_inference_policy(root, overrides={'backend': 'tensorrt', 'precision': 'fp16'}), trt.return_value)
                torch.assert_not_called()

    def test_native_api_import_does_not_import_torch_or_optional_trt(self):
        script = "from importlib import import_module; import sys; import_module('100hzs1.inference'); assert 'torch' not in sys.modules; assert 'tensorrt' not in sys.modules"
        subprocess.run([sys.executable, '-c', script], check=True, capture_output=True)


class RuntimeContractTests(unittest.TestCase):
    def setUp(self):
        self.runtime = common.PolicyRuntime()
        self.runtime._initialize(metadata(), options.load_inference_options(), 'cuda:0')

    def test_state_reads_first_history_images_read_latest_and_rgb_is_not_flipped(self):
        image = np.zeros((1, 4, 2, 4, 3), dtype=np.uint8)
        image[:, -1, ..., 0] = 255
        state = np.arange(12, dtype=np.float32).reshape(1, 4, 3)
        result, batch = self.runtime._inputs(dict(head_cam=image, state=state))
        self.assertEqual(batch, 1)
        np.testing.assert_array_equal(result['state'], state[:, :3])
        self.assertEqual(result['head_cam'].shape, (1, 1, 3, 2, 4))
        np.testing.assert_array_equal(result['head_cam'][0, 0, 0], np.ones((2, 4)))
        np.testing.assert_array_equal(result['head_cam'][0, 0, 2], np.zeros((2, 4)))
        self.assertTrue(np.all(image[:, -1, ..., 0] == 255))

    def test_finite_shape_and_rgb_range_validation(self):
        good = dict(head_cam=np.zeros((1, 1, 3, 2, 4), np.float32), state=np.zeros((1, 3, 3), np.float32))
        for key, value in [('state', np.zeros((2, 3, 3))), ('state', np.full((1, 3, 3), np.nan)),
                           ('head_cam', np.full((1, 1, 3, 2, 4), 255, np.float32)),
                           ('head_cam', np.zeros((1, 0, 3, 2, 4)))]:
            with self.subTest(key=key, shape=value.shape), self.assertRaises(ValueError):
                self.runtime._inputs(dict(good, **{key: value}))

    def test_noise_is_seeded_identically_without_consuming_default_rng(self):
        other = common.PolicyRuntime()
        other._initialize(metadata(), options.load_inference_options(), 'cuda:0')
        np.testing.assert_array_equal(self.runtime._noise(1, None, None), other._noise(1, None, None))
        np.testing.assert_array_equal(self.runtime._noise(1, 123, None), other._noise(1, 123, None))
        self.runtime._noise(1, 999, None)
        np.testing.assert_array_equal(self.runtime._noise(1, None, None), other._noise(1, None, None))
        with self.assertRaises(ValueError):
            self.runtime._noise(1, None, np.zeros((1, 4, 3)))


class TensorRTCacheTests(unittest.TestCase):
    def test_atomic_build_warm_reuse_corruption_and_recipe_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint'
            checkpoint.mkdir()
            (checkpoint / 'config.json').write_text('{}')
            (checkpoint / 'model.safetensors').write_bytes(b'checkpoint fingerprint')
            opts = options.load_inference_options(overrides={'backend': 'tensorrt', 'precision': 'fp16', 'tensorrt_engine_dir': str(root / 'cache')})
            hardware = dict(gpu='fixture', sm=[8, 9], tensorrt='fixture', cuda=12080, driver=12090)
            def child(command, **kwargs):
                output = Path(command[command.index('--output')+1])
                for scope in ['encoder', 'step', 'post']:
                    (output / f'{scope}.plan').write_bytes(scope.encode())
                (output / 'metadata.json').write_text(json.dumps(metadata()))
                return SimpleNamespace(returncode=0)
            with patch.object(native, 'source_hash', return_value='source-v1'), patch.object(native.subprocess, 'run', side_effect=child) as build:
                path, info = native.ensure_engines(checkpoint, opts, hardware, 'cuda:0')
                self.assertEqual(info, metadata())
                self.assertEqual(build.call_count, 1)
                # NFE and CUDA Graph are execution settings; generic step engines are reused.
                same, _ = native.ensure_engines(checkpoint, dict(opts, num_inference_timesteps=4, use_cuda_graph=True), hardware, 'cuda:0')
                self.assertEqual(path, same)
                self.assertEqual(build.call_count, 1)
                changed, _ = native.ensure_engines(checkpoint, dict(opts, precision='fp32'), hardware, 'cuda:0')
                self.assertNotEqual(path, changed)
                self.assertEqual(build.call_count, 2)
                (path / 'encoder.plan').write_bytes(b'corrupt')
                with self.assertRaisesRegex(RuntimeError, 'modified'):
                    native.ensure_engines(checkpoint, opts, hardware, 'cuda:0')

    def test_failed_builder_publishes_no_cache_and_never_falls_back(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text('{}')
            (root / 'model.safetensors').write_bytes(b'weights')
            cache = root / 'cache'
            opts = options.load_inference_options(overrides={'backend': 'tensorrt', 'precision': 'fp16', 'tensorrt_engine_dir': str(cache)})
            with patch.object(native, 'source_hash', return_value='source'), patch.object(native.subprocess, 'run', return_value=SimpleNamespace(returncode=17)):
                with self.assertRaisesRegex(RuntimeError, 'failed \(17\)'):
                    native.ensure_engines(root, opts, {}, 'cuda:0')
            self.assertFalse(list(cache.rglob('manifest.json')))
            self.assertTrue(list(cache.glob('*.last-build.log')))


if __name__ == '__main__':
    unittest.main()
