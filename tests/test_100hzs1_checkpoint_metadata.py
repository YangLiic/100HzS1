"""Current namespace metadata and shared runtime YAML contract."""
from importlib import import_module
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
c = import_module("100hzs1.common.checkpoint")
i = import_module("100hzs1.common.inference")

class CurrentMetadataTests(unittest.TestCase):
    def test_current_metadata_is_checked_without_mutation(self):
        value = {"pkg_map": {"AutoConfig": "100hzs1.models.S1DitFlowMatchingConfig",
                             "AutoModel": "100hzs1.models.S1DitFlowMatching"}}
        before = deepcopy(value)
        self.assertEqual(c.prepare_checkpoint_metadata(value), value)
        self.assertEqual(value, before)

    def test_other_namespaces_are_rejected(self):
        with self.assertRaises(ValueError):
            c.validate_class_path("other_package.models.Policy")

    def test_default_yaml_and_explicit_overrides(self):
        self.assertEqual(i.load_inference_options(), dict(num_inference_timesteps=10,
            use_kv_cache=True, use_cuda_graph=False, precision="bf16", backend="eager",
            tensorrt_batch_size=1, tensorrt_engine_dir=None))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"runtime.yaml"
            p.write_text("num_inference_timesteps: 4\nuse_kv_cache: false\nuse_cuda_graph: false\n")
            v=i.load_inference_options(p)
            self.assertFalse(v["use_kv_cache"])
            self.assertFalse(v["use_cuda_graph"])
            self.assertEqual(v["num_inference_timesteps"],4)
            self.assertTrue(i.load_inference_options(p, {"use_kv_cache":True})["use_kv_cache"])

    def test_invalid_options_fail_early(self):
        for v in [{"num_inference_timesteps":0},{"use_kv_cache":"false"},{"unknown":1}]:
            with self.assertRaises(ValueError):i.load_inference_options(overrides=v)
