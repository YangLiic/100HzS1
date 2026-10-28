"""Check namespace migration without loading models or rewriting checkpoints."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from vfa.common.checkpoint import (
    canonical_class_path,
    load_checkpoint_config,
    normalize_checkpoint_metadata,
)


class CheckpointNamespaceTests(unittest.TestCase):
    def test_rtpolicy_metadata_is_mapped_without_mutating_the_input(self):
        original = {
            "pkg_map": {
                "AutoConfig": "rtpolicy.models.RTPolicyDitFlowMatchingConfig",
                "AutoModel": "rtpolicy.models.policy.rtpolicy.dit_fm_policy.RTPolicyDitFlowMatching",
            },
            "auto_map": {
                "AutoConfig": "dit_fm_policy.RTPolicyDitFlowMatchingConfig",
                "AutoModel": ["dit_fm_policy.RTPolicyDitFlowMatching", None],
            },
            "architectures": ["RTPolicyDitFlowMatching"],
            "dataset_id": "rtpolicy/example",
            "backbone_path": "/weights/rtpolicy/dinov3",
        }
        saved = deepcopy(original)
        mapped = normalize_checkpoint_metadata(original)
        self.assertEqual(original, saved)
        self.assertEqual(mapped["pkg_map"]["AutoConfig"], "vfa.models.VFADitFlowMatchingConfig")
        self.assertEqual(
            mapped["pkg_map"]["AutoModel"],
            "vfa.models.policy.vfa.dit_fm_policy.VFADitFlowMatching",
        )
        self.assertEqual(mapped["auto_map"]["AutoModel"], ["dit_fm_policy.VFADitFlowMatching", None])
        self.assertEqual(mapped["architectures"], ["VFADitFlowMatching"])
        self.assertEqual(mapped["dataset_id"], saved["dataset_id"])
        self.assertEqual(mapped["backbone_path"], saved["backbone_path"])

    def test_legacy_dataset_and_policy_paths(self):
        cases = {
            "drrm.models.policy.vodpp.dit_fm_policy.VODPPlusDitFlowMatching":
                "vfa.models.policy.vfa.dit_fm_policy.VFADitFlowMatching",
            "drrm.datasets.drrm_dataset.DRRMDataset": "vfa.datasets.vfa_dataset.VFADataset",
            "rtpolicy.datasets.rtpolicy_dataset_lance.RTPolicyDataset":
                "vfa.datasets.vfa_dataset_lance.VFADataset",
        }
        for old, new in cases.items():
            with self.subTest(path=old):
                self.assertEqual(canonical_class_path(old), new)

    def test_current_metadata_is_idempotent(self):
        config = {"pkg_map": {"AutoModel": "vfa.models.VFADitFlowMatching"}}
        self.assertEqual(normalize_checkpoint_metadata(config), config)

    def test_legacy_visual_mode_is_loading_only_and_explicit_native_is_preserved(self):
        legacy = {"obs_encoder": {"emvis_config": {"backbone_target": "efficientvit"}}}
        saved = deepcopy(legacy)
        mapped = normalize_checkpoint_metadata(legacy)
        self.assertEqual(legacy, saved)
        self.assertEqual(mapped["obs_encoder"]["emvis_config"]["efficientvit_fusion_mode"], "projected")
        self.assertEqual(normalize_checkpoint_metadata(mapped), mapped)
        native = deepcopy(legacy)
        native["obs_encoder"]["emvis_config"]["efficientvit_fusion_mode"] = "native"
        self.assertEqual(normalize_checkpoint_metadata(native), native)

    def test_other_packages_are_not_redirected(self):
        path = "another_package.models.RTPolicyDitFlowMatching"
        self.assertEqual(canonical_class_path(path), path)

    def test_reading_a_checkpoint_preserves_its_files(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            weights = Path(directory) / "model.safetensors"
            config.write_text(json.dumps({
                "pkg_map": {"AutoModel": "rtpolicy.models.RTPolicyDitFlowMatching"},
            }))
            weights.write_bytes(b"unchanged weight bytes")
            before = config.read_bytes(), weights.read_bytes()
            mapped = load_checkpoint_config(directory)
            self.assertEqual(mapped["pkg_map"]["AutoModel"], "vfa.models.VFADitFlowMatching")
            self.assertEqual((config.read_bytes(), weights.read_bytes()), before)


if __name__ == "__main__":
    unittest.main()
