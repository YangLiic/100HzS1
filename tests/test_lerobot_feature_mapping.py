"""Check raw LeRobot naming, independent state/action targets and cache parity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from test_video_episode_cache import _write_fixture, _make_dataset, _legacy_frames
from vfa.datasets.managed_episode_cache import close_managed_episode_caches


class LeRobotFeatureMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory(prefix="vfa_mapping_test_")
        cls.root = Path(cls.directory.name)
        _write_fixture(cls.root)
        raw_names = {"state": "observation.state", "head_cam": "observation.images.head",
                     "wrist_cam": "observation.images.left"}
        info_path = cls.root / "meta/info.json"
        info = json.loads(info_path.read_text())
        info["features"] = {raw_names.get(key, key): value for key, value in info["features"].items()}
        info["features"]["annotation.human.task_description"] = {"dtype": "int64", "shape": [1], "names": None}
        info_path.write_text(json.dumps(info))
        for parquet in cls.root.glob("data/*/*.parquet"):
            table = pq.read_table(parquet)
            table = table.rename_columns([raw_names.get(key, key) for key in table.column_names])
            table = table.append_column("annotation.human.task_description", pa.array([0] * table.num_rows))
            pq.write_table(table, parquet)
        for old, new in raw_names.items():
            directory = cls.root / "videos/chunk-000" / old
            if directory.exists():
                directory.rename(directory.with_name(new))
        stats_path = cls.root / "meta/episodes_stats.jsonl"
        stats_path.write_text("".join(json.dumps({
            **entry, "stats": {raw_names.get(key, key): value for key, value in entry["stats"].items()}
        }) + "\n" for entry in map(json.loads, stats_path.read_text().splitlines())))
        cls.mapping = {value: key for key, value in raw_names.items()}

    @classmethod
    def tearDownClass(cls):
        close_managed_episode_caches()
        cls.directory.cleanup()

    def make(self, **kwargs):
        return _make_dataset(self.root, feature_key_map=self.mapping, bgr_to_rgb=True, **kwargs)

    def test_selected_model_camera_maps_to_raw_source_and_preserves_action_column(self):
        baseline = self.make(cache_to_ram=False)
        with patch.object(LeRobotDataset, "_query_videos", _legacy_frames):
            expected = baseline[3]
            for decoded in [False, True]:
                actual = self.make(cache_decoded_video=decoded)[3]
                for section in ["obs", "action"]:
                    self.assertEqual(expected[section].keys(), actual[section].keys())
                    for key in actual[section]:
                        torch.testing.assert_close(actual[section][key], expected[section][key], rtol=0, atol=0)
            self.assertIn("head_cam", expected["obs"])
            self.assertNotIn("wrist_cam", expected["obs"])
            self.assertNotIn("observation.state", expected["obs"])
            self.assertFalse(torch.equal(expected["obs"]["state"], expected["action"]["action"]))
            # The raw action is preserved; the measured state remains a separate condition.
            torch.testing.assert_close(expected["action"]["action"], torch.tensor([[3., 4.], [4., 5.], [5., 6.], [6., 7.]]))

    def test_validation_preserves_mapping_color_and_state_history(self):
        train = self.make(val_ratio=0.5)
        validation = train.get_validation_dataset()
        self.assertEqual(validation.feature_key_map, self.mapping)
        self.assertTrue(validation.bgr_to_rgb)
        self.assertEqual(validation.n_obs_steps, train.n_obs_steps)
        self.assertFalse(validation.cache_to_ram)
        with patch.object(LeRobotDataset, "_query_videos", _legacy_frames):
            self.assertIn("head_cam", validation[0]["obs"])

    def test_normalizer_keeps_distinct_training_state_and_action_statistics(self):
        train = self.make(episodes=[0])
        normalizer = train.get_normalizer()
        self.assertIn("state", normalizer.params_dict)
        self.assertIn("action", normalizer.params_dict)
        self.assertNotIn("observation.state", normalizer.params_dict)
        self.assertNotIn("annotation.human.task_description", normalizer.params_dict)
        rows = train.base_hf_dataset[:]
        states = torch.stack(rows["observation.state"])
        actions = torch.stack(rows["action"])
        torch.testing.assert_close(normalizer["state"].normalize(states).amin(0), -torch.ones(2))
        torch.testing.assert_close(normalizer["action"].normalize(actions).amin(0), -torch.ones(2))
        self.assertFalse(torch.equal(normalizer["state"].params_dict["offset"],
                                     normalizer["action"].params_dict["offset"]))

    def test_invalid_and_colliding_mappings_fail_early(self):
        for mapping in [{"missing": "state"}, {"observation.state": "action"},
                        {"observation.state": "x", "action": "x"}]:
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                _make_dataset(self.root, feature_key_map=mapping)
