from importlib import import_module as _import_module
import importlib.util
import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "simulation/robotwin2.0/policy/100hzs1"
sys.path.insert(0, str(POLICY.parents[1]))
CAMERAS = _import_module('policy.100hzs1.deploy_policy').CAMERAS
ObservationHistory = _import_module('policy.100hzs1.deploy_policy').ObservationHistory
encode_obs = _import_module('policy.100hzs1.deploy_policy').encode_obs
evaluate = _import_module('policy.100hzs1.deploy_policy').eval

spec = importlib.util.spec_from_file_location(
    "s1_direct_launcher", POLICY / "evaluate.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def observation(step=0):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[..., 0] = 255
    return {
        "joint_action": {"vector": np.full(14, step, dtype=np.float32)},
        "observation": {name: {"rgb": image.copy()} for name in CAMERAS.values()},
    }


class TestRobotwinDirect(unittest.TestCase):
    def test_online_rgb_and_jpeg_preserve_channels(self):
        for jpeg in [False, True]:
            result = encode_obs(
                observation(), ["head_cam", "left_cam", "right_cam"], jpeg
            )
            for key in ["head_cam", "left_cam", "right_cam"]:
                self.assertGreater(result[key][..., 0].mean(), 250)
                self.assertLess(result[key][..., 2].mean(), 5)
            self.assertEqual(result["state"].shape, (14,))

    def test_history_padding_and_chunk_boundary(self):
        class Model:
            rgb_keys = ["head_cam"]
            history = ObservationHistory(3, 1, rgb_keys)
            inputs = []

            def observe(self, obs):
                self.history.append(encode_obs(obs, self.rgb_keys, False))

            def predict(self):
                self.inputs.append(self.history.stacked()["state"][:, 0].tolist())
                return np.zeros((5, 14))

        class Env:
            step = 0
            eval_success = False

            def take_action(self, action, action_type):
                self.step += 1
                assert action_type == "qpos"

            def get_obs(self):
                return observation(self.step)

        model, env = Model(), Env()
        evaluate(env, model, env.get_obs())
        evaluate(env, model, env.get_obs())
        self.assertEqual(model.inputs, [[0, 0, 0], [3, 4, 5]])
        self.assertEqual(model.history.stacked()["head_cam"].shape, (1, 240, 320, 3))

    def test_seed_provenance_requires_valid_metadata_for_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "seeds.txt"
            path.write_text("100000\n100002\n")
            with self.assertRaises(FileNotFoundError):
                launcher.seed_provenance(path, "blocks_ranking_rgb", "demo_clean", True)
            self.assertFalse(
                launcher.seed_provenance(path, "blocks_ranking_rgb", "demo_clean")[
                    "verified"
                ]
            )
            metadata = {
                "verified": True,
                "task": "blocks_ranking_rgb",
                "task_config": "demo_clean",
                "count": 2,
                "seed_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            path.with_suffix(".expert.json").write_text(json.dumps(metadata))
            self.assertTrue(
                launcher.seed_provenance(
                    path, "blocks_ranking_rgb", "demo_clean", True
                )["verified"]
            )
            with self.assertRaises(ValueError):
                launcher.seed_provenance(path, "another_task", "demo_clean", True)
            path.write_text("100000\n100003\n")
            with self.assertRaises(ValueError):
                launcher.seed_provenance(path, "blocks_ranking_rgb", "demo_clean", True)

    def test_partition_and_reject_incomplete_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seeds = root / "seeds.txt"
            seeds.write_text("".join(f"{s}\n" for s in range(100000, 100100)))
            groups = launcher.split_seeds(seeds, 100, 4)
            self.assertEqual([len(g) for g in groups], [25] * 4)
            records = _import_module("100hzs1.evaluation.records")
            (root / "logs").mkdir()
            for gpu, group in enumerate(groups):
                journal = root / "logs" / f"gpu{gpu}.jsonl"
                for seed in group:
                    records.append_event(journal, "episode", record=dict(seed=seed, success=False))
                records.append_event(journal, "completed")
            self.assertEqual(len(launcher.merge_results(root, groups, [0, 1, 2, 3])), 100)
            (root / "logs/gpu2.jsonl").write_text("")
            with self.assertRaises(RuntimeError):
                launcher.merge_results(root, groups, [0, 1, 2, 3])



if __name__ == "__main__":
    unittest.main()
