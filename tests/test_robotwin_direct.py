import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "simulation/robotwin2.0/policy/VFA"
sys.path.insert(0, str(POLICY.parent))
from VFA.deploy_policy import CAMERAS, ObservationHistory, encode_obs, eval as evaluate

spec = importlib.util.spec_from_file_location(
    "vfa_direct_launcher", POLICY / "evaluate.py"
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

    def test_partition_and_reject_incomplete_results(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seeds = root / "seeds.txt"
            seeds.write_text("".join(f"{s}\n" for s in range(100000, 100100)))
            groups = launcher.split_seeds(seeds, 100, 4)
            self.assertEqual([len(g) for g in groups], [25] * 4)
            for gpu, group in enumerate(groups):
                part = root / f"gpu{gpu}"
                part.mkdir()
                (part / "EVALUATION_COMPLETED").touch()
                (part / "_eval_seeds.txt").write_text("".join(f"{s}\n" for s in group))
                (part / "result.json").write_text(
                    json.dumps(
                        {"results": [{"seed": s, "success": False} for s in group]}
                    )
                )
            self.assertEqual(
                len(launcher.merge_results(root, groups, [0, 1, 2, 3])), 100
            )
            (root / "gpu2/result.json").write_text(json.dumps({"results": []}))
            with self.assertRaises(RuntimeError):
                launcher.merge_results(root, groups, [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
