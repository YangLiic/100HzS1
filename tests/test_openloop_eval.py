"""Check the offline/online boundary: image color, temporal alignment, metric units."""

import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location(
    "s1_openloop", Path(__file__).parents[1] / "scripts/openloop_eval/evaluate.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OpenLoopTest(unittest.TestCase):
    def test_cuda_graph_cli_default_and_opt_out(self):
        required = ['--checkpoint', '/unused/checkpoint', '--output', '/unused/output']
        self.assertIsNone(module.parser().parse_args(required).cuda_graph)
        self.assertFalse(module.parser().parse_args(required + ['--no-cuda-graph']).cuda_graph)

    def test_default_flat_png_and_checkpoint_identity(self):
        args=module.parser().parse_args(["--checkpoint","/weights/model"])
        self.assertEqual(str(args.output),"/DRoboticsAILab/users/licheng.yang/openloop_eval")
        self.assertIsNone(args.episodes)
        path=module.figure_path(args.output,Path("/weights/recipe/checkpoint-10000"),"test",4)
        self.assertEqual(path.name,"recipe_checkpoint-10000_test_ep000004.png")
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            module.save_state_comparison(np.zeros((46,14)),np.ones((46,14)),root,23,filename=path.name)
            self.assertEqual([p.name for p in root.iterdir()],[path.name])

    def test_default_episodes_match_actual_train_validation_split(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        metadata=SimpleNamespace(total_episodes=50,tasks={0:"task"},
            episodes={i:dict(tasks=["task"]) for i in range(50)})
        original=module._import_module
        def imports(name):
            if name=="lerobot.datasets.lerobot_dataset":
                return SimpleNamespace(LeRobotDatasetMetadata=lambda **_:metadata)
            return original(name)
        with patch.object(module,"_import_module",side_effect=imports):
            selected=module.choose_episodes(Path("/unused"),dict(seed=42,val_ratio=.02,max_train_episodes=50))
            self.assertEqual(selected,[("train",0),("test",4)])
            with self.assertRaises(ValueError):
                module.choose_episodes(Path("/unused"),dict(seed=42,val_ratio=.02),[4,4])

    def test_no_future_observations_and_exact_future_gt(self):
        state, image, action = module.window_indices(30, 3, 1, 5, 100)
        np.testing.assert_array_equal(state, [27, 28, 29])
        np.testing.assert_array_equal(image, [29])
        np.testing.assert_array_equal(action, [30, 31, 32, 33, 34])
        self.assertLess(max(image), min(action))
        for anchor in [2, 96]:
            with self.assertRaises(ValueError):
                module.window_indices(anchor, 3, 1, 5, 100)

    def test_obs1_horizon24_has_no_state_history(self):
        state, image, action = module.window_indices(30, 1, 1, 23, 100)
        np.testing.assert_array_equal(state, [29])
        np.testing.assert_array_equal(image, [29])
        np.testing.assert_array_equal(action, np.arange(30, 53))

    def test_single_color_conversion_and_tensor_layout(self):
        raw = np.array([[[10, 20, 30]]], dtype=np.uint8)
        state = np.arange(60, dtype=np.float32).reshape(5, 12)
        images = {"head_cam": {2: raw}}
        window = module.window_indices(3, 3, 1, 2, 5)
        obs = module.build_observation(
            state, images, window, ["head_cam"], "cpu", "bgr"
        )
        np.testing.assert_array_equal(obs["state"].numpy()[0], state[:3])
        np.testing.assert_allclose(
            obs["head_cam"].numpy()[0, 0, :, 0, 0], np.array([30, 20, 10]) / 255.0
        )
        np.testing.assert_array_equal(raw, [[[10, 20, 30]]])
        np.testing.assert_array_equal(module.source_to_rgb(raw, "rgb"), raw)

    def test_grippers_do_not_pollute_joint_radians(self):
        gt = np.zeros((2, 5, 14))
        prediction = gt.copy()
        prediction[..., [6, 13]] = 1
        value = module.metrics(prediction, gt)
        self.assertEqual(value["joint_rmse_rad"], 0)
        self.assertEqual(value["gripper_mae"], 1)
        self.assertEqual(len(value["joint_rmse_by_horizon"]), 5)


if __name__ == "__main__":
    unittest.main()
