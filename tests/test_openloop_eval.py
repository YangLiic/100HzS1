"""Check the offline/online boundary: image color, temporal alignment, metric units."""

import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location(
    "vfa_openloop", Path(__file__).parents[1] / "scripts/openloop_eval/evaluate.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class OpenLoopTest(unittest.TestCase):
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
