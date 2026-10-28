"""Expert validity must require both flags and must not hide infrastructure errors."""

import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).parents[1] / "simulation/robotwin2.0/script/expert_check.py"
spec = importlib.util.spec_from_file_location("vfa_expert_check", path)
expert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expert)


class Environment:
    def __init__(self, results):
        self.results = iter(results)
        self.closed = 0

    def setup_demo(self, **kwargs):
        self.seed = kwargs["seed"]

    def play_once(self):
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        self.plan_success, self.task_success = result
        return {"info": {"seed": self.seed}}

    def check_success(self):
        return self.task_success

    def close_env(self, clear_cache):
        assert clear_cache
        self.closed += 1


class ExpertTests(unittest.TestCase):
    def test_both_conditions_and_cleanup_before_accepting(self):
        env = Environment([(False, True), (True, False), (True, True)])
        info, attempts = expert.check_scene(env, 100000, 0, {}, attempts=3)
        self.assertEqual(info["info"]["seed"], 100000)
        self.assertEqual([r["success"] for r in attempts], [False, False, True])
        self.assertEqual(env.closed, 3)

    def test_unstable_scene_rejected_but_infrastructure_error_propagates(self):
        class Unstable(Exception):
            pass

        env = Environment([Unstable("scene"), (False, False)])
        info, records = expert.check_scene(env, 1, 0, {}, 2, (Unstable,))
        self.assertIsNone(info)
        self.assertFalse(records[-1]["success"])
        self.assertEqual(env.closed, 2)
        env = Environment([RuntimeError("CUDA error")])
        with self.assertRaisesRegex(RuntimeError, "CUDA error"):
            expert.check_scene(env, 1, 0, {}, 3, (Unstable,))
        self.assertEqual(env.closed, 1)

    def test_resume_selection_rejects_gaps_and_preserves_order(self):
        rows = [
            {"seed": 10, "success": True},
            {"seed": 11, "success": False},
            {"seed": 12, "success": True},
        ]
        self.assertEqual(expert.accepted_seeds(rows, 10), [10, 12])
        rows[-1]["seed"] = 13
        with self.assertRaises(ValueError):
            expert.accepted_seeds(rows, 10)


if __name__ == "__main__":
    unittest.main()
