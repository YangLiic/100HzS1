"""A scene without an expert approach is rejected, not an infrastructure crash."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

class AlarmclockExpertTests(unittest.TestCase):
    def test_unreachable_approach_marks_plan_failure_before_motion(self):
        p=Path(__file__).resolve().parents[1]/"simulation/robotwin2.0/envs/click_alarmclock.py"
        tree=ast.parse(p.read_text())
        task=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        method=next(n for n in task.body if isinstance(n,ast.FunctionDef) and n.name=="play_once")
        namespace={"ArmTag":lambda x:x}
        exec(compile(ast.Module(body=[method],type_ignores=[]),str(p),"exec"),namespace)
        def unexpected(*args,**kwargs):raise AssertionError("motion was attempted")
        fake=SimpleNamespace(alarm=SimpleNamespace(get_pose=lambda:SimpleNamespace(p=[0.1,0,0])),
            get_grasp_pose=lambda *a,**k:None,move=unexpected,plan_success=True,info={"info":{}})
        self.assertIs(namespace["play_once"](fake),fake.info)
        self.assertFalse(fake.plan_success)
