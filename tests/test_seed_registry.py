"""Task/profile seed sets must be immutable, shared and prepared only once."""
from importlib import import_module
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

seeds=import_module("100hzs1.evaluation.seeds")


def make_robotwin(root):
    files={"envs/new_task.py":"task source", "envs/_base_task.py":"base source",
           "envs/_GLOBAL_CONFIGS.py":"globals", "script/expert_check.py":"expert acceptance",
           "task_config/demo_clean.yml":"embodiment: [robot]",
           "task_config/_camera_config.yml":"camera: fixed",
           "task_config/_embodiment_config.yml":"robot: {file_path: assets/robot}",
           "assets/robot/config.yml":"robot: fixed","envs/utils/common.py":"geometry"}
    for name,value in files.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(value)


def selected(build, start=100000):
    build.mkdir(parents=True,exist_ok=True)
    text="".join(str(seed)+"\n" for seed in range(start,start+100))
    path=build/"seeds.txt";path.write_text(text)
    info=dict(verified=True,count=100,task="new_task",task_config="demo_clean",
              seed_sha256=hashlib.sha256(text.encode()).hexdigest())
    (build/"summary.json").write_text(json.dumps(info))
    return path,info


class SeedRegistryTest(unittest.TestCase):
    def setup_registry(self, root):
        robotwin=root/"robotwin";make_robotwin(robotwin)
        return seeds.SeedRegistry(robotwin,root/"registry")

    def test_first_selection_then_reuse_and_no_silent_reselection(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry=self.setup_registry(Path(tmp));calls=[]
            def select(build):
                calls.append(build);return selected(build)
            path,info=registry.ensure("new_task","demo_clean",select)
            before=path.read_bytes()
            def forbidden(_):raise AssertionError("A backbone change must not screen again")
            path2,info2=registry.ensure("new_task","demo_clean",forbidden)
            self.assertEqual((path,info["seed_sha256"]),(path2,info2["seed_sha256"]))
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(len(calls),1)
            (registry.robotwin/"task_config/_camera_config.yml").write_text("camera: changed")
            with self.assertRaisesRegex(ValueError,"changed"):
                registry.ensure("new_task","demo_clean",forbidden)

    def test_adopt_verified_bundled_list_without_expert_or_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registry=self.setup_registry(root)
            raw,info=selected(root/"old_screen")
            target=registry.robotwin/"task_config/seedlists/new_task_demo_clean.txt"
            target.parent.mkdir(parents=True);target.write_bytes(raw.read_bytes())
            target.with_suffix(".expert.json").write_text(json.dumps(info))
            def forbidden(_):raise AssertionError("Existing verified list must be reused")
            path,meta=registry.ensure("new_task","demo_clean",forbidden)
            self.assertEqual(meta["seed_sha256"],info["seed_sha256"])
            self.assertEqual(path.read_bytes(),raw.read_bytes())
            self.assertEqual(registry.export("new_task","demo_clean"),target)

    def test_incomplete_or_changed_sets_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registry=self.setup_registry(root)
            def incomplete(build):
                path,info=selected(build);path.write_text("100000\n")
                return path,info
            with self.assertRaises(ValueError):
                registry.ensure("new_task","demo_clean",incomplete)
            self.assertIsNone(registry.resolve("new_task","demo_clean"))
            path,_=registry.ensure("new_task","demo_clean",selected)
            path.write_text(path.read_text().replace("100001","100000"))
            with self.assertRaises(ValueError):
                registry.get("new_task","demo_clean")
            with self.assertRaises(ValueError):
                registry.directory("../other","demo_clean")

    def test_completed_screening_is_recovered_without_reselection(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self.setup_registry(Path(tmp))
            build = registry.directory("new_task", "demo_clean") / "build"
            path, info = selected(build)
            (build / "EXPERT_SEEDS_COMPLETED").write_text(info["seed_sha256"])
            def forbidden(_):
                raise AssertionError("Completed expert work must not be repeated")
            locked, metadata = registry.ensure("new_task", "demo_clean", forbidden)
            self.assertEqual(locked.read_bytes(), path.read_bytes())
            self.assertEqual(metadata["seed_sha256"], info["seed_sha256"])
            commit = locked.with_name("COMMITTED.json")
            self.assertTrue(commit.is_file())
            locked.with_name("summary.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "metadata was modified"):
                registry.get("new_task", "demo_clean")

    def test_uncommitted_directory_is_not_a_published_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = self.setup_registry(Path(tmp))
            directory = registry.directory("new_task", "demo_clean") / "set"
            selected(directory)
            self.assertIsNone(registry.get("new_task", "demo_clean"))
            calls = []
            def select(build):
                calls.append(build)
                return selected(build)
            registry.ensure("new_task", "demo_clean", select)
            self.assertEqual(len(calls), 1)

    def test_exported_identity_is_checked_in_a_new_shared_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = self.setup_registry(root)
            registry.ensure("new_task", "demo_clean", selected)
            registry.export("new_task", "demo_clean")
            other = seeds.SeedRegistry(registry.robotwin, root / "other_registry")
            (registry.robotwin / "envs/utils/common.py").write_text("different geometry")
            with self.assertRaisesRegex(ValueError, "definitions changed"):
                other.resolve("new_task", "demo_clean")

    def test_four_concurrent_processes_select_once_and_share_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);registry=self.setup_registry(root)
            program="""
from importlib import import_module
from pathlib import Path
import hashlib,json,sys,time
robotwin,cache,counter=map(Path,sys.argv[1:])
registry=import_module("100hzs1.evaluation.seeds").SeedRegistry(robotwin,cache)
def select(build):
    with counter.open("a") as h:h.write("selected"+chr(10))
    time.sleep(.2)
    build.mkdir(parents=True,exist_ok=True)
    text="".join(str(s)+chr(10) for s in range(100000,100100))
    path=build/"seeds.txt";path.write_text(text)
    info=dict(verified=True,count=100,task="new_task",task_config="demo_clean",
              seed_sha256=hashlib.sha256(text.encode()).hexdigest())
    (build/"summary.json").write_text(json.dumps(info))
    return path,info
path,info=registry.ensure("new_task","demo_clean",select,timeout=10)
print(info["seed_sha256"])
"""
            command=[sys.executable,"-c",program,str(registry.robotwin),str(registry.root),str(root/"calls.txt")]
            processes=[subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(4)]
            digests=[]
            for process in processes:
                stdout,stderr=process.communicate(timeout=20)
                self.assertEqual(process.returncode,0,stderr)
                digests.append(stdout.strip())
            self.assertEqual(len(set(digests)),1)
            self.assertEqual((root/"calls.txt").read_text().splitlines(),["selected"])


if __name__=="__main__":
    unittest.main()
