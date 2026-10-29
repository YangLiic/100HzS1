"""Exercise real multi-process polling, partial SR, failure and atomic summary updates."""
from importlib import import_module, util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = util.spec_from_file_location("live_eval_controller", ROOT/"simulation/robotwin2.0/policy/100hzs1/evaluate.py")
controller = util.module_from_spec(spec)
spec.loader.exec_module(controller)
records = import_module("100hzs1.evaluation.records")

FAKE = """
import argparse,os,time
from pathlib import Path
from importlib import import_module
p=argparse.ArgumentParser()
p.add_argument("--journal",type=Path)
p.add_argument("--output",type=Path)
p.add_argument("--seeds",type=int,nargs="+")
p.add_argument("--task")
a,_=p.parse_known_args()
r=import_module("100hzs1.evaluation.records")
done=[e["record"]["seed"] for e in r.read_events(a.journal) if e["event"]=="episode"]
r.append_event(a.journal,"runtime",runtime={"fake":True})
for seed in a.seeds:
    if seed in done:continue
    r.append_event(a.journal,"started",seed=seed)
    time.sleep(.015*(int(os.environ["GPU_ID"])+1))
    name=f"videos/seed_{seed}.mp4"
    (a.output/name).write_bytes(b"fixture")
    r.append_event(a.journal,"episode",record=dict(seed=seed,success=seed%2==0,video=name))
    if a.task=="failure":raise SystemExit(2)
    marker=a.output.parent/("gpu"+os.environ["GPU_ID"]+"_retried")
    if a.task=="retry_scene" and not marker.exists():
        marker.touch()
        r.append_event(a.journal,"failed",error="Expert scene check failed for pinned seed",
                       retryable_scene=True)
        raise SystemExit(2)
r.append_event(a.journal,"completed",graph={"graph_captures":1,"graph_replays":len(a.seeds)})
"""


class LiveResultsTest(unittest.TestCase):
    def arguments(self, root, task="fixture"):
        checkpoint=root/"weight";checkpoint.mkdir()
        (checkpoint/"config.json").write_text("{}")
        worker=root/"fake_worker.py";worker.write_text(FAKE)
        return SimpleNamespace(output=root/"evaluation", checkpoint=checkpoint,
            gpus=[0,1,2,3],task=task,task_config="demo_clean",nfe=10,
            cuda_graph=True,kv_cache=True,precision="bf16",inference_config=None),worker

    def test_four_workers_publish_partial_success_before_full_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,worker=self.arguments(Path(tmp))
            groups=[list(range(i*25,(i+1)*25)) for i in range(4)]
            observed=[];stop=threading.Event()
            def observe():
                while not stop.is_set():
                    path=args.output/"result.json"
                    if path.exists():
                        # os.replace must never expose half-written JSON.
                        observed.append(json.loads(path.read_text()))
                    time.sleep(.005)
            thread=threading.Thread(target=observe);thread.start()
            try:
                final=controller.run_workers(args,dict(task=args.task),groups,worker_entry=worker,interval=.03)
            finally:
                stop.set();thread.join()
            partial=[d for d in observed if 0<d["completed_episodes"]<100 and d["status"]=="running"]
            self.assertTrue(partial)
            for d in partial:
                self.assertEqual(d["completed_episodes"],len(d["results"]))
                self.assertEqual(d["success_rate"],d["successes"]/d["completed_episodes"])
            self.assertEqual((final["status"],final["completed_episodes"],final["successes"]),("completed",100,50))
            self.assertEqual({p.name for p in args.output.iterdir()},{"result.json","logs","videos"})
            self.assertEqual(len(list((args.output/"videos").iterdir())),100)

    def test_worker_crash_keeps_partial_results_and_marks_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,worker=self.arguments(Path(tmp),"failure")
            groups=[list(range(i*25,(i+1)*25)) for i in range(4)]
            with self.assertRaises(RuntimeError):
                controller.run_workers(args,dict(task=args.task),groups,worker_entry=worker,interval=.03)
            value=json.loads((args.output/"result.json").read_text())
            self.assertEqual(value["status"],"failed")
            self.assertLess(value["completed_episodes"],100)
            self.assertIn("error",value)

    def test_same_seed_restart_keeps_completed_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            args,worker=self.arguments(Path(tmp),"retry_scene")
            groups=[list(range(i*3,(i+1)*3)) for i in range(4)]
            value=controller.run_workers(args,dict(task=args.task),groups,worker_entry=worker,interval=.03)
            self.assertEqual(value["status"],"completed")
            self.assertEqual(value["completed_episodes"],12)
            self.assertEqual(len({r["seed"] for r in value["results"]}),12)
            self.assertTrue(all(w.get("restarts")==1 for w in value["workers"].values()))

    def test_resume_preserves_old_outcomes_and_rejects_changed_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args,worker=self.arguments(root)
            groups=[list(range(i*3,(i+1)*3)) for i in range(4)]
            worker.write_text(FAKE.replace('if a.task=="failure":','if True:'))
            with self.assertRaises(RuntimeError):
                controller.run_workers(args,dict(task=args.task),groups,worker_entry=worker,interval=.03)
            source=args.output
            old=json.loads((source/"result.json").read_text())
            self.assertGreater(old["completed_episodes"],0)
            args.resume_from=source
            args.output=root/"invalid_resume";args.nfe=4
            with self.assertRaises(ValueError):
                controller.run_workers(args,dict(task=args.task),groups,worker_entry=worker,interval=.03)
            args.output=root/"resumed";args.nfe=10
            worker.write_text(FAKE)
            final=controller.run_workers(args,dict(task=args.task),groups,worker_entry=worker,interval=.03)
            self.assertEqual(final["status"],"completed")
            self.assertEqual(final["completed_episodes"],12)
            self.assertEqual(final["preserved_episodes"],old["completed_episodes"])
            by_seed={row["seed"]:row for row in final["results"]}
            for row in old["results"]:
                self.assertEqual(by_seed[row["seed"]],row)

    def test_duplicate_wrong_gpu_and_partial_last_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"logs").mkdir()
            journal=root/"logs/gpu0.jsonl"
            records.append_event(journal,"episode",record=dict(seed=10,success=True))
            with journal.open("a") as h:h.write('{"unfinished":')
            value=records.snapshot({},root,[[10,11]],[0])
            self.assertEqual((value["completed_episodes"],value["success_rate"]),(1,1))
            with self.assertRaises(RuntimeError):
                records.collect(root,[[10,11]],[0],require_complete=True)
            journal.write_text("")
            records.append_event(journal,"episode",record=dict(seed=20,success=True))
            with self.assertRaises(ValueError):
                records.collect(root,[[10,11]],[0])
            journal.write_text("")
            for _ in range(2):
                records.append_event(journal,"episode",record=dict(seed=10,success=True))
            with self.assertRaises(ValueError):
                records.collect(root,[[10,11]],[0])


if __name__=="__main__":
    unittest.main()
