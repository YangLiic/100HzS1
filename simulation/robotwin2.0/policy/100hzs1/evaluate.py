from importlib import import_module as _import_module
"""Run one direct RoboTwin/100HzS1 process per GPU and merge fixed seed results."""

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("/DRoboticsAILab/users/licheng.yang/robotwin2.0/100HzS1")


load_inference_options = _import_module("100hzs1.common.inference").checkpoint_inference_options
records = _import_module("100hzs1.evaluation.records")

def split_seeds(path, total, workers):
    seeds = [
        int(line.strip())
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if total < workers or len(seeds) < total or len(set(seeds[:total])) != total:
        raise ValueError("Need enough distinct pinned seeds and at least one per GPU")
    size, extra = divmod(total, workers)
    groups = []
    start = 0
    for worker in range(workers):
        count = size + (worker < extra)
        groups.append(seeds[start : start + count])
        start += count
    return groups


def seed_provenance(path, task, task_config, require_verified=False):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path = path.with_suffix(".expert.json")
    if not metadata_path.exists() and path.name=="seeds.txt":
        metadata_path = path.with_name("summary.json")
    if not metadata_path.exists():
        if require_verified:
            raise FileNotFoundError(
                f"Missing expert verification: {metadata_path}; run scripts/eval/check_expert.sh first"
            )
        return {"verified": False, "seed_sha256": digest, "seed_list": str(path)}
    metadata = json.loads(metadata_path.read_text())
    values = [int(line) for line in path.read_text().splitlines() if line.strip()]
    if (
        metadata.get("verified") is not True
        or metadata.get("task") != task
        or metadata.get("task_config") != task_config
        or metadata.get("seed_sha256") != digest
        or metadata.get("count") != len(values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("Expert seed metadata does not match task/profile/content")
    return {
        "verified": True,
        "seed_sha256": digest,
        "seed_list": str(path),
        "metadata": str(metadata_path),
        "count": len(values),
        "scene_sha256": metadata.get("scene_identity", {}).get("sha256"),
        "seed_set_scope": metadata.get("seed_set_scope", "verified_task_profile_list"),
    }


def merge_results(output, groups, gpus):
    return records.collect(output, groups, gpus, require_complete=True)[0]


def terminate(process):
    if process is not None and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--seed-list",
        type=Path,
        help="Override the task/profile's expert-verified default seed list",
    )
    parser.add_argument("--seed-root", type=Path)
    parser.add_argument("--task", default="beat_block_hammer")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--nfe", type=int, default=None)
    parser.add_argument("--inference-config", type=Path)
    parser.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--kv-cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default=None)
    parser.add_argument("--backend", choices=["eager", "tensorrt"], default=None)
    parser.add_argument("--tensorrt-engine-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-worker-restarts", type=int, default=3)
    args = parser.parse_args()
    config_path = args.checkpoint / "config.json"
    flow_type = json.loads(config_path.read_text()).get("flow_type", "fm") if config_path.is_file() else "fm"
    options = load_inference_options(args.checkpoint, args.inference_config, {
        "num_inference_timesteps": args.nfe, "use_cuda_graph": args.cuda_graph,
        "use_kv_cache": args.kv_cache, "precision": args.precision,
        "backend": args.backend, "tensorrt_engine_dir": args.tensorrt_engine_dir,
    })
    args.nfe = options["num_inference_timesteps"]
    args.cuda_graph = options["use_cuda_graph"]
    args.kv_cache = options["use_kv_cache"]
    args.precision = options["precision"]
    args.backend = options["backend"]
    args.tensorrt_engine_dir = options["tensorrt_engine_dir"]
    if (
        args.nfe < 1 or args.max_worker_restarts < 0
        or len(set(args.gpus)) != len(args.gpus)
        or any(g < 0 for g in args.gpus)
    ):
        raise ValueError("Invalid nfe or GPU IDs")
    args.checkpoint = args.checkpoint.resolve()
    if not args.dry_run and not (args.checkpoint/"model.safetensors").is_file():
        raise FileNotFoundError(args.checkpoint)
    explicit_seed_list = args.seed_list is not None
    if args.seed_list is None:
        registry = _import_module("100hzs1.evaluation.seeds").SeedRegistry(ROOT,args.seed_root)
        found = registry.resolve(args.task,args.task_config)
        if args.dry_run and found is None:
            print(json.dumps(dict(task=args.task,task_config=args.task_config,
                seed_status="needs_expert_screening",seed_set_size=100,
                seed_registry=str(registry.directory(args.task,args.task_config))),indent=2))
            return
        if not args.dry_run:
            # The preparer locks by task/profile and imports existing verified lists without GPU work.
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpus[0]),GPU_ID=str(args.gpus[0]),
                       KRAKEN_EGL_DEVICE=str(args.gpus[0]))
            command = [sys.executable,str(ROOT/"script/check_expert.py"),"--task",args.task,
                       "--task-config",args.task_config,"--seed-root",str(registry.root)]
            subprocess.run(command,cwd=ROOT,env=env,check=True)
            found = registry.get(args.task,args.task_config)
        args.seed_list = found[0]
    args.seed_list = args.seed_list.resolve()
    seed_info = seed_provenance(
        args.seed_list,
        args.task,
        args.task_config,
        require_verified=not explicit_seed_list,
    )
    args.checkpoint = args.checkpoint.resolve()
    if args.output is None:
        args.output = (
            DEFAULT_OUTPUT
            / args.task
            / args.checkpoint.name
            / datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S")
        )
    args.output = args.output.resolve()
    groups = split_seeds(args.seed_list, args.episodes, len(args.gpus))
    manifest = {
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "task_config": args.task_config,
        "planned_episodes": args.episodes,
        "gpus": args.gpus,
        "seed_groups": groups,
        "expert_seed_provenance": seed_info,
        "use_cuda_graph": args.cuda_graph,
        "use_kv_cache": args.kv_cache,
        "precision": args.precision,
        "backend": args.backend,
        "cuda_graph_scope": "full_pipeline" if args.backend == "tensorrt" else "visual_backbone",
        "nfe": args.nfe,
        "flow_type": flow_type,
        "tau": [(1 - i / args.nfe) if flow_type == "meanflow" else i / args.nfe
                for i in range(args.nfe + 1)],
        "runtime": "single_process_direct_inference",
        "python": sys.executable,
        "channels": "RGB_no_flip",
        "jpeg_roundtrip": True,
        "output": str(args.output),
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    if not (args.checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(args.checkpoint)
    run_workers(args, manifest, groups)


def run_workers(args, manifest, groups, worker_entry=None, interval=1):
    """Only this controller writes result.json; each worker owns one journal."""
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    (args.output / "logs").mkdir()
    (args.output / "videos").mkdir()
    sources = [Path(__file__), ROOT / "script/eval_policy.py",
               ROOT / "policy/100hzs1/deploy_policy.py", Path(records.__file__)]
    manifest["source_sha256"] = {
        str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else "evaluation/records.py":
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest["started_at"] = records.now()
    manifest["checkpoint_config_sha256"] = hashlib.sha256(
        (args.checkpoint / "config.json").read_bytes()).hexdigest()
    manifest = dict(manifest)
    manifest.update(checkpoint=str(args.checkpoint.resolve()), task=args.task,
        task_config=args.task_config, seed_groups=groups, gpus=args.gpus, nfe=args.nfe,
        use_cuda_graph=args.cuda_graph, use_kv_cache=args.kv_cache, precision=args.precision,
        backend=getattr(args, "backend", "tensorrt"),
        channels="RGB_no_flip", max_worker_restarts=getattr(args, "max_worker_restarts", 3))
    processes, handles, commands, environments = [], [], [], []
    worker_entry = worker_entry or ROOT / "script/eval_policy.py"
    restarts = [0] * len(args.gpus)

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    def launch(index):
        return subprocess.Popen(commands[index], cwd=ROOT, env=environments[index],
            stdout=handles[index], stderr=subprocess.STDOUT, start_new_session=True)

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGINT)}
    records.snapshot(manifest, args.output, groups, args.gpus)
    last_completed = -1
    try:
        source = getattr(args, "resume_from", None)
        if source:
            records.restore_progress(source, args.output, manifest, groups, args.gpus)
            records.snapshot(manifest, args.output, groups, args.gpus)
        saved, _ = records.collect(args.output, groups, args.gpus)
        for index, (gpu, seeds) in enumerate(zip(args.gpus, groups)):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), GPU_ID=str(gpu),
                       KRAKEN_EGL_DEVICE=str(gpu), PYTHONUNBUFFERED="1", S1_JPEG_ROUNDTRIP="1")
            environments.append(env)
            handle = (args.output / "logs" / f"gpu{gpu}.log").open("a")
            handles.append(handle)
            command = [sys.executable, "-u", str(worker_entry),
                       "--checkpoint", str(args.checkpoint), "--output", str(args.output),
                       "--journal", str(args.output / "logs" / f"gpu{gpu}.jsonl"),
                       "--seeds", *map(str, seeds), "--task", args.task,
                       "--task-config", args.task_config, "--nfe", str(args.nfe),
                       "--cuda-graph" if args.cuda_graph else "--no-cuda-graph",
                       "--kv-cache" if args.kv_cache else "--no-kv-cache",
                       "--precision", args.precision,
                       "--backend", getattr(args, "backend", "tensorrt")]
            if getattr(args, "tensorrt_engine_dir", None):
                command += ["--tensorrt-engine-dir", str(args.tensorrt_engine_dir)]
            if source:
                command.append("--resume")
            if args.inference_config:
                command += ["--inference-config", str(args.inference_config.resolve())]
            commands.append(command)
            if sum(row["gpu"] == gpu for row in saved) == len(seeds):
                processes.append(None)
            else:
                processes.append(launch(index))
                print(f"WORKER_STARTED gpu={gpu} episodes={len(seeds)}", flush=True)
        while True:
            current = records.snapshot(manifest, args.output, groups, args.gpus)
            if current["completed_episodes"] != last_completed:
                rate = current["success_rate"]
                rate_text = f"{rate:.1%}" if rate is not None else "pending"
                print(f"PROGRESS {current['completed_episodes']}/{current['planned_episodes']} "
                      f"successes={current['successes']} success_rate={rate_text} "
                      "(completed episodes)", flush=True)
                last_completed = current["completed_episodes"]
            codes = [p.poll() if p is not None else 0 for p in processes]
            for index, code in enumerate(codes):
                if code in (None, 0):
                    continue
                gpu = args.gpus[index]
                journal = args.output / "logs" / f"gpu{gpu}.jsonl"
                events = records.read_events(journal)
                retryable = events and events[-1]["event"] == "failed" and events[-1].get("retryable_scene") is True
                if not retryable or restarts[index] >= manifest["max_worker_restarts"]:
                    raise RuntimeError(f"GPU{gpu} worker exited {code}; inspect logs/gpu{gpu}.log")
                restarts[index] += 1
                records.append_event(journal, "retry", attempt=restarts[index], error=events[-1]["error"])
                if "--resume" not in commands[index]:
                    commands[index].append("--resume")
                processes[index] = launch(index)
                codes[index] = None
                print(f"WORKER_SCENE_RETRY gpu={gpu} restart={restarts[index]} same_seeds=True", flush=True)
            if all(code == 0 for code in codes):
                break
            time.sleep(interval)
        result = records.snapshot(manifest, args.output, groups, args.gpus, status="completed", final=True)
        print(f"EVALUATION_COMPLETED successes={result['successes']}/"
              f"{result['completed_episodes']} = {result['success_rate']:.2%}", flush=True)
        return result
    except BaseException as exc:
        for process in processes:
            terminate(process)
        try:
            records.snapshot(manifest, args.output, groups, args.gpus,
                status="interrupted" if isinstance(exc, (SystemExit, KeyboardInterrupt)) else "failed",
                error=exc)
        except Exception:
            current = json.loads((args.output / "result.json").read_text())
            current.update(status="failed", error=str(exc), updated_at=records.now())
            records.atomic_json(args.output / "result.json", current)
        raise
    finally:
        for process in processes:
            terminate(process)
        for handle in handles:
            handle.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
