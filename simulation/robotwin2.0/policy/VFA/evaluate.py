"""Run one direct RoboTwin/VFA process per GPU and merge fixed seed results."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path("/DRoboticsAILab/users/licheng.yang/robotwin2.0/VFA")


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
    }


def merge_results(output, groups, gpus):
    records = []
    for gpu, expected in zip(gpus, groups):
        part = output / f"gpu{gpu}"
        if not (part / "EVALUATION_COMPLETED").is_file():
            raise RuntimeError(f"GPU{gpu} did not complete")
        result = json.loads((part / "result.json").read_text())
        receipt = [int(x) for x in (part / "_eval_seeds.txt").read_text().splitlines()]
        rows = result["results"]
        if receipt != expected or [r["seed"] for r in rows] != expected:
            raise RuntimeError(
                f"GPU{gpu} seed coverage changed; refusing partial success rate"
            )
        if any(type(r["success"]) is not bool for r in rows):
            raise ValueError("Every episode needs a boolean result")
        records.extend(dict(row, gpu=gpu) for row in rows)
    if len({r["seed"] for r in records}) != len(records):
        raise ValueError("Overlapping seeds")
    return sorted(records, key=lambda row: row["seed"])


def terminate(process):
    if process.poll() is None:
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
    parser.add_argument("--task", default="blocks_ranking_rgb")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--nfe", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        args.nfe < 1
        or len(set(args.gpus)) != len(args.gpus)
        or any(g < 0 for g in args.gpus)
    ):
        raise ValueError("Invalid nfe or GPU IDs")
    explicit_seed_list = args.seed_list is not None
    if args.seed_list is None:
        args.seed_list = (
            ROOT / "task_config/seedlists" / f"{args.task}_{args.task_config}.txt"
        )
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
            / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
    args.output = args.output.resolve()
    groups = split_seeds(args.seed_list, args.episodes, len(args.gpus))
    manifest = {
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "task_config": args.task_config,
        "episodes": args.episodes,
        "gpus": args.gpus,
        "seed_groups": groups,
        "expert_seed_provenance": seed_info,
        "nfe": args.nfe,
        "tau": [i / args.nfe for i in range(args.nfe + 1)],
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
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.json").exists():
        raise FileExistsError("Use a fresh output directory")
    sources = [
        Path(__file__),
        ROOT / "script/eval_policy.py",
        ROOT / "policy/VFA/deploy_policy.py",
    ]
    manifest["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    workers = []
    handles = []

    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    try:
        for gpu, seeds in zip(args.gpus, groups):
            part = args.output / f"gpu{gpu}"
            part.mkdir()
            seed_file = part / "seeds.txt"
            seed_file.write_text("".join(f"{s}\n" for s in seeds))
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=str(gpu),
                GPU_ID=str(gpu),
                KRAKEN_EGL_DEVICE=str(gpu),
                PYTHONUNBUFFERED="1",
                VFA_JPEG_ROUNDTRIP="1",
            )
            handle = (part / "worker.log").open("w")
            handles.append(handle)
            cmd = [
                sys.executable,
                "-u",
                str(ROOT / "script/eval_policy.py"),
                "--checkpoint",
                str(args.checkpoint),
                "--output",
                str(part),
                "--seed-list",
                str(seed_file),
                "--episodes",
                str(len(seeds)),
                "--task",
                args.task,
                "--task-config",
                args.task_config,
                "--nfe",
                str(args.nfe),
            ]
            process = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            workers.append(process)
            print(
                f"WORKER_STARTED gpu={gpu} episodes={len(seeds)} pid={process.pid}",
                flush=True,
            )
        while any(p.poll() is None for p in workers):
            if any(p.poll() not in (None, 0) for p in workers):
                raise RuntimeError(
                    "Evaluation worker failed; inspect per-GPU worker.log"
                )
            time.sleep(1)
        if any(p.returncode != 0 for p in workers):
            raise RuntimeError("Evaluation worker failed")
        records = merge_results(args.output, groups, args.gpus)
        result = dict(
            manifest, results=records, successes=sum(r["success"] for r in records)
        )
        result["success_rate"] = result["successes"] / len(records)
        (args.output / "result.json").write_text(json.dumps(result, indent=2))
        (args.output / "_result.txt").write_text(
            f"Success rate: {result['successes']}/{len(records)} = {result['success_rate']:.2%}\n"
        )
        (args.output / "EVALUATION_COMPLETED").touch()
        print(
            "EVALUATION_COMPLETED " + (args.output / "_result.txt").read_text().strip(),
            flush=True,
        )
    finally:
        for process in workers:
            terminate(process)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
