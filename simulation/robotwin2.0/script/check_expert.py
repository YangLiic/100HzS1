"""Select the first N expert-solvable scene seeds without loading a 100HzS1 checkpoint."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "script"))
from expert_check import accepted_seeds, check_scene


def atomic_write(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2) + "\n")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="blocks_ranking_rgb")
    p.add_argument("--task-config", default="demo_clean")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--start-seed", type=int, default=100000)
    p.add_argument("--max-candidates", type=int, default=2000)
    p.add_argument("--attempts", type=int, default=3)
    p.add_argument("--output", type=Path)
    p.add_argument("--seed-root", type=Path)
    p.add_argument("--publish", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def screen(args):
    if (
        args.count < 1
        or args.start_seed < 0
        or args.max_candidates < args.count
        or args.attempts < 1
    ):
        raise ValueError("Invalid count, start-seed, max-candidates or attempts")
    settings = {
        key: getattr(args, key)
        for key in [
            "task",
            "task_config",
            "count",
            "start_seed",
            "max_candidates",
            "attempts",
        ]
    }
    if args.dry_run:
        print(json.dumps(dict(settings, output=str(args.output.resolve())), indent=2))
        return
    from eval_policy import task_arguments
    from envs.utils.create_actor import UnStableError
    import envs
    import torch
    import curobo
    import subprocess

    torch.set_num_threads(1)
    if not Path(envs.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError("Wrong RoboTwin checkout")
    if any(
        "DRRM-VODPP" in str(getattr(m, "__file__", "")) for m in sys.modules.values()
    ):
        raise RuntimeError("External legacy code leaked into expert screening")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    kwargs, _ = task_arguments(args.task, args.task_config, output)
    kwargs.update(
        render_freq=0, eval_video_log=False, eval_video_save_dir=None, save_data=False
    )
    sources = [
        Path(__file__),
        ROOT / "script/expert_check.py",
        ROOT / "script/eval_policy.py",
        ROOT / "envs/_base_task.py",
        ROOT / "envs/robot/planner.py",
        ROOT / f"envs/{args.task}.py",
        ROOT / f"task_config/{args.task_config}.yml",
        ROOT / "task_config/_camera_config.yml",
        ROOT / "task_config/_embodiment_config.yml",
    ]
    identity = {
        "settings": settings,
        "sources": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "resolved_task_config_sha256": hashlib.sha256(
            json.dumps(kwargs, sort_keys=True, default=str).encode()
        ).hexdigest(),
    }
    state_path = output / "state.json"
    if args.resume:
        state = json.loads(state_path.read_text())
        if state["identity"] != identity:
            raise ValueError("Cannot resume after changing code or scene configuration")
    else:
        if any(output.iterdir()):
            raise FileExistsError("Use a fresh output or --resume")
        curobo_sha = subprocess.check_output(
            ["git", "-C", str(ROOT / "envs/curobo"), "rev-parse", "HEAD"], text=True
        ).strip()
        state = {
            "identity": identity,
            "records": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "runtime": {
                "python": sys.executable,
                "environment": sys.prefix,
                "torch": torch.__version__,
                "sapien": importlib.metadata.version("sapien"),
                "gpu": torch.cuda.get_device_name(0),
                "curobo_commit": curobo_sha,
                "envs": str(Path(envs.__file__).resolve()),
                "curobo": str(Path(curobo.__file__).resolve()),
            },
        }
        write_json(state_path, state)
    selected = accepted_seeds(state["records"], args.start_seed)
    if len(selected) > args.count:
        raise ValueError("Too many accepted seeds in saved state")
    environment = getattr(importlib.import_module(f"envs.{args.task}"), args.task)()
    for candidate in range(len(state["records"]), args.max_candidates):
        if len(selected) == args.count:
            break
        seed = args.start_seed + candidate
        print(
            f"EXPERT_SCENE_STARTED seed={seed} accepted={len(selected)}/{args.count}",
            flush=True,
        )
        info, attempts = check_scene(
            environment, seed, len(selected), kwargs, args.attempts, (UnStableError,)
        )
        success = bool(attempts[-1]["success"])
        state["records"].append(
            {"seed": seed, "success": success, "attempts": attempts}
        )
        selected = accepted_seeds(state["records"], args.start_seed)
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)
        atomic_write(output / "seeds.partial.txt", "".join(f"{s}\n" for s in selected))
        print(
            f"EXPERT_SCENE_COMPLETED seed={seed} success={success} "
            f"accepted={len(selected)}/{args.count}",
            flush=True,
        )
    if len(selected) != args.count:
        raise RuntimeError(
            f"Only {len(selected)}/{args.count} passed; candidate limit reached"
        )
    seed_text = "".join(f"{seed}\n" for seed in selected)
    atomic_write(output / "seeds.txt", seed_text)
    summary = {
        "verified": True,
        "task": args.task,
        "task_config": args.task_config,
        "count": len(selected),
        "candidate_count": len(state["records"]),
        "rejected_seeds": [r["seed"] for r in state["records"] if not r["success"]],
        "seed_sha256": hashlib.sha256(seed_text.encode()).hexdigest(),
        "selection": "first N consecutive candidates that pass expert planning AND task success",
        "policy_checkpoint": None,
        "settings": settings,
        "runtime": state["runtime"],
        "source_sha256": identity["sources"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "summary.json", summary)
    atomic_write(output / "EXPERT_SEEDS_COMPLETED", summary["seed_sha256"] + "\n")
    print("EXPERT_SEEDS_COMPLETED " + json.dumps(summary), flush=True)
    return output / "seeds.txt", summary


def main():
    from importlib import import_module
    registry_class = import_module("100hzs1.evaluation.seeds").SeedRegistry
    args = parser().parse_args()
    if args.count != 100:
        if args.output is None:
            raise ValueError("A diagnostic count other than 100 requires --output")
        if args.publish:
            raise ValueError("Only a complete 100-seed set can be published")
        return screen(args)
    registry = registry_class(ROOT,args.seed_root)
    if args.dry_run:
        found = registry.resolve(args.task,args.task_config)
        print(json.dumps(dict(task=args.task,task_config=args.task_config,count=100,
            action="reuse" if found else "select_and_lock",
            seed_list=str(found[0]) if found else None,
            registry=str(registry.directory(args.task,args.task_config))),indent=2))
        return
    def select(build):
        from copy import copy
        options = copy(args)
        options.output = args.output or build
        if (options.output/"EXPERT_SEEDS_COMPLETED").exists():
            return options.output/"seeds.txt", json.loads((options.output/"summary.json").read_text())
        options.resume = (options.output/"state.json").exists()
        return screen(options)
    path, info = registry.ensure(args.task,args.task_config,select)
    if args.output is not None and args.output.resolve() != path.parent:
        args.output.mkdir(parents=True,exist_ok=True)
        for name, data in [("seeds.txt",path.read_bytes()),("summary.json",(json.dumps(info,indent=2)+"\n").encode())]:
            target = args.output/name
            if target.exists() and name=="seeds.txt" and target.read_bytes()!=data:
                raise FileExistsError("Output contains a different seed set")
            if not target.exists():
                target.write_bytes(data)
        (args.output/"EXPERT_SEEDS_COMPLETED").write_text(info["seed_sha256"]+"\n")
    if args.publish:
        registry.export(args.task,args.task_config)
    print("EXPERT_SEEDS_READY "+json.dumps(dict(task=args.task,task_config=args.task_config,
        count=100,seed_list=str(path),seed_sha256=info["seed_sha256"])),flush=True)


if __name__ == "__main__":
    main()
