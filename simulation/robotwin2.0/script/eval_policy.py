"""Fixed-seed RoboTwin evaluation with an in-process VFA policy."""

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parents[1]
sys.path[:0] = [str(ROOT / "policy"), str(ROOT), str(REPOSITORY / "src")]
os.chdir(ROOT)

import numpy as np
import torch
import yaml
from description.utils.generate_episode_instructions import (
    generate_episode_descriptions,
)


def read_seeds(path, count):
    values = [
        int(line.strip())
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]
    if count < 1 or len(values) < count or len(set(values[:count])) != count:
        raise ValueError(
            "Need the requested number of distinct newline-separated seeds"
        )
    return values[:count]


def task_arguments(task, profile, output):
    with (ROOT / "task_config" / f"{profile}.yml").open() as handle:
        args = yaml.safe_load(handle)
    with (ROOT / "task_config/_embodiment_config.yml").open() as handle:
        embodiments = yaml.safe_load(handle)
    with (ROOT / "task_config/_camera_config.yml").open() as handle:
        cameras = yaml.safe_load(handle)
    kinds = args["embodiment"]
    if len(kinds) not in (1, 3):
        raise ValueError(
            "Expected one dual-arm embodiment or two arms and their distance"
        )
    for side, kind in zip(
        ("left", "right"), (kinds[0], kinds[0] if len(kinds) == 1 else kinds[1])
    ):
        path = Path(embodiments[kind]["file_path"]).resolve()
        args[f"{side}_robot_file"] = str(path)
        with (path / "config.yml").open() as handle:
            args[f"{side}_embodiment_config"] = yaml.safe_load(handle)
        args[f"{side}_embodiment_config"]["planner"] = args.get(
            "motion_planner", "curobo"
        )
    args.update(
        task_name=task,
        task_config=profile,
        policy_name="VFA",
        eval_mode=True,
        dual_arm_embodied=len(kinds) == 1,
    )
    if len(kinds) == 3:
        args["embodiment_dis"] = kinds[2]
    camera = cameras[args["camera"]["head_camera_type"]]
    args.update(head_camera_h=camera["h"], head_camera_w=camera["w"])
    if args.get("eval_video_log"):
        args["eval_video_save_dir"] = output
    return args, f"{camera['w']}x{camera['h']}"


def runtime_record(policy_module):
    import envs
    import curobo
    import sapien
    import vfa

    modules = {
        "envs": envs,
        "policy": policy_module,
        "vfa": vfa,
        "torch": torch,
        "sapien": sapien,
        "curobo": curobo,
    }
    origins = {
        name: str(
            Path(
                getattr(module, "__file__", None) or next(iter(module.__path__))
            ).resolve()
        )
        for name, module in modules.items()
    }
    if not Path(origins["envs"]).is_relative_to(ROOT):
        raise RuntimeError(f"Wrong RoboTwin import: {origins}")
    if any(
        "DRRM-VODPP" in str(getattr(module, "__file__", ""))
        for module in sys.modules.values()
    ):
        raise RuntimeError("An external legacy checkout leaked into the runtime")
    return {
        "python": sys.executable,
        "environment": sys.prefix,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sapien": importlib.metadata.version("sapien"),
        "gpu": torch.cuda.get_device_name(0),
        "pid": os.getpid(),
        "module_paths": origins,
        "architecture": "single_process_direct_inference",
        "rgb_channel_flip": False,
    }


def check_expert(environment, seed, index, args):
    from expert_check import check_scene
    from envs.utils.create_actor import UnStableError

    info, records = check_scene(
        environment, seed, index, args, attempts=3, unstable_errors=(UnStableError,)
    )
    if records[-1]["success"]:
        return info
    raise RuntimeError(
        f"Expert scene check failed for pinned seed {seed}; refusing to replace it"
    )


def run_episode(environment, model, policy, seed, index, args, video_size, output):
    info = check_expert(environment, seed, index, args)
    environment.setup_demo(now_ep_num=index, seed=seed, is_test=True, **args)
    device = environment.scene.render_system.device
    print(
        f"SIM_DEVICE cuda_id={device.cuda_id} pci={device.pci_string} "
        f"visible={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    np.random.seed(seed)
    descriptions = generate_episode_descriptions(args["task_name"], [info["info"]], 100)
    environment.set_instruction(np.random.choice(descriptions[0]["unseen"]))
    model.set_episode_seed(seed)
    policy.reset_model(model)
    video = None
    try:
        if args.get("eval_video_log"):
            video = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    str(output / f"seed_{seed}.mp4"),
                ],
                stdin=subprocess.PIPE,
            )
            environment._set_eval_video_ffmpeg(video)
        while (
            environment.take_action_cnt < environment.step_lim
            and not environment.eval_success
        ):
            policy.eval(environment, model, environment.get_obs())
        return {
            "seed": seed,
            "success": bool(environment.eval_success),
            "actions": environment.take_action_cnt,
            "inference_calls": model.infer_index,
        }
    finally:
        if video is not None:
            environment._del_eval_video_ffmpeg()
            if video.returncode != 0:
                raise RuntimeError(
                    f"Video encoder exited {video.returncode} for seed {seed}"
                )
        environment.close_env(clear_cache=(index + 1) % args["clear_cache_freq"] == 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-list", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--task", default="blocks_ranking_rgb")
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--nfe", type=int, default=4)
    args = parser.parse_args()
    seeds = read_seeds(args.seed_list, args.episodes)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "episodes.jsonl").exists():
        raise FileExistsError("Use a fresh output directory")
    import VFA as policy

    torch.set_num_threads(1)
    model = policy.get_model({"ckpt_setting": str(args.checkpoint), "nfe": args.nfe})
    (args.output / "runtime.json").write_text(
        json.dumps(runtime_record(policy), indent=2)
    )
    kwargs, video_size = task_arguments(args.task, args.task_config, args.output)
    kwargs["ckpt_setting"] = str(args.checkpoint)
    environment = getattr(importlib.import_module(f"envs.{args.task}"), args.task)()
    records = []
    for index, seed in enumerate(seeds):
        print(f"EPISODE_STARTED index={index + 1}/{len(seeds)} seed={seed}", flush=True)
        record = run_episode(
            environment, model, policy, seed, index, kwargs, video_size, args.output
        )
        records.append(record)
        with (args.output / "episodes.jsonl").open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"EPISODE_COMPLETED {json.dumps(record)}", flush=True)
    (args.output / "_eval_log.txt").write_text(
        "".join(
            f"seed: {r['seed']}, result: {'Success' if r['success'] else 'Fail'}\n"
            for r in records
        )
    )
    (args.output / "_eval_seeds.txt").write_text("".join(f"{seed}\n" for seed in seeds))
    result = {
        "checkpoint": str(args.checkpoint),
        "task": args.task,
        "nfe": args.nfe,
        "episodes": len(records),
        "successes": sum(r["success"] for r in records),
        "results": records,
    }
    result["success_rate"] = result["successes"] / len(records)
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    (args.output / "EVALUATION_COMPLETED").touch()
    print(
        f"EVALUATION_COMPLETED successes={result['successes']}/{len(records)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
