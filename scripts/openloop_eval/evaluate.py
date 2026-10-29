"""Direct 100HzS1 open-loop inference on joint14 LeRobot videos (no simulator/server)."""
from importlib import import_module as _import_module

import argparse
import contextlib
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_model

load_policy = _import_module('100hzs1.models.base_runner').load_policy
load_inference_options = _import_module("100hzs1.common.inference").load_inference_options
configure_policy_inference = _import_module("100hzs1.common.inference").configure_policy_inference

CAMERAS = {
    "left_cam": "observation.images.left",
    "right_cam": "observation.images.right",
    "head_cam": "observation.images.head",
}
JOINTS = [*range(6), *range(7, 13)]
GRIPPERS = [6, 13]


def window_indices(anchor, n_obs, n_image, chunk, length):
    """Anchor is the FIRST future action, not the latest observation."""
    if n_image > n_obs or anchor < n_obs or anchor + chunk > length:
        raise ValueError("Window crosses an episode boundary or lacks image history")
    return (
        np.arange(anchor - n_obs, anchor),
        np.arange(anchor - n_image, anchor),
        np.arange(anchor, anchor + chunk),
    )


def source_to_rgb(image, source_color="bgr"):
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"Expected HWC uint8 color image, got {image.shape}/{image.dtype}"
        )
    return np.ascontiguousarray(image[..., ::-1] if source_color == "bgr" else image)


def decode_frames(path, indices):
    wanted = set(map(int, indices))
    frames = {}
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                frames[index] = frame.to_ndarray(format="rgb24")
            if index >= max(wanted):
                break
    if frames.keys() != wanted:
        raise ValueError(
            f"Missing video frames: {path}, {sorted(wanted - frames.keys())}"
        )
    return frames


def load_episode(root, info, episode, rgb_keys, anchors, n_obs, n_image, chunk):
    location = dict(episode_index=episode, episode_chunk=episode // info["chunks_size"])
    table = pq.read_table(root / info["data_path"].format(**location))
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    action = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if state.shape != action.shape or state.shape[1:] != (14,):
        raise ValueError("This evaluator requires absolute joint14 state/action")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("Non-finite dataset values")
    if table["frame_index"].to_pylist() != list(range(len(state))):
        raise ValueError("Episode rows must be contiguous local frame indices")
    if set(table["episode_index"].to_pylist()) != {episode}:
        raise ValueError("Parquet episode_index does not match requested episode")
    windows = [window_indices(a, n_obs, n_image, chunk, len(state)) for a in anchors]
    wanted = np.unique(np.concatenate([w[1] for w in windows]))
    images = {
        key: decode_frames(
            root / info["video_path"].format(**location, video_key=CAMERAS[key]), wanted
        )
        for key in rgb_keys
    }
    return state, action, images, windows


def build_observation(
    state,
    images,
    indices,
    keys,
    device,
    source_color,
    variant="correct",
    alternate_indices=None,
):
    state_indices, image_indices, _ = indices
    obs = {
        "state": torch.from_numpy(state[state_indices].copy()).unsqueeze(0).to(device)
    }
    for key in keys:
        chosen = alternate_indices if variant == "shuffle_images" else image_indices
        colors = "rgb" if variant == "raw_color" else source_color
        frames = [source_to_rgb(images[key][int(i)], colors) for i in chosen]
        if variant == "black_images":
            frames = [np.zeros_like(frame) for frame in frames]
        if variant == "jpeg":
            import cv2

            # Match the existing direct RoboTwin adapter byte-for-byte.
            frames = [
                cv2.imdecode(cv2.imencode(".jpg", frame)[1], cv2.IMREAD_COLOR)
                for frame in frames
            ]
        value = np.moveaxis(np.stack(frames).astype(np.float32) / 255.0, -1, 1)
        obs[key] = torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0).to(device)
    return obs


def predict(model, obs, seed, precision, use_cuda_graph=None, use_kv_cache=None):
    torch.manual_seed(seed)
    cast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if precision == "bf16"
        else contextlib.nullcontext()
    )
    calls = []
    hook = model.model.register_forward_hook(lambda *_: calls.append(1))
    try:
        with torch.inference_mode(), cast:
            action = model.predict_action(obs, use_cuda_graph=use_cuda_graph, use_kv_cache=use_kv_cache)["action"][0].float().cpu().numpy()
    finally:
        hook.remove()
    if use_cuda_graph and next(iter(obs.values())).is_cuda:
        runner = model.obs_encoder.scene_encoder._backbone_cuda_graph
        if runner is None or runner.captures < 1 or runner.replays < 1:
            raise RuntimeError("CUDA Graph requested but no visual replay occurred")
    if len(calls) != model.num_inference_timesteps:
        raise RuntimeError(
            f"Expected NFE={model.num_inference_timesteps}, got {len(calls)}"
        )
    if action.shape != (model.n_action_steps, 14) or not np.isfinite(action).all():
        raise ValueError(f"Invalid predicted action: {action.shape}")
    return action


def metrics(prediction, target):
    diff = prediction.astype(np.float64) - target
    return {
        "joint_rmse_rad": float(np.sqrt(np.mean(diff[..., JOINTS] ** 2))),
        "gripper_mae": float(np.mean(np.abs(diff[..., GRIPPERS]))),
        "mse_all": float(np.mean(diff**2)),
        "joint_rmse_by_horizon": np.sqrt(
            np.mean(diff[..., JOINTS] ** 2, axis=(0, 2))
        ).tolist(),
    }


def save_state_comparison(
    gt_states: np.ndarray,
    pred_states: np.ndarray,
    save_dir: Path,
    chunk_size: int,
    state_names: list = None,
    filename: str = "state_comparison.png",
):
    """OpenPI's trajectory panel, without its additional MAE figure."""
    import os
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    n_frames, n_dims = gt_states.shape
    timesteps = np.arange(n_frames)

    if state_names is None:
        state_names = [f"dim_{i}" for i in range(n_dims)]

    # --- 1. 逐 state 维度对比 ---
    n_cols = 4
    n_rows = (n_dims + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.5 * n_rows))
    fig.suptitle("State: GT vs Predicted (full episode)", fontsize=14, fontweight="bold")
    axes = axes.flatten()

    for i in range(n_dims):
        ax = axes[i]
        ax.plot(timesteps, gt_states[:, i], "b-", label="GT", linewidth=1.5, alpha=0.8)
        ax.plot(timesteps, pred_states[:, i], "r--", label="Pred", linewidth=1.5, alpha=0.8)
        # 画 chunk 分界线
        for seg_start in range(0, n_frames, chunk_size):
            ax.axvline(x=seg_start, color="gray", linestyle=":", alpha=0.3)
        ax.set_title(state_names[i], fontsize=11, fontweight="bold")
        ax.set_xlabel("t")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for i in range(n_dims, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    path1 = save_dir / filename
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()


def save_plots(
    output, episode, state, action, anchors, predictions, names, n_obs,
    images, keys, source_color, nfe,
):
    """Plot the existing future-GT windows in the user's OpenPI style."""
    chunk = predictions.shape[1]
    gt_states = np.concatenate(
        [action[int(anchor):int(anchor) + chunk] for anchor in anchors], axis=0
    )
    pred_states = predictions.reshape(-1, predictions.shape[-1])
    if gt_states.shape != pred_states.shape:
        raise ValueError("GT and predicted plot trajectories must align")
    save_state_comparison(gt_states, pred_states, output, chunk, state_names=names)

def use_fp32(model):
    model.float()
    # This plain attribute is not a parameter/buffer, so Module.float() cannot update it.
    model.model.t_embedder.dtype = torch.float32


def training_dataset_options(checkpoint, config_path=None):
    """Read the actual resolved training recipe, including checkpoint subdirectories."""
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    candidates = ([Path(config_path)] if config_path else [
        checkpoint / "resolved_config.yaml", checkpoint.parent / "resolved_config.yaml",
        checkpoint / "hydra/.hydra/config.yaml"])
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise FileNotFoundError("Need the run's resolved_config.yaml or --training-config to identify train/test episodes")
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return cfg["train_dataset"]


def choose_episodes(root, dataset_options, requested=None):
    """Reuse the training dataset's exact split/filter/downsampling logic, without decoding video."""
    from types import SimpleNamespace
    Dataset = _import_module("100hzs1.datasets.dataset").S1Dataset
    Metadata = _import_module("lerobot.datasets.lerobot_dataset").LeRobotDatasetMetadata
    metadata = Metadata(repo_id=dataset_options.get("repo_id", str(root)), root=root)
    holder = SimpleNamespace(dataset_meta=metadata,
        val_ratio=dataset_options.get("val_ratio", .02),
        seed=dataset_options.get("seed", 42),
        task_list=dataset_options.get("task_list"),
        task_index_list=dataset_options.get("task_index_list"),
        black_index=dataset_options.get("black_index"),
        max_train_episodes=dataset_options.get("max_train_episodes", dataset_options.get("demo")))
    train = Dataset._setup_train_val_split(holder).tolist()
    test = np.flatnonzero(holder.val_mask).tolist() if holder.val_mask is not None else []
    if requested is None:
        if not train or not test:
            raise ValueError("Need both training and held-out episodes")
        return [("train", train[0]), ("test", test[0])]
    if len(set(requested)) != len(requested):
        raise ValueError("Duplicate episode selection")
    result = []
    for episode in requested:
        if episode not in train and episode not in test:
            raise ValueError(f"Episode {episode} is outside this run's train/test split")
        result.append(("train" if episode in train else "test", episode))
    return result


def checkpoint_name(checkpoint):
    name = checkpoint.name
    if name.startswith("checkpoint-") or name == "ema":
        name = checkpoint.parent.name + "_" + name
    return name


def figure_path(output, checkpoint, split, episode):
    return output / f"{checkpoint_name(checkpoint)}_{split}_ep{episode:06d}.png"


def run(args):
    options = load_inference_options(args.inference_config, {
        "num_inference_timesteps": args.nfe, "precision": args.precision,
        "use_cuda_graph": args.cuda_graph, "use_kv_cache": args.kv_cache,
    })
    args.checkpoint = args.checkpoint.resolve()
    root = args.dataset
    if not (root / "meta/info.json").is_file():
        root = root / args.task
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("robotwin_action_space") != "absolute_joint14":
        raise ValueError("Dataset must declare robotwin_action_space=absolute_joint14")
    episodes = choose_episodes(root, training_dataset_options(args.checkpoint, args.training_config), args.episodes)
    paths = [figure_path(args.output, args.checkpoint, split, ep) for split, ep in episodes]
    if any(p.exists() for p in paths):
        raise FileExistsError("An output image already exists; select another output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps(dict(checkpoint=str(args.checkpoint), dataset=str(root),
            inference=options, source_color=args.source_color,
            episodes=[dict(split=split, episode=ep, image=str(path))
                      for (split, ep), path in zip(episodes, paths)]), indent=2))
        return
    torch.set_num_threads(4)
    model = load_policy(str(args.checkpoint), device="cpu", inference_config=args.inference_config).eval()
    configure_policy_inference(model, args.inference_config, options)
    model.to(args.device)
    if model.action_key != "state" or model.supervision_key not in {"action", "state"}:
        raise ValueError("Only absolute joint14 policies are supported")
    keys = list(model.obs_encoder.rgb_key)
    if set(keys) - CAMERAS.keys():
        raise ValueError(f"Unmapped cameras: {keys}")
    chunk, n_obs, n_image = model.n_action_steps, model.n_obs_steps, model.n_image_obs_steps
    if chunk != model.horizon - n_obs:
        raise ValueError("Expected full future chunk")
    report = []
    names = info["features"]["action"]["names"]
    for (split, episode), path in zip(episodes, paths):
        location = dict(episode_index=episode, episode_chunk=episode // info["chunks_size"])
        length = pq.read_metadata(root / info["data_path"].format(**location)).num_rows
        anchors = args.anchors or list(range(n_obs, length - chunk + 1, args.stride or chunk))
        if not anchors:
            raise ValueError(f"No complete windows in episode {episode}")
        state, action, images, windows = load_episode(root, info, episode, keys, anchors, n_obs, n_image, chunk)
        predictions = []
        for anchor, window in zip(anchors, windows):
            obs = build_observation(state, images, window, keys, args.device, args.source_color)
            noise_seed = (args.seed + episode * 1_000_003 + int(anchor)) % (2**31)
            predictions.append(predict(model, obs, noise_seed, options["precision"],
                use_cuda_graph=options["use_cuda_graph"], use_kv_cache=options["use_kv_cache"]))
        predictions = np.stack(predictions)
        target = np.stack([action[window[2]] for window in windows])
        # Exact same GT/predicted trajectory panel as the specified OpenPI plot.
        save_state_comparison(target.reshape(-1, 14), predictions.reshape(-1, 14),
                              args.output, chunk, state_names=names, filename=path.name)
        result = dict(split=split, episode=episode, image=str(path), chunks=len(anchors),
                      prediction=metrics(predictions, target))
        report.append(result)
        print("OPENLOOP_EPISODE " + json.dumps(result), flush=True)
    print("OPENLOOP_COMPLETED " + json.dumps(report), flush=True)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/DRoboticsAILab/vepfs_20260622/licheng.yang/datasets/robotwin2.0_lerobot_clean_joint14"
        ),
    )
    p.add_argument("--task", default="beat_block_hammer")
    p.add_argument("--episodes", type=int, nargs="+", default=None,
                   help="Default: first training and first held-out episode from this run")
    p.add_argument("--training-config", type=Path, help="Resolved training YAML; otherwise inferred from checkpoint")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--anchors",
        type=int,
        nargs="+",
        help="First future action indices; default whole episode",
    )
    p.add_argument("--stride", type=int, help="Default action chunk length")
    p.add_argument("--output", type=Path, default=Path("/DRoboticsAILab/users/licheng.yang/openloop_eval"))
    p.add_argument("--source-color", choices=["bgr", "rgb"], default="bgr")
    p.add_argument("--precision", choices=["bf16", "fp32"], default=None)
    p.add_argument("--inference-config", type=Path)
    p.add_argument("--cuda-graph", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--kv-cache", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--nfe", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    if (arguments.nfe is not None and arguments.nfe < 1) or (arguments.stride is not None and arguments.stride < 1):
        raise ValueError("nfe and stride must be positive")
    run(arguments)
