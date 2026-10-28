"""Direct VFA open-loop inference on joint14 LeRobot videos (no simulator/server)."""

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

from vfa.models.base_runner import load_policy

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


def predict(model, obs, seed, precision):
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
            action = model.predict_action(obs)["action"][0].float().cpu().numpy()
    finally:
        hook.remove()
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


def save_plots(
    output,
    episode,
    state,
    action,
    anchors,
    predictions,
    names,
    n_obs,
    images,
    keys,
    source_color,
    nfe,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chunk = predictions.shape[1]
    fig, axes = plt.subplots(7, 2, figsize=(18, 18), sharex=True)
    for dim, ax in enumerate(axes.flat):
        ax.plot(np.arange(len(action)), action[:, dim], color="black", lw=1, label="GT")
        for index, (anchor, prediction) in enumerate(zip(anchors, predictions)):
            ax.plot(
                np.arange(anchor, anchor + chunk),
                prediction[:, dim],
                color="tab:red",
                lw=0.8,
                alpha=0.8,
                label="Predicted chunks" if index == 0 else None,
            )
        ax.set_title(names[dim] + (" (gripper)" if dim in GRIPPERS else " (rad)"))
        ax.grid(alpha=0.2)
        if dim == 0:
            ax.legend()
    for ax in axes[-1]:
        ax.set_xlabel(
            "Dataset frame (30 Hz); each red segment is independently conditioned on GT"
        )
    fig.suptitle(
        f"Episode {episode}: GT vs VFA chunks, NFE={nfe}, action chunk={chunk}"
    )
    fig.tight_layout()
    fig.savefig(output / "trajectory.png", dpi=130)
    plt.close(fig)

    selected = np.unique(
        np.linspace(0, len(anchors) - 1, min(3, len(anchors)), dtype=int)
    )
    for index in selected:
        anchor = int(anchors[index])
        fig, axes = plt.subplots(7, 2, figsize=(15, 16), sharex=True)
        past = np.arange(anchor - n_obs, anchor)
        future = np.arange(anchor, anchor + chunk)
        for dim, ax in enumerate(axes.flat):
            ax.plot(
                past,
                state[past, dim],
                "o-",
                color="tab:blue",
                label="Input state history",
            )
            ax.plot(
                future,
                action[future, dim],
                "o-",
                color="black",
                label="GT future actions",
            )
            ax.plot(
                future, predictions[index, :, dim], "x--", color="tab:red", label="VFA"
            )
            ax.axvline(anchor - 0.5, color="gray", ls=":")
            ax.set_title(names[dim])
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            f"Episode {episode}, first predicted frame={anchor}, latest input={anchor - 1}"
        )
        fig.tight_layout()
        fig.savefig(output / f"chunk_{anchor:06d}.png", dpi=130)
        plt.close(fig)
    anchor = int(anchors[len(anchors) // 2])
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4), squeeze=False)
    for ax, key in zip(axes.flat, keys):
        ax.imshow(source_to_rgb(images[key][anchor - 1], source_color))
        ax.set_title(f"{key}: input frame {anchor - 1}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output / "input_rgb.png", dpi=130)
    plt.close(fig)


def use_fp32(model):
    model.float()
    # This plain attribute is not a parameter/buffer, so Module.float() cannot update it.
    model.model.t_embedder.dtype = torch.float32


def run(args):
    root = args.dataset
    if not (root / "meta/info.json").is_file():
        root = root / args.task
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("robotwin_action_space") != "absolute_joint14":
        raise ValueError("Dataset must declare robotwin_action_space=absolute_joint14")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Use a fresh output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = load_policy(str(args.checkpoint), device="cpu").eval()
    load_model(model, str(args.checkpoint / "model.safetensors"), strict=True)
    model.to(args.device)
    if args.precision == "fp32":
        use_fp32(model)  # Convert stored tensors AND the timestep embedding dtype.
    model.num_inference_timesteps = args.nfe
    if model.action_key != "state" or model.supervision_key not in {"action", "state"}:
        raise ValueError("Only absolute joint14 state/action policies are supported")
    keys = list(model.obs_encoder.rgb_key)
    if set(keys) - CAMERAS.keys():
        raise ValueError(f"Unmapped cameras: {keys}")
    chunk, n_obs, n_image = (
        model.n_action_steps,
        model.n_obs_steps,
        model.n_image_obs_steps,
    )
    if chunk != model.horizon - n_obs:
        raise ValueError("Expected full future chunk: horizon - n_obs_steps")
    report = {
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checkpoint_config_sha256": hashlib.sha256(
            (args.checkpoint / "config.json").read_bytes()
        ).hexdigest(),
        "checkpoint_weights_size": (args.checkpoint / "model.safetensors")
        .stat()
        .st_size,
        "constant_velocity_available": n_obs >= 2,
        "checkpoint": str(args.checkpoint),
        "dataset": str(root),
        "nfe": args.nfe,
        "precision": args.precision,
        "source_color": args.source_color,
        "n_obs_steps": n_obs,
        "n_image_obs_steps": n_image,
        "action_chunk": chunk,
        "horizon": model.horizon,
        "image_keys": keys,
        "seed": args.seed,
        "alignment": "states[a-n_obs:a], images[a-n_image:a], GT actions[a:a+chunk]",
        "original_recipe_validation_episode": 4,
        "episodes": {},
    }
    names = info["features"]["action"]["names"]
    all_pred, all_gt, all_hold, all_velocity = [], [], [], []
    ablation_cases = []
    for episode in args.episodes:
        location = dict(
            episode_index=episode, episode_chunk=episode // info["chunks_size"]
        )
        length = pq.read_metadata(root / info["data_path"].format(**location)).num_rows
        anchors = args.anchors or list(
            range(n_obs, length - chunk + 1, args.stride or chunk)
        )
        if not anchors:
            raise ValueError(f"No complete windows in episode {episode}")
        state, action, images, windows = load_episode(
            root, info, episode, keys, anchors, n_obs, n_image, chunk
        )
        predictions = []
        for index, window in enumerate(windows):
            obs = build_observation(
                state, images, window, keys, args.device, args.source_color
            )
            noise_seed = (args.seed + episode * 1_000_003 + int(anchors[index])) % (
                2**31
            )
            predictions.append(predict(model, obs, noise_seed, args.precision))
            if index % 25 == 0:
                print(
                    f"episode={episode} chunks={index + 1}/{len(anchors)}", flush=True
                )
        predictions = np.stack(predictions)
        target = np.stack([action[w[2]] for w in windows])
        hold = np.stack([np.repeat(state[w[0][-1:]], chunk, axis=0) for w in windows])
        if n_obs >= 2:
            delta = np.stack([state[w[0][-1]] - state[w[0][-2]] for w in windows])
            velocity = hold + delta[:, None] * np.arange(1, chunk + 1)[None, :, None]
        else:
            velocity = hold.copy()
        result = {
            "chunks": len(anchors),
            "frames": length,
            "state_action_max_abs_diff": float(np.max(np.abs(state - action))),
            "prediction": metrics(predictions, target),
            "hold_last": metrics(hold, target),
            "constant_velocity": metrics(velocity, target),
        }
        moving = (
            np.max(np.abs(target[..., JOINTS] - hold[..., JOINTS]), axis=(1, 2)) > 0.01
        )
        result["moving_chunks"] = int(moving.sum())
        if moving.any():
            result["moving_only"] = {
                name: metrics(value[moving], target[moving])
                for name, value in [
                    ("prediction", predictions),
                    ("hold_last", hold),
                    ("constant_velocity", velocity),
                ]
            }
        folder = args.output / f"episode_{episode:06d}"
        folder.mkdir()
        np.savez_compressed(
            folder / "predictions.npz",
            anchors=np.asarray(anchors),
            prediction=predictions,
            gt=target,
            hold_last=hold,
            constant_velocity=velocity,
            input_state=np.stack([state[w[0]] for w in windows]),
            image_indices=np.stack([w[1] for w in windows]),
            state=state,
            action=action,
        )
        save_plots(
            folder,
            episode,
            state,
            action,
            anchors,
            predictions,
            names,
            n_obs,
            images,
            keys,
            args.source_color,
            args.nfe,
        )
        (folder / "metrics.json").write_text(json.dumps(result, indent=2))
        report["episodes"][str(episode)] = result
        for group, value in zip(
            [all_pred, all_gt, all_hold, all_velocity],
            [predictions, target, hold, velocity],
        ):
            group.append(value)
        if args.ablations:
            for i in np.unique(
                np.linspace(0, len(windows) - 1, min(4, len(windows)), dtype=int)
            ):
                ablation_cases.append(
                    (
                        episode,
                        anchors[i],
                        state,
                        images,
                        windows[i],
                        windows[(i + len(windows) // 2) % len(windows)][1],
                        predictions[i],
                        target[i],
                    )
                )
        print(json.dumps({"episode": episode, **result}), flush=True)
    report["aggregate"] = {
        name: metrics(np.concatenate(values), np.concatenate(all_gt))
        for name, values in [
            ("prediction", all_pred),
            ("hold_last", all_hold),
            ("constant_velocity", all_velocity),
        ]
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    if args.ablations:
        # BF16 production model first; only cast to FP32 after the other variants are finished.
        variants = ["black_images", "shuffle_images", "raw_color", "jpeg", "fp32"]
        report["ablations"] = {}
        for variant in variants:
            if variant == "fp32":
                use_fp32(model)
            values, targets, deltas = [], [], []
            for (
                episode,
                anchor,
                state,
                images,
                window,
                alternate,
                reference,
                target,
            ) in ablation_cases:
                obs = build_observation(
                    state,
                    images,
                    window,
                    keys,
                    args.device,
                    args.source_color,
                    variant,
                    alternate,
                )
                seed = (args.seed + episode * 1_000_003 + int(anchor)) % (2**31)
                value = predict(
                    model, obs, seed, "fp32" if variant == "fp32" else args.precision
                )
                values.append(value)
                targets.append(target)
                deltas.append(value - reference)
            report["ablations"][variant] = {
                **metrics(np.stack(values), np.stack(targets)),
                "prediction_change_rmse": float(
                    np.sqrt(np.mean(np.stack(deltas) ** 2))
                ),
                "samples": len(values),
            }
        report["ablations"]["correct_reference"] = metrics(
            np.stack([s[-2] for s in ablation_cases]),
            np.stack([s[-1] for s in ablation_cases]),
        )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2))
    (args.output / "OPENLOOP_COMPLETED").write_text(
        "Complete; this is not closed-loop success rate.\n"
    )
    print("OPENLOOP_COMPLETED " + json.dumps(report["aggregate"]), flush=True)


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
    p.add_argument("--task", default="blocks_ranking_rgb")
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 4])
    p.add_argument(
        "--anchors",
        type=int,
        nargs="+",
        help="First future action indices; default whole episode",
    )
    p.add_argument("--stride", type=int, help="Default action chunk length")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-color", choices=["bgr", "rgb"], default="bgr")
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--nfe", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--ablations",
        action="store_true",
        help="Four sampled chunks per episode: image/precision tests",
    )
    return p


if __name__ == "__main__":
    arguments = parser().parse_args()
    if arguments.nfe < 1 or (arguments.stride is not None and arguments.stride < 1):
        raise ValueError("nfe and stride must be positive")
    run(arguments)
