#!/usr/bin/env python3
"""ONNX Runtime inference script for exported VODP+ policy models.

This script:
  1. Loads an ONNX model and its companion ``*.metadata.json`` (or ``--metadata-path``)
  2. Preprocesses raw RGB images (resize / crop / normalize) per metadata ``preprocess``,
     unless ``image_preprocess_in_onnx`` is true (preprocessing is already in the ONNX graph)
  3. Builds ``state``, ``noise_trajectory``, and other inputs, then runs ONNX Runtime
  4. Prints or saves output tensors (typically ``action`` and ``action_pred``)

Dependencies
------------
numpy, torch, onnxruntime

Command-line usage
------------------
::

    python scripts/infer_onnx.py \\
      --onnx-path checkpoint-160000-fp16.onnx \\
      --state checkpoints/.../state.npy \\
      --noise-trajectory checkpoints/.../noise_trajectory.npy \\
      --image head_cam=checkpoints/.../head_cam.npy \\
      [--metadata-path checkpoint-160000-fp16.onnx.metadata.json] \\
      [--output-dir ./outputs] \\
      [--provider CUDAExecutionProvider] \\
      [--num-runs 100]
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F


_ONNX_NUMPY_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(bool)": np.bool_,
}


def load_metadata(onnx_path, metadata_path=None):
    with open(metadata_path or f"{onnx_path}.metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _chw(image):
    image = np.asarray(image, dtype=np.float32)
    if image.shape[-1] == 3:
        image = np.moveaxis(image, -1, 0)
    if image.max() > 2.0:
        image = image / 255.0
    return image


def _resize(image, height, width, mode):
    kwargs = {"align_corners": False, "antialias": False} if mode in ("bilinear", "bicubic") else {}
    tensor = torch.from_numpy(image).unsqueeze(0)
    return F.interpolate(tensor, size=(height, width), mode=mode, **kwargs).squeeze(0).numpy()


def _interpolate_config(value, width):
    if not value:
        return None
    if isinstance(value, int):
        return "bilinear", value
    if isinstance(value, (list, tuple)):
        return value[0], width if len(value) < 2 or value[1] is None else int(value[1])
    return value, width


def preprocess_images(images, cfg):
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None]

    patch_size = int(cfg["patch_size"])
    mode = cfg.get("mode", "crop")
    processed = []

    for image in images:
        image = np.clip(_chw(image), 0.0, 1.0)
        _, height, width = image.shape

        interpolate = _interpolate_config(cfg.get("interpolate", "bilinear"), width)
        if interpolate:
            interpolate_mode, target_size = interpolate
            target_width = round(target_size / patch_size) * patch_size
            resized_height = round(height * (target_width / width) / patch_size) * patch_size
            image = _resize(image, int(resized_height), int(target_width), interpolate_mode)
            _, height, width = image.shape

        if mode == "pad":
            size = round(max(width, height) // patch_size) * patch_size
            h_pad, w_pad = size - height, size - width
            image = np.pad(
                image,
                ((0, 0), (h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2)),
                mode="constant",
                constant_values=1.0,
            )
        else:
            new_width = (width // patch_size) * patch_size
            new_height = min((height // patch_size) * patch_size, new_width)
            y0 = (height - new_height) // 2
            x0 = (width - new_width) // 2
            image = image[:, y0 : y0 + new_height, x0 : x0 + new_width]

        processed.append(image.astype(np.float32, copy=False))

    max_h = max(image.shape[-2] for image in processed)
    max_w = max(image.shape[-1] for image in processed)
    processed = [
        np.pad(
            image,
            (
                (0, 0),
                (
                    (max_h - image.shape[-2]) // 2,
                    max_h - image.shape[-2] - (max_h - image.shape[-2]) // 2,
                ),
                (
                    (max_w - image.shape[-1]) // 2,
                    max_w - image.shape[-1] - (max_w - image.shape[-1]) // 2,
                ),
            ),
            mode="constant",
            constant_values=1.0,
        )
        for image in processed
    ]

    if cfg.get("do_normalize", False):
        mean = np.asarray(cfg["image_mean"], dtype=np.float32).reshape(3, 1, 1)
        std = np.asarray(cfg["image_std"], dtype=np.float32).reshape(3, 1, 1)
        processed = [(image - mean) / std for image in processed]

    return np.stack(processed).astype(np.float32, copy=False)


def preprocess_image_sequence(images, cfg):
    images = np.asarray(images)
    if images.ndim <= 4:
        return preprocess_images(images, cfg)

    batch, steps = images.shape[:2]
    flat = images.reshape((-1, *images.shape[2:]))
    processed = preprocess_images(flat, cfg)
    return processed.reshape((batch, steps, *processed.shape[1:]))


def _image_preprocess_in_onnx(metadata):
    return bool(metadata.get("image_preprocess_in_onnx"))


def build_inputs(images, state, noise_trajectory, metadata):
    if _image_preprocess_in_onnx(metadata):
        inputs = {
            key: np.asarray(images[key], dtype=np.float32)
            for key in metadata["rgb_keys"]
        }
    else:
        inputs = {
            key: preprocess_image_sequence(images[key], metadata["preprocess"])
            for key in metadata["rgb_keys"]
        }
    for key, value in list(inputs.items()):
        if value.ndim == 4:
            inputs[key] = value[None]

    state_key = metadata.get("state_key") or "state"
    state = np.asarray(state, dtype=np.float32)
    if state.ndim == 1:
        state = state[None, None]
    elif state.ndim == 2:
        state = state[None]
    inputs[state_key] = state

    noise_trajectory = np.asarray(noise_trajectory, dtype=np.float32)
    if noise_trajectory.ndim == 2:
        noise_trajectory = noise_trajectory[None]
    inputs["noise_trajectory"] = noise_trajectory
    return inputs


def _is_fp16_model(metadata):
    dtype = str(metadata.get("onnx_input_dtype") or metadata.get("export_dtype") or "").lower()
    return dtype in {"fp16", "float16", "half"}


def _default_providers(ort, metadata):
    available = set(ort.get_available_providers())
    if _is_fp16_model(metadata) and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _create_session(onnx_path, metadata, providers=None):
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    disabled_optimizers = None
    if _is_fp16_model(metadata):
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        disabled_optimizers = ["SimplifiedLayerNormFusion"]

    return ort.InferenceSession(
        onnx_path,
        sess_options=sess_options,
        providers=providers or _default_providers(ort, metadata),
        disabled_optimizers=disabled_optimizers,
    )


def _cast_input_to_onnx_type(value, onnx_type):
    dtype = _ONNX_NUMPY_DTYPES.get(onnx_type)
    if dtype is None:
        return value
    return np.asarray(value).astype(dtype, copy=False)


def _prepare_session_inputs(session, images, state, noise_trajectory, metadata):
    inputs = build_inputs(images, state, noise_trajectory, metadata)
    return {
        item.name: _cast_input_to_onnx_type(inputs[item.name], item.type)
        for item in session.get_inputs()
    }


def infer(onnx_path, images, state, noise_trajectory, metadata_path=None, providers=None):
    metadata = load_metadata(onnx_path, metadata_path)
    session = _create_session(onnx_path, metadata, providers=providers)
    inputs = _prepare_session_inputs(session, images, state, noise_trajectory, metadata)
    outputs = session.run(None, inputs)
    return {item.name: value for item, value in zip(session.get_outputs(), outputs)}


def benchmark(
    onnx_path,
    images,
    state,
    noise_trajectory,
    metadata_path=None,
    providers=None,
    num_runs=100,
    warmup_runs=5,
):
    metadata = load_metadata(onnx_path, metadata_path)
    session = _create_session(onnx_path, metadata, providers=providers)
    inputs = _prepare_session_inputs(session, images, state, noise_trajectory, metadata)

    for _ in range(warmup_runs):
        session.run(None, inputs)

    times_ms = []
    last_outputs = None
    for _ in range(num_runs):
        start = time.perf_counter()
        last_outputs = session.run(None, inputs)
        times_ms.append((time.perf_counter() - start) * 1000.0)

    outputs = {item.name: value for item, value in zip(session.get_outputs(), last_outputs)}
    stats = {
        "num_runs": num_runs,
        "mean_ms": float(np.mean(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
    }
    return outputs, stats


def format_array(value, precision=6):
    with np.printoptions(precision=precision, suppress=True, linewidth=120):
        return np.array2string(np.asarray(value), separator=", ")


def print_outputs(outputs):
    for key, value in outputs.items():
        arr = np.asarray(value)
        print(f"{key} shape={arr.shape} dtype={arr.dtype}")
        print(f"  min={arr.min():.6g} max={arr.max():.6g} mean={arr.mean():.6g}")
        print(format_array(arr))


def print_benchmark(stats):
    print(f"Benchmark ({stats['num_runs']} runs):")
    print(f"  mean={stats['mean_ms']:.3f} ms")
    print(f"  min={stats['min_ms']:.3f} ms")
    print(f"  max={stats['max_ms']:.3f} ms")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--metadata-path")
    parser.add_argument("--state", required=True)
    parser.add_argument("--noise-trajectory", required=True)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--provider", action="append")
    parser.add_argument("--num-runs", type=int, default=100, help="Timed inference iterations for benchmark")
    return parser.parse_args()


def main():
    args = parse_args()
    images = {}
    for item in args.image:
        key, path = item.split("=", 1)
        images[key] = np.load(path)

    outputs, stats = benchmark(
        args.onnx_path,
        images=images,
        state=np.load(args.state),
        noise_trajectory=np.load(args.noise_trajectory),
        metadata_path=args.metadata_path,
        providers=args.provider,
        num_runs=args.num_runs,
    )
    print_benchmark(stats)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for key, value in outputs.items():
            np.save(os.path.join(args.output_dir, f"{key}.npy"), value)
    print_outputs(outputs)


if __name__ == "__main__":
    main()