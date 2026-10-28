#!/usr/bin/env python3
"""RTPolicy ONNX inference example (real_stack_cover 3v cnnb export series).

Interface (raw images in, action out; preprocessing is embedded in the graph):
  Inputs:
    head_cam / left_cam / right_cam : (1, 3, 3, 480, 640)  raw RGB [0,1]
        ^ time dim = n_obs_steps = 3 (only the latest frame is used, but the
          input must provide all 3 frames)
    state                            : (1, 3, 16)           3 frames of history state
    noise_trajectory                 : (1, 48, 16)          horizon x action_dim
  Outputs:
    action      : (1, 45, 16)   first 45 executed action steps (unnormalized)
    action_pred : (1, 48, 16)   full-horizon prediction

Usage:
  python scripts/infer_rtpolicy.py --onnx-path model.onnx \
      --head-cam h.npy --left-cam l.npy --right-cam r.npy --state s.npy
  Or --image-dir <dir with head_cam/left_cam/right_cam + state npy files>
"""
import argparse
import glob
import json
import os

import numpy as np
import onnxruntime as ort


class VodppOnnxRunner:
    """RTPolicy ONNX inference with a history buffer.

    Typical deployment usage: call push() every frame, then predict() once
    n_obs_steps frames have accumulated.
    """

    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        self.metadata = self._load_metadata(onnx_path)
        self.session = ort.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"],
        )
        m = self.metadata
        self.n_obs_steps = m["n_obs_steps"]
        self.horizon = m["horizon"]
        self.action_dim = m["action_dim"]
        self.rgb_keys = m["rgb_keys"]
        self.rng = np.random.default_rng(0)
        # History queues: rgb stored as (T,C,H,W), state as (T,D)
        self.rgb_history = {k: [] for k in self.rgb_keys}
        self.state_history = []

    @staticmethod
    def _load_metadata(onnx_path):
        path = f"{onnx_path}.metadata.json"
        if not os.path.isfile(path):
            return {}
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _to_chw_float(image) -> np.ndarray:
        """hwc uint8 / hwc float / chw float -> chw float [0,1]"""
        img = np.asarray(image)
        if img.ndim == 3 and img.shape[-1] == 3:
            img = np.moveaxis(img, -1, 0)
        img = img.astype(np.float32)
        if img.max() > 2.0:
            img = img / 255.0
        return img

    def push(self, images: dict, state: np.ndarray) -> None:
        """Push one observation frame. images: {rgb_key: HWC or CHW image}"""
        for key in self.rgb_keys:
            self.rgb_history[key].append(self._to_chw_float(images[key]))
        self.state_history.append(np.asarray(state, dtype=np.float32).reshape(-1))

    def ready(self) -> bool:
        return len(self.state_history) >= self.n_obs_steps

    def _take_last(self, seq, n):
        return seq[-n:] if len(seq) >= n else None

    def predict(self, noise: np.ndarray = None) -> dict:
        """Requires ready(). Returns {'action': (1,45,16), 'action_pred': (1,48,16)}."""
        if not self.ready():
            raise RuntimeError(f"need {self.n_obs_steps} frames, have {len(self.state_history)}")
        T = self.n_obs_steps
        feed = {}
        for key in self.rgb_keys:
            frames = self._take_last(self.rgb_history[key], T)
            feed[key] = np.stack(frames, axis=0)[None]          # (1,T,C,H,W)
        feed["state"] = np.stack(self.state_history[-T:], axis=0)[None]  # (1,T,D)
        if noise is None:
            noise = self.rng.standard_normal((1, self.horizon, self.action_dim))
        feed["noise_trajectory"] = noise.astype(np.float32)

        # Feed only inputs the session declares (order of input_names does not matter)
        declared = {i.name for i in self.session.get_inputs()}
        feed = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in feed.items() if k in declared}
        outputs = self.session.run(None, feed)
        return {o.name: v for o, v in zip(self.session.get_outputs(), outputs)}

    def reset(self) -> None:
        self.rgb_history = {k: [] for k in self.rgb_keys}
        self.state_history = []


def demo_standalone(args):
    """CLI mode: read inputs from npy files and run one inference."""
    runner = VodppOnnxRunner(args.onnx_path)
    m = runner.metadata
    T = m.get("n_obs_steps", 3)
    H, W = 480, 640
    raw_shape = m.get("preprocess", {}).get("raw_image_shape")
    if raw_shape:
        _, H, W = raw_shape

    rng = np.random.default_rng(0)
    feed = {}
    for key in m.get("rgb_keys", ["head_cam", "left_cam", "right_cam"]):
        if args.image_dir:
            feed[key] = np.load(os.path.join(args.image_dir, f"{key}.npy"))
        elif getattr(args, key.replace("_cam", "_cam"), None):
            feed[key] = np.load(getattr(args, key))
        else:
            print(f"[warn] {key} not provided, using a random placeholder image")
            feed[key] = rng.random((1, T, 3, H, W), dtype=np.float32)
    state = (
        np.load(args.state) if args.state
        else rng.random((1, T, 16), dtype=np.float32)
    )
    noise = (
        np.load(args.noise_trajectory) if args.noise_trajectory
        else rng.standard_normal((1, m.get("horizon", 48), m.get("action_dim", 16)))
    )

    runner.push({k: v[0, 0] for k, v in feed.items()}, state[0, 0])
    for t in range(1, T):
        runner.push({k: v[0, t] for k, v in feed.items()}, state[0, t])
    out = runner.predict(noise)

    for name, value in out.items():
        print(f"{name}: shape={value.shape} min={value.min():.4f} max={value.max():.4f}")
        print(np.array2string(value[0, :3], precision=4, suppress_small=True))
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        for name, value in out.items():
            np.save(os.path.join(args.output_dir, f"{name}.npy"), value)


def parse_args():
    parser = argparse.ArgumentParser(description="RTPolicy ONNX inference")
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--head-cam"), parser.add_argument("--left-cam"), parser.add_argument("--right-cam")
    parser.add_argument("--state", help="npy (T,16) or (1,T,16)")
    parser.add_argument("--noise-trajectory", help="npy (horizon,16) or (1,horizon,16)")
    parser.add_argument("--image-dir", help="directory containing <rgb_key>.npy files")
    parser.add_argument("--output-dir")
    return parser.parse_args()


if __name__ == "__main__":
    demo_standalone(parse_args())
