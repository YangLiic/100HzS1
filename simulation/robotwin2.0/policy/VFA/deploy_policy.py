"""Run VFA in the RoboTwin process and the same Python/Torch environment."""

from collections import deque
from pathlib import Path
import os
import cv2
import numpy as np
import torch
from safetensors.torch import load_model
from vfa.models.base_runner import load_policy

CAMERAS = {
    "left_cam": "left_camera",
    "right_cam": "right_camera",
    "head_cam": "head_camera",
    "front_cam": "front_camera",
}


def encode_obs(observation, rgb_keys, jpeg=True):
    result = {
        "state": np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
    }
    if result["state"].shape != (14,):
        raise ValueError("This adapter requires joint14 observations")
    for key in rgb_keys:
        image = np.asarray(observation["observation"][CAMERAS[key]]["rgb"])
        if image.dtype != np.uint8 or image.shape != (240, 320, 3):
            raise ValueError(
                f"{key}: expected RGB uint8 240x320x3, got {image.shape}/{image.dtype}"
            )
        if jpeg:
            ok, encoded = cv2.imencode(".jpg", image)
            if not ok:
                raise RuntimeError("JPEG preprocessing failed")
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        result[key] = np.ascontiguousarray(image)  # SAPIEN already supplies RGB.
    return result


class ObservationHistory:
    def __init__(self, n_state, n_image, rgb_keys):
        self.n_state, self.n_image, self.rgb_keys = n_state, n_image, set(rgb_keys)
        self.values = deque(maxlen=max(n_state, n_image))

    def clear(self):
        self.values.clear()

    def append(self, value):
        self.values.append(value)

    def stacked(self):
        if not self.values:
            raise ValueError("No observation history")
        result = {}
        for key in self.values[-1]:
            count = self.n_image if key in self.rgb_keys else self.n_state
            values = [row[key] for row in list(self.values)[-count:]]
            result[key] = np.stack([values[0]] * (count - len(values)) + values)
        return result


class DirectPolicy:
    def __init__(self, checkpoint, device="cuda:0", nfe=4, jpeg=True, use_cuda_graph=True):
        if nfe < 1:
            raise ValueError("nfe must be positive")
        self.policy = load_policy(str(checkpoint), device="cpu").eval()
        load_model(
            self.policy, str(Path(checkpoint) / "model.safetensors"), strict=True
        )
        self.use_cuda_graph = bool(use_cuda_graph)
        self.device = torch.device(device)
        self.policy.to(self.device)
        self.policy.num_inference_timesteps = nfe
        if self.policy.action_key != "state":
            raise ValueError("The current joint14 profile requires action_key=state")
        self.rgb_keys = list(self.policy.obs_encoder.rgb_key)
        self.history = ObservationHistory(
            self.policy.n_obs_steps, self.policy.n_image_obs_steps, self.rgb_keys
        )
        self.jpeg, self.episode_seed, self.infer_index = jpeg, None, 0
        self.network_calls = 0
        self.policy.model.register_forward_hook(self._count_call)
        print(
            f"[VFA] direct/same-process device={self.device} nfe={nfe} "
            f"RGB/no-flip views={self.rgb_keys} obs={self.policy.n_obs_steps} "
            f"images={self.policy.n_image_obs_steps} actions={self.policy.n_action_steps} "
            f"cuda_graph={self.use_cuda_graph} graph_scope=visual_backbone",
            flush=True,
        )

    def _count_call(self, *_):
        self.network_calls += 1

    def set_episode_seed(self, seed):
        self.episode_seed, self.infer_index = int(seed), 0

    def reset(self):
        self.history.clear()
        self.infer_index = 0

    def observe(self, observation):
        self.history.append(encode_obs(observation, self.rgb_keys, self.jpeg))

    def predict(self):
        if self.episode_seed is None:
            raise ValueError("Set the episode seed before inference")
        arrays = self.history.stacked()
        observations = {}
        for key, value in arrays.items():
            if key in self.rgb_keys:
                value = np.moveaxis(value.astype(np.float32) / 255.0, -1, 1)
            observations[key] = (
                torch.from_numpy(np.ascontiguousarray(value))
                .unsqueeze(0)
                .to(self.device)
            )
        torch.manual_seed((self.episode_seed * 1_000_003 + self.infer_index) % (2**31))
        before = self.network_calls
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, dtype=torch.bfloat16),
        ):
            action = (
                self.policy.predict_action(observations, use_cuda_graph=self.use_cuda_graph)["action"][0]
                .float()
                .cpu()
                .numpy()
            )
        if self.use_cuda_graph and self.device.type == "cuda":
            runner = self.policy.obs_encoder.scene_encoder._backbone_cuda_graph
            if runner is None or runner.captures < 1 or runner.replays < 1:
                raise RuntimeError("CUDA Graph was requested but no visual replay occurred")
        actual = self.network_calls - before
        if actual != self.policy.num_inference_timesteps:
            raise RuntimeError(
                f"Expected {self.policy.num_inference_timesteps} network calls, got {actual}"
            )
        if (
            action.shape != (self.policy.n_action_steps, 14)
            or not np.isfinite(action).all()
        ):
            raise ValueError(f"Invalid joint14 actions: {action.shape}")
        self.infer_index += 1
        if self.infer_index == 1 or self.infer_index % 25 == 0:
            print(
                f"[VFA] seed={self.episode_seed} infer={self.infer_index} nfe={actual} "
                f"action_steps={len(action)} cuda_graph={self.use_cuda_graph}"
                + (f" graph_captures={runner.captures} graph_replays={runner.replays}"
                   if self.use_cuda_graph and self.device.type == "cuda" else ""),
                flush=True,
            )
        return action


def get_model(usr_args):
    return DirectPolicy(
        usr_args["ckpt_setting"],
        device=usr_args.get("device", "cuda:0"),
        nfe=int(usr_args.get("nfe", 4)),
        use_cuda_graph=usr_args.get("use_cuda_graph", True),
        jpeg=os.environ.get("VFA_JPEG_ROUNDTRIP", "1") == "1",
    )


def reset_model(model):
    model.reset()


def eval(TASK_ENV, model, observation):
    if not model.history.values:
        model.observe(observation)
    for action in model.predict():
        TASK_ENV.take_action(action, action_type="qpos")
        model.observe(TASK_ENV.get_obs())
        if TASK_ENV.eval_success:
            break
