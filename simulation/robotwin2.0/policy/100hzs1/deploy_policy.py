"""Run YAML-selected eager/TensorRT inference inside the RoboTwin control loop."""
from importlib import import_module as _import_module

from collections import deque
from pathlib import Path
import os
import cv2
import numpy as np
load_inference_policy = _import_module('100hzs1.inference').load_inference_policy

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
    def __init__(self, checkpoint, device="cuda:0", nfe=None, jpeg=True,
                 use_cuda_graph=None, use_kv_cache=None, precision=None, inference_config=None,
                 backend=None, tensorrt_engine_dir=None):
        self.policy = load_inference_policy(checkpoint, device, inference_config, {
            "num_inference_timesteps": nfe, "use_cuda_graph": use_cuda_graph,
            "use_kv_cache": use_kv_cache, "precision": precision, "backend": backend,
            "tensorrt_engine_dir": tensorrt_engine_dir,
        })
        options = self.policy.inference_options
        self.backend, self.precision = options["backend"], options["precision"]
        self.use_cuda_graph, self.use_kv_cache = options["use_cuda_graph"], options["use_kv_cache"]
        self.device = str(device)
        if self.policy.state_key != "state" or self.policy.action_dim != 14:
            raise ValueError("The RoboTwin joint14 profile requires state_key=state, action_dim=14")
        if self.backend == "tensorrt" and options["tensorrt_batch_size"] != 1:
            raise ValueError("Direct RoboTwin control requires tensorrt_batch_size=1")
        self.rgb_keys = list(self.policy.rgb_keys)
        self.history = ObservationHistory(self.policy.n_obs_steps, self.policy.n_image_obs_steps, self.rgb_keys)
        self.jpeg, self.episode_seed, self.infer_index = jpeg, None, 0
        self.network_calls = 0
        print(f"[100HzS1] backend={self.backend} precision={self.precision} device={self.device} "
              f"flow_type={self.policy.flow_type} nfe={self.policy.num_inference_timesteps} "
              f"RGB/no-flip views={self.rgb_keys} cuda_graph={self.use_cuda_graph} "
              f"graph_scope={self.graph_stats['graph_scope']}", flush=True)

    @property
    def graph_stats(self):
        return self.policy.graph_stats

    def set_episode_seed(self, seed):
        self.episode_seed, self.infer_index = int(seed), 0

    def reset(self):
        self.history.clear()
        self.infer_index = 0
        self.policy.reset()

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
            observations[key] = np.ascontiguousarray(value)[None]
        seed = (self.episode_seed * 1_000_003 + self.infer_index) % (2**31)
        action = self.policy.predict_action(observations, seed=seed)["action"][0]
        actual = self.policy.last_nfe
        if actual != self.policy.num_inference_timesteps:
            raise RuntimeError(f"Expected {self.policy.num_inference_timesteps} steps, got {actual}")
        if self.use_cuda_graph and not self.graph_stats["graph_replays"]:
            raise RuntimeError("CUDA Graph was requested but did not replay")
        if action.shape != (self.policy.n_action_steps, 14) or not np.isfinite(action).all():
            raise ValueError(f"Invalid joint14 actions: {action.shape}")
        self.network_calls += actual
        self.infer_index += 1
        if self.infer_index == 1 or self.infer_index % 25 == 0:
            print(f"[100HzS1] seed={self.episode_seed} infer={self.infer_index} "
                  f"backend={self.backend} nfe={actual} action_steps={len(action)} "
                  f"cuda_graph={self.use_cuda_graph}", flush=True)
        return action

    def close(self):
        self.policy.close()


def get_model(usr_args):
    return DirectPolicy(
        usr_args["ckpt_setting"],
        device=usr_args.get("device", "cuda:0"),
        nfe=usr_args.get("nfe"),
        use_cuda_graph=usr_args.get("use_cuda_graph"),
        use_kv_cache=usr_args.get("use_kv_cache"),
        precision=usr_args.get("precision"),
        inference_config=usr_args.get("inference_config"),
        backend=usr_args.get("backend"),
        tensorrt_engine_dir=usr_args.get("tensorrt_engine_dir"),
        jpeg=os.environ.get("S1_JPEG_ROUNDTRIP", "1") == "1",
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
