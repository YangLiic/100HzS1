"""Backend-independent observation, noise and output contract; no Torch import."""
from threading import RLock

import numpy as np


def numpy(value):
    # Accept existing Torch observations without requiring Torch in TRT workers.
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class PolicyRuntime:
    def _initialize(self, metadata, options, device):
        self.metadata = dict(metadata)
        self.inference_options = dict(options)
        self.device = str(device)
        for key in ["horizon", "action_dim", "n_action_steps", "n_obs_steps",
                    "n_image_obs_steps", "state_key", "supervision_key", "flow_type"]:
            setattr(self, key, metadata[key])
        self.rgb_keys = list(metadata["rgb_keys"])
        self.num_inference_timesteps = options["num_inference_timesteps"]
        self.backend = options["backend"]
        self.precision = options["precision"]
        self.network_calls = self.last_nfe = 0
        self._rng = np.random.default_rng(0)
        self._lock = RLock()

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
        self.last_nfe = 0

    def _inputs(self, observations):
        batch = None
        result = {}
        for key in self.rgb_keys:
            x = numpy(observations[key])
            shape = tuple(self.metadata["raw_image_shapes"][key])
            # Canonical form [B,T,C,H,W]; also accept uint8 [B,T,H,W,C].
            if x.ndim == 5 and tuple(x.shape[-3:]) != shape and tuple(x.shape[-3:]) == (shape[1], shape[2], 3):
                x = np.moveaxis(x, -1, -3)
            if x.ndim != 5 or tuple(x.shape[-3:]) != shape or x.shape[1] < self.n_image_obs_steps:
                raise ValueError(f"{key}: expected [B,T,{shape}], T>={self.n_image_obs_steps}; got {x.shape}")
            if batch is None:
                batch = x.shape[0]
            if x.shape[0] != batch or batch < 1:
                raise ValueError("All observations must share a positive batch size")
            is_byte = x.dtype == np.uint8
            x = np.ascontiguousarray(x[:, -self.n_image_obs_steps:], dtype=np.float32)
            if is_byte:
                x = x / np.float32(255)
            if not np.isfinite(x).all() or x.min() < 0 or x.max() > 1:
                raise ValueError(f"{key}: expected finite RGB in [0,1] or uint8")
            result[key] = x
        key = self.state_key
        state = numpy(observations[key])
        if state.ndim != 3 or state.shape[0] != batch or state.shape[1] < self.n_obs_steps or state.shape[2] != self.action_dim:
            raise ValueError(f"{key}: expected [B,T,{self.action_dim}], T>={self.n_obs_steps}; got {state.shape}")
        result[key] = np.ascontiguousarray(state[:, :self.n_obs_steps], dtype=np.float32)
        if not np.isfinite(result[key]).all():
            raise ValueError("State contains nonfinite values")
        if "delta" in self.state_key:
            base_key = self.state_key.replace("delta_", "")
            base = numpy(observations[base_key])
            if base.ndim != 3 or base.shape[0] != batch or base.shape[1] < 1 or base.shape[2] != self.action_dim:
                raise ValueError(f"{base_key}: expected absolute [B,T,{self.action_dim}] observations")
            result[base_key] = np.ascontiguousarray(base[:, :1], dtype=np.float32)
            if not np.isfinite(result[base_key]).all():
                raise ValueError("Absolute state contains nonfinite values")
        return result, batch

    def _noise(self, batch, seed, noise):
        shape = (batch, self.horizon, self.action_dim)
        if noise is None:
            rng = np.random.default_rng(int(seed)) if seed is not None else self._rng
            noise = rng.standard_normal(shape).astype(np.float32)
        else:
            noise = np.ascontiguousarray(numpy(noise), dtype=np.float32)
        if noise.shape != shape or not np.isfinite(noise).all():
            raise ValueError(f"Expected finite noise_trajectory {shape}, got {noise.shape}")
        return noise

    def _validate_output(self, result, batch):
        shape = (batch, self.n_action_steps, self.action_dim)
        if result["action"].shape != shape:
            raise RuntimeError(f"Backend returned action {result['action'].shape}; expected {shape}")
        for key in ["action", "predicted_raw"]:
            if not np.isfinite(result[key]).all():
                raise RuntimeError(f"{self.backend} returned nonfinite {key}")
        return result

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
