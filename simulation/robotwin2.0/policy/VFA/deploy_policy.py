import os
import numpy as np
import cv2
from vfa.models.policy.vfa.env_runner.base_runner import VFARunner
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

# --- Train/eval imaging-pipeline match (ambiguous first-grasp bias fix) ---
# Training frames went through cv2.imencode(".jpg") (default quality=95) at
# collection time (simulation/robotwin2.0/envs/utils/pkl2hdf5.py: images_encoding),
# so the policy was trained on JPEG(q95)-degraded images, while eval feeds raw
# renders. At visually ambiguous states this sub-gray-level gap deterministically
# flips the policy's first-grasp mode (docs/ambiguous_left_bias_rootcause.md).
# Re-apply the identical roundtrip here so eval inputs match the training
# distribution. Disable with VFA_JPEG_ROUNDTRIP=0.
_JPEG_ROUNDTRIP = os.environ.get("VFA_JPEG_ROUNDTRIP", "1") == "1"


def _jpeg_roundtrip(rgb):
    # Mirrors collection exactly: encode the raw RGB array as-is (no BGR
    # conversion), default quality=95, decode with IMREAD_COLOR.
    enc = cv2.imencode(".jpg", rgb)[1]
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def encode_obs(observation):
    obs_raw = observation["observation"]
    if _JPEG_ROUNDTRIP:
        head_rgb = _jpeg_roundtrip(obs_raw["head_camera"]["rgb"])
        left_rgb = _jpeg_roundtrip(obs_raw["left_camera"]["rgb"])
        right_rgb = _jpeg_roundtrip(obs_raw["right_camera"]["rgb"])
        front_rgb = _jpeg_roundtrip(obs_raw["front_camera"]["rgb"])
    else:
        head_rgb = obs_raw["head_camera"]["rgb"]
        left_rgb = obs_raw["left_camera"]["rgb"]
        right_rgb = obs_raw["right_camera"]["rgb"]
        front_rgb = obs_raw["front_camera"]["rgb"]
    head_cam = (np.moveaxis(head_rgb, -1, 0) / 255)
    left_cam = (np.moveaxis(left_rgb, -1, 0) / 255)
    right_cam = (np.moveaxis(right_rgb, -1, 0) / 255)
    front_cam = (np.moveaxis(front_rgb, -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
        front_cam=front_cam,
    )
    obs["state"] = observation["joint_action"]["vector"]

    endpose = observation["endpose"]
    obs["endpose"] = np.concatenate([
        endpose["left_endpose"], 
        [endpose["left_gripper"]], 
        endpose["right_endpose"], 
        [endpose["right_gripper"]]
    ])
    
    return obs


def get_model(usr_args):
    print(f"[VFA] jpeg_roundtrip={'ON' if _JPEG_ROUNDTRIP else 'OFF'} "
          f"(VFA_JPEG_ROUNDTRIP env; matches pkl2hdf5 cv2.imencode q95)")
    ckpt_setting = usr_args["ckpt_setting"]
    # Support passing a full checkpoint path directly from CLI.
    if (
        os.path.isabs(ckpt_setting)
        or ckpt_setting.startswith("./")
        or ckpt_setting.startswith("../")
    ):
        ckpt_file = os.path.abspath(os.path.expanduser(ckpt_setting))
    else:
        ckpt_file = f"{ROOT_DIR}/checkpoints/vfa/{usr_args['task_name']}_{usr_args['expert_data_num']}/{ckpt_setting}"
    if usr_args['checkpoint_num'] is not None:
        ckpt_file = f"{ckpt_file}/checkpoint-{usr_args['checkpoint_num']}"
    kwargs = {}
    if 'device' in usr_args:
        kwargs['device'] = f"cuda:{usr_args['device']}"
    runner = VFARunner(checkpoint_dir=ckpt_file, **kwargs)
    _nfe = os.environ.get("VFA_NFE")
    if _nfe:
        runner.policy.num_inference_timesteps = int(_nfe)
        print(f"[VFA] NFE override: {runner.policy.num_inference_timesteps}")
    return runner


def eval(TASK_ENV, model, observation):
    """
    TASK_ENV: Task Environment Class, you can use this class to interact with the environment
    model: The model from 'get_model()' function
    observation: The observation about the environment
    """
    obs = encode_obs(observation)
    instruction = TASK_ENV.get_instruction()
    # if all([(model.obs[-1][k] == obs[k]).all() for k in model.obs[-1]]):
    #     print("Warning: Redundant Observation Detected!")

    # ======== Get Action ========
    actions, action_type = model.get_action(obs)

    for action in actions:
        kwargs = {'action_type': action_type} if action_type else {}
        TASK_ENV.take_action(action, **kwargs)
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)

def reset_model(model):
    model.reset_obs()
