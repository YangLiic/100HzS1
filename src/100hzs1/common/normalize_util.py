from .normalizer import SingleFieldLinearNormalizer
from .pytorch_util import dict_apply, dict_apply_reduce, dict_apply_split
import numpy as np


def detect_static_dims_from_episode_ranges(
        episode_ranges, ratio_threshold=100.0, zero_eps=1e-12):
    """Find dimensions that are relatively static in every training episode.

    Within each episode, a dimension is a static candidate when the largest
    dimension range in that episode is at least ``ratio_threshold`` times its
    range.  The final mask is the intersection of all per-episode candidates,
    so a dimension is excluded only when every selected training episode
    independently supports that decision.

    Args:
        episode_ranges: Array shaped ``(num_episodes, num_dims)``.
        ratio_threshold: Required ratio between the episode's largest range
            and a candidate static dimension's range.
            ``None`` or a non-positive value disables detection.
        zero_eps: Denominator floor used only while computing ratios.

    Returns:
        A dictionary containing the final ``mask``, per-dimension support
        counts and the worst (minimum) ratio seen across episodes.
    """
    episode_ranges = np.asarray(episode_ranges, dtype=np.float64)
    if episode_ranges.ndim != 2 or episode_ranges.shape[0] == 0:
        raise ValueError(
            "episode_ranges must have shape (num_episodes, num_dims) "
            f"with at least one episode, got {episode_ranges.shape}"
        )
    if np.any(~np.isfinite(episode_ranges)) or np.any(episode_ranges < 0):
        raise ValueError(f"Invalid per-episode ranges: {episode_ranges}")

    num_episodes, num_dims = episode_ranges.shape
    mask = np.zeros(num_dims, dtype=bool)
    result = {
        'mask': mask,
        'episode_ranges': episode_ranges.astype(np.float32),
        'support_counts': np.zeros(num_dims, dtype=np.int64),
        'min_range_ratios': np.zeros(num_dims, dtype=np.float32),
        'num_episodes': num_episodes,
    }
    if num_dims == 0 or ratio_threshold is None or ratio_threshold <= 0:
        return result

    episode_max = episode_ranges.max(axis=1, keepdims=True)
    range_ratios = episode_max / np.maximum(episode_ranges, zero_eps)
    per_episode_mask = range_ratios >= float(ratio_threshold)

    # A completely frozen episode provides no contrary evidence about which
    # dimensions move, so it must not clear candidates supported by all other
    # episodes.
    frozen_episodes = episode_max[:, 0] <= zero_eps
    per_episode_mask[frozen_episodes] = True
    range_ratios[frozen_episodes] = np.inf

    result.update({
        'mask': per_episode_mask.all(axis=0),
        'support_counts': per_episode_mask.sum(axis=0).astype(np.int64),
        'min_range_ratios': range_ratios.min(axis=0).astype(np.float32),
    })
    return result


def detect_static_dims_from_range(stat, ratio_threshold=100.0, zero_eps=1e-12):
    """Compatibility wrapper for applying the episode rule to one range row."""
    input_min = np.asarray(stat['min'], dtype=np.float64).reshape(-1)
    input_max = np.asarray(stat['max'], dtype=np.float64).reshape(-1)
    if input_min.shape != input_max.shape:
        raise ValueError(
            f"min/max shape mismatch: {input_min.shape} vs {input_max.shape}"
        )
    return detect_static_dims_from_episode_ranges(
        (input_max - input_min)[None],
        ratio_threshold=ratio_threshold,
        zero_eps=zero_eps,
    )

def get_range_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    # -1, 1 normalization
    input_max = stat['max']
    input_min = stat['min']
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_image_range_normalizer():
    scale = np.array([2], dtype=np.float32)
    offset = np.array([-1], dtype=np.float32)
    stat = {
        'min': np.array([0], dtype=np.float32),
        'max': np.array([1], dtype=np.float32),
        'mean': np.array([0.5], dtype=np.float32),
        'std': np.array([np.sqrt(1/12)], dtype=np.float32)
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_identity_normalizer_from_stat(stat):
    scale = np.ones_like(stat['min'])
    offset = np.zeros_like(stat['min'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def robomimic_abs_action_normalizer_from_stat(stat, rotation_transformer):
    result = dict_apply_split(
        stat, lambda x: {
            'pos': x[...,:3],
            'rot': x[...,3:6],
            'gripper': x[...,6:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    def get_rot_param_info(stat):
        example = rotation_transformer.forward(stat['mean'])
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info
    
    def get_gripper_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos_param, pos_info = get_pos_param_info(result['pos'])
    rot_param, rot_info = get_rot_param_info(result['rot'])
    gripper_param, gripper_info = get_gripper_param_info(result['gripper'])

    param = dict_apply_reduce(
        [pos_param, rot_param, gripper_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos_info, rot_info, gripper_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def robomimic_abs_action_only_normalizer_from_stat(stat):
    result = dict_apply_split(
        stat, lambda x: {
            'pos': x[...,:3],
            'other': x[...,3:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    
    def get_other_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos_param, pos_info = get_pos_param_info(result['pos'])
    other_param, other_info = get_other_param_info(result['other'])

    param = dict_apply_reduce(
        [pos_param, other_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos_info, other_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat):
    Da = stat['max'].shape[-1]
    Dah = Da // 2
    result = dict_apply_split(
        stat, lambda x: {
            'pos0': x[...,:3],
            'other0': x[...,3:Dah],
            'pos1': x[...,Dah:Dah+3],
            'other1': x[...,Dah+3:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    
    def get_other_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos0_param, pos0_info = get_pos_param_info(result['pos0'])
    pos1_param, pos1_info = get_pos_param_info(result['pos1'])
    other0_param, other0_info = get_other_param_info(result['other0'])
    other1_param, other1_info = get_other_param_info(result['other1'])

    param = dict_apply_reduce(
        [pos0_param, other0_param, pos1_param, other1_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos0_info, other0_info, pos1_info, other1_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def array_to_stats(arr: np.ndarray):
    stat = {
        'min': np.min(arr, axis=0),
        'max': np.max(arr, axis=0),
        'mean': np.mean(arr, axis=0),
        'std': np.std(arr, axis=0)
    }
    return stat
