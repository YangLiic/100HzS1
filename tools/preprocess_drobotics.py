"""Convert JSON teleoperation recordings directly to RTPolicy LeRobot format.

The conversion logic is based on json_to_lerobot.py, but the output field
layout follows json_to_lerobot.py:

  - agent_pos: camera-time arm joint state from ``info.states``
  - action:    an aligned copy of agent_pos
  - images:    leaf camera names, e.g. ``head_camera`` instead of
               ``observation.images.head_camera``
  - endpose:   optional end-effector state, unless dropped

Resize, fps downsampling, video/image writing, camera renames, and metadata
generation are kept compatible with json_to_lerobot.py.
Usage:
    python tools/json_to_lerobot.py \
        --raw-dir tools/episode_0000 \
        --repo-id D-Robotics/pick_pen_into_holder \
        --image-size 640 480 \
        --target-fps 10
"""

import inspect
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Set, Tuple

import cv2
import tyro
import numpy as np
from tqdm import tqdm
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ModuleNotFoundError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

try:
    from drdc_util.dataset import discover_episodes as _drdc_discover_episodes
    from drdc_util.dataset import load_episode as _drdc_load_episode
except ImportError:
    _drdc_discover_episodes = None
    _drdc_load_episode = None


_DEFAULT_OUTPUT_DIR = Path.cwd() / 'datasets' / 'lerobot'
_DEFAULT_STATE_OUTPUT_KEY = "agent_pos"

_INTERPOLATION_MAP = {
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4,
}


def discover_episodes(raw_dir: Path) -> List[Path]:
    """Discover JSON episode directories.

    Accepts either a single episode directory containing data.json or a parent
    directory containing episode subdirectories. If the shared drdc_util package
    is available, its discovery result is used after the single-episode fast path.
    """
    raw_dir = Path(raw_dir).expanduser()
    if (raw_dir / "data.json").is_file():
        return [raw_dir]

    if _drdc_discover_episodes is not None:
        return [Path(p) for p in _drdc_discover_episodes(raw_dir)]

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    episode_dirs = [
        path
        for path in sorted(raw_dir.iterdir())
        if path.is_dir() and (path / "data.json").is_file()
    ]
    if episode_dirs:
        return episode_dirs

    return sorted({path.parent for path in raw_dir.rglob("data.json")})


def load_episode(episode_dir: Path) -> dict:
    """Load a JSON episode directory."""
    episode_dir = Path(episode_dir)
    if _drdc_load_episode is not None:
        return _drdc_load_episode(episode_dir)

    data_path = episode_dir / "data.json"
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing episode data.json: {data_path}")
    with data_path.open("r") as f:
        return json.load(f)


def output_key_for(source_key: str) -> str:
    """Match json_to_lerobot.py leaf-name field behavior."""
    return source_key.rsplit(".", 1)[-1]


def parse_drop_fields(values: Tuple[str, ...]) -> Set[str]:
    fields: Set[str] = set()
    for value in values or ():
        for field in value.replace(",", " ").split():
            if field:
                fields.add(field)
    return fields


# ============================================================================
# Schema derivation from data.json
# ============================================================================

def _split_arm_ee(keys: list) -> Tuple[List[str], List[str]]:
    """Partition state keys into (arm_keys, ee_keys).

    Keys ending with '_ee' are classified as end-effector.
    Order is preserved as written by episode_writer (insertion order).
    """
    arm_keys, ee_keys = [], []
    for k in keys:
        (ee_keys if k.endswith('_ee') else arm_keys).append(k)
    return arm_keys, ee_keys


def _flatten_names(meta: dict, keys: List[str]) -> List[str]:
    names: List[str] = []
    for k in keys:
        names.extend(meta.get(k, {}).get('keys', []))
    return names


def _endpose_rot6d_names(ee_keys: List[str]) -> List[str]:
    """Auto-generate rot6d names per ee section: [x,y,z,r1x,r1y,r1z,r2x,r2y,r2z].

    Gripper is appended only when present in the source ee keys (state ee has 7D
    quat — no gripper; action ee has 8D — last is gripper).
    """
    names: List[str] = []
    for k in ee_keys:
        for axis in ('x', 'y', 'z',
                     'r1x', 'r1y', 'r1z',
                     'r2x', 'r2y', 'r2z'):
            names.append(f'{k}_{axis}')
    return names


def _resolve_camera_map(
        sample_episode: dict,
        info_video_map: Dict[str, str],
        rename_overrides: Dict[str, str],
        drop_fields: Set[str]) -> Dict[str, str]:
    """Build {original_camera_name: dataset_key} for every camera present in
    ``data[*].colors`` of the first episode.

    Priority: rename_overrides > info_video_map > default leaf camera name.
    Any dotted mapped name is stripped to the final component to match
    json_to_lerobot.py output fields.
    """
    seen = []
    for step in sample_episode.get('data', []):
        for cam in (step.get('colors') or {}).keys():
            if cam not in seen:
                seen.append(cam)
    mapping = {}
    for cam in seen:
        if cam in rename_overrides:
            dataset_key = output_key_for(rename_overrides[cam])
        elif cam in info_video_map:
            dataset_key = output_key_for(info_video_map[cam])
        else:
            dataset_key = cam
        if cam in drop_fields or dataset_key in drop_fields:
            continue
        mapping[cam] = dataset_key
    return mapping


def derive_config(
        sample_episode: dict,
        *,
        target_fps: Optional[int],
        image_size: Optional[Tuple[int, int]],
        use_videos: bool,
        resize_interpolation: str,
        tolerance_s: float,
        rename: Dict[str, str],
        robot_type_override: Optional[str],
        source_fps_override: Optional[int],
        state_output_key: str,
        drop_fields: Set[str]) -> dict:
    """Derive the conversion config dict directly from a data.json sample.

    Returns a dict with the keys consumed by create_empty_dataset / populate_dataset.

    Output follows preprocess_lerobot_vodpp.py / RTPolicy convention:
      - ``state_output_key``: arm joint angles + gripper (no ee).
      - ``endpose``: end-effector poses as a top-level key (UMI style).
      - ``action``: aligned copy of ``state_output_key``.
      - ``meta/modality.json``: slice map for GR00T N1.7 compatibility.
    """
    info = sample_episode.get('info', {}) or {}
    states_meta = info.get('states', {}) or {}

    if not state_output_key:
        raise ValueError("state_output_key must be non-empty")
    if state_output_key in {"action", "timestamp", "frame_index",
                            "episode_index", "index", "task_index"}:
        raise ValueError(f"Invalid state_output_key: {state_output_key!r}")
    if state_output_key in drop_fields:
        raise ValueError(f"Cannot drop required state field: {state_output_key}")
    if "action" in drop_fields:
        raise ValueError("Cannot drop required output field: action")

    # Preserve insertion order from data.json for all state sub-keys.
    all_state_keys = list(states_meta.keys())  # e.g. [left_arm, left_ee, right_arm, right_ee]

    # Split state keys: arm joints -> state_output_key, ee -> endpose.
    arm_state_keys, ee_state_keys = _split_arm_ee(all_state_keys)
    if "endpose" in drop_fields:
        ee_state_keys = []

    # Build low-dimensional state name list (arm joints only).
    obs_state_names: List[str] = []
    obs_state_modality: Dict[str, dict] = {}  # for modality.json
    offset = 0
    for k in arm_state_keys:
        keys = states_meta[k].get('keys', [])
        prefixed = [f"{k}.{name}" for name in keys]
        obs_state_modality[k] = {"start": offset, "end": offset + len(prefixed)}
        obs_state_names.extend(prefixed)
        offset += len(prefixed)

    # Build endpose name list (end-effector only)
    endpose_names: List[str] = []
    endpose_modality: Dict[str, dict] = {}  # for modality.json
    offset = 0
    for k in ee_state_keys:
        keys = states_meta[k].get('keys', [])
        prefixed = [f"{k}.{name}" for name in keys]
        endpose_modality[k] = {"start": offset, "end": offset + len(prefixed)}
        endpose_names.extend(prefixed)
        offset += len(prefixed)

    # preprocess_lerobot_vodpp.py writes action as a copy of state_key.
    # Keep feature shape and slice metadata identical so action/state stay aligned.
    action_names = list(obs_state_names)
    action_modality = {
        key: dict(value) for key, value in obs_state_modality.items()
    }

    camera_map = _resolve_camera_map(
        sample_episode, info.get('video_map') or {}, rename, drop_fields)
    output_keys = [state_output_key, "action"]
    if ee_state_keys:
        output_keys.append("endpose")
    output_keys.extend(camera_map.values())
    duplicates = sorted({key for key in output_keys if output_keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"Output feature name collision: {duplicates}")

    # Robot type & source fps (with backfill paths for legacy data.json)
    robot_type = robot_type_override or info.get('robot_type') or 'unknown'
    source_fps = source_fps_override or info.get('fps') \
        or info.get('image', {}).get('fps') or 10
    if 'robot_type' not in info and not robot_type_override:
        print(f"[warn] data.json missing 'info.robot_type'; using "
              f"'{robot_type}'. Pass --robot-type to override or re-record.")
    if 'fps' not in info and not source_fps_override:
        print(f"[warn] data.json missing 'info.fps'; using {source_fps}. "
              f"Pass --source-fps to override or re-record.")

    target_fps = target_fps or source_fps
    interp = _INTERPOLATION_MAP.get(resize_interpolation, cv2.INTER_LINEAR)

    return {
        'robot_type': robot_type,
        'source_fps': source_fps,
        'fps': target_fps,
        'image_size': image_size,  # None = auto-detect from first image
        'resize_interpolation': interp,
        'use_videos': use_videos,
        'tolerance_s': tolerance_s,
        'image_writer_processes': 10,
        'image_writer_threads': 5,
        'video_backend': None,
        '_state_output_key': state_output_key,
        '_arm_state_keys': arm_state_keys,
        '_ee_state_keys': ee_state_keys,
        '_obs_state_names': obs_state_names,
        '_endpose_names': endpose_names,
        '_action_names': action_names,
        '_obs_state_modality': obs_state_modality,
        '_endpose_modality': endpose_modality,
        '_action_modality': action_modality,
        '_camera_map': camera_map,
    }


# ============================================================================
# Data extraction
# ============================================================================

def extract_qpos(episode_data: dict, key: str, state_keys: list) -> np.ndarray:
    """Extract and concatenate qpos from named keys under a top-level JSON key.

    Args:
        episode_data: Raw JSON dict from load_episode().
        key: 'states' or 'actions'.
        state_keys: JSON keys, e.g. ['left_arm'] or ['left_arm', 'right_arm'].

    Returns:
        Array of shape (T, D).
    """
    result = []
    for step in episode_data['data']:
        row = np.array([], dtype='float32')
        for sk in state_keys:
            section = step.get(key, {}).get(sk)
            if section is not None and section.get('qpos'):
                qpos = np.array(section['qpos'], dtype='float32')
                row = np.concatenate([row, qpos])
        result.append(row)
    return np.array(result)


def quat_to_rot6d(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [qx, qy, qz, qw] to rotation 6D (first two columns of R)."""
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat(quat).as_matrix()
    return np.array([*R[:, 0], *R[:, 1]], dtype='float32')


def convert_endpose_to_rot6d(endpose: np.ndarray) -> np.ndarray:
    """Convert endpose from [x,y,z, qx,qy,qz,qw, gripper?] (7D or 8D)
    to [x,y,z, r1x,r1y,r1z, r2x,r2y,r2z, gripper?] (9D or 10D).
    """
    pos = endpose[:3]
    quat = endpose[3:7]
    gripper = endpose[7:]
    rot6d = quat_to_rot6d(quat)
    return np.concatenate([pos, rot6d, gripper]).astype('float32')


# ============================================================================
# Image processing
# ============================================================================

def smart_resize(image: np.ndarray, target_size: tuple,
                 interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Resize only if dimensions differ from target."""
    h, w = image.shape[:2]
    tw, th = target_size
    if w == tw and h == th:
        return image
    return cv2.resize(image, target_size, interpolation=interpolation)


def load_episode_images(
    episode_dir: Path,
    episode_data: dict,
    cfg: dict,
) -> Dict[str, List[np.ndarray]]:
    """Load camera images, mapping JSON keys to LeRobot keys via camera_map."""
    camera_map = cfg['_camera_map']
    image_size = cfg.get('image_size')
    interpolation = cfg.get('resize_interpolation', cv2.INTER_LINEAR)

    images: Dict[str, List[np.ndarray]] = defaultdict(list)

    for step in episode_data['data']:
        for json_key, dataset_key in camera_map.items():
            cam_data = step.get('colors', {}).get(json_key)
            if not cam_data:
                continue
            rel_path = cam_data.get('path') if isinstance(cam_data, dict) else cam_data
            if not rel_path:
                continue
            img_path = os.path.join(episode_dir, rel_path)

            bgr = cv2.imread(img_path)
            if bgr is None:
                raise RuntimeError(f"Failed to read image: {img_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if image_size is not None:
                rgb = smart_resize(rgb, tuple(image_size), interpolation)
            images[dataset_key].append(rgb)

    return dict(images)


# ============================================================================
# LeRobot dataset creation
# ============================================================================

def create_empty_dataset(
    repo_id: str,
    output_dir: Path,
    cfg: dict,
    *,
    mode: Literal["video", "image"] | None = None,
    image_shapes: Optional[Dict[str, tuple]] = None,
):
    """Create an empty LeRobotDataset with RTPolicy leaf-name features."""
    if mode is None:
        mode = "video" if cfg.get('use_videos', False) else "image"
    state_key = cfg['_state_output_key']
    obs_state_names = cfg['_obs_state_names']
    endpose_names = cfg['_endpose_names']
    action_names = cfg['_action_names']

    features = {
        state_key: {
            "dtype": "float32",
            "shape": (len(obs_state_names),),
            "names": obs_state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(action_names),),
            "names": action_names,
        },
    }

    # End-effector pose as top-level key (UMI convention, not auto-consumed by policy)
    if endpose_names:
        features["endpose"] = {
            "dtype": "float32",
            "shape": (len(endpose_names),),
            "names": endpose_names,
        }

    # Camera features — image_shapes from auto-detection wins; fallback to cfg.image_size.
    fallback_size = cfg.get('image_size') or [640, 480]
    for cam in cfg['_camera_map'].values():
        if image_shapes and cam in image_shapes:
            height, width = image_shapes[cam]
        else:
            width, height = fallback_size
        features[cam] = {
            "dtype": mode,
            "shape": (3, height, width),
            "names": ["channels", "height", "width"],
        }

    dataset_path = output_dir / repo_id
    if dataset_path.exists():
        shutil.rmtree(dataset_path)

    return LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_path,
        fps=cfg.get('fps', 30),
        robot_type=cfg.get('robot_type', 'unknown'),
        features=features,
        use_videos=cfg.get('use_videos', False),
        tolerance_s=cfg.get('tolerance_s', 0.0001),
        image_writer_processes=cfg.get('image_writer_processes', 10),
        image_writer_threads=cfg.get('image_writer_threads', 5),
        video_backend=cfg.get('video_backend'),
    )


def add_frame_compat(dataset, frame: dict, task_name: str) -> None:
    """Support both old and new LeRobotDataset.add_frame signatures."""
    add_frame_params = inspect.signature(dataset.add_frame).parameters
    if "task" in add_frame_params:
        dataset.add_frame(frame, task=task_name)
    else:
        frame_with_task = dict(frame)
        frame_with_task["task"] = task_name
        dataset.add_frame(frame_with_task)


# ============================================================================
# Populate dataset
# ============================================================================

def _build_modality_json(cfg: dict) -> dict:
    """Build GR00T N1.7 modality.json from the derived config."""
    video_modality = {}
    for json_key, dataset_key in cfg['_camera_map'].items():
        video_modality[json_key] = {"original_key": dataset_key}

    modality = {
        "state": cfg['_obs_state_modality'],
        "action": cfg['_action_modality'],
        "video": video_modality,
    }
    if cfg['_endpose_modality']:
        modality["endpose"] = cfg['_endpose_modality']
    return modality


def populate_dataset(dataset, episode_dirs: List[Path], cfg: dict,
                     task_override: str = ''):
    """Fill dataset with RTPolicy state, optional endpose, images, and action=state."""
    state_key = cfg['_state_output_key']
    arm_state_keys = cfg['_arm_state_keys']
    ee_state_keys = cfg['_ee_state_keys']
    source_fps = cfg.get('source_fps', cfg.get('fps', 30))
    target_fps = cfg.get('fps', source_fps)
    stride = max(1, source_fps // target_fps)
    has_endpose = len(ee_state_keys) > 0

    for i, ep_dir in enumerate(tqdm(episode_dirs, desc="Converting episodes")):
        episode_data = load_episode(ep_dir)

        # Arm joint angles -> RTPolicy low-dimensional state.
        obs_state = extract_qpos(episode_data, 'states', arm_state_keys)
        # End-effector poses -> endpose (top-level, UMI style)
        if has_endpose:
            endpose = extract_qpos(episode_data, 'states', ee_state_keys)

        task = task_override or episode_data.get('text', {}).get('goal', '')
        cameras = load_episode_images(ep_dir, episode_data, cfg)

        num_frames = len(obs_state)
        expected_state_dim = len(cfg['_obs_state_names'])
        if obs_state.ndim != 2 or obs_state.shape[1] != expected_state_dim:
            raise ValueError(
                f"{ep_dir} state shape {obs_state.shape} does not match "
                f"schema dim {expected_state_dim}"
            )
        if has_endpose and len(endpose) != num_frames:
            raise ValueError(
                f"{ep_dir} endpose length {len(endpose)} does not match "
                f"state length {num_frames}"
            )
        for cam_name, img_list in cameras.items():
            if len(img_list) != num_frames:
                raise ValueError(
                    f"{ep_dir} camera {cam_name!r} has {len(img_list)} images, "
                    f"expected {num_frames}"
                )
        sampled = list(range(0, num_frames, stride))
        print(f"Episode {i}: {num_frames} -> {len(sampled)} "
              f"({source_fps}Hz -> {target_fps}Hz, stride={stride})")

        for j in sampled:
            state = obs_state[j]
            frame = {
                state_key: state,
                "action": state.copy(),
            }
            if has_endpose:
                frame["endpose"] = endpose[j]

            for cam_name, img_list in cameras.items():
                frame[cam_name] = img_list[j]

            add_frame_compat(dataset, frame, task)

        dataset.save_episode()

    return dataset


# ============================================================================
# Main entry point
# ============================================================================

def _parse_renames(rename: Tuple[str, ...]) -> Dict[str, str]:
    out = {}
    for entry in rename or ():
        if '=' not in entry:
            raise ValueError(
                f"--rename expects 'original=dataset_key', got: {entry!r}")
        k, v = entry.split('=', 1)
        out[k.strip()] = v.strip()
    return out


def json_to_lerobot_vodpp(
    raw_dir: Path,
    repo_id: str,
    *,
    output_dir: Path = _DEFAULT_OUTPUT_DIR,
    target_fps: int = 10,
    image_size: Optional[Tuple[int, int]] = None,
    use_videos: bool = False,
    resize_interpolation: str = 'INTER_LINEAR',
    tolerance_s: float = 0.0001,
    rename: Tuple[str, ...] = (),
    push_to_hub: bool = False,
    task: str = '',
    robot_type: Optional[str] = None,
    source_fps: Optional[int] = None,
    state_output_key: str = _DEFAULT_STATE_OUTPUT_KEY,
    drop_fields: Tuple[str, ...] = (),
):
    """Convert JSON episodes to RTPolicy LeRobot dataset format.

    Args:
        raw_dir: Directory containing episode data.
        repo_id: HuggingFace-style repo ID.
        output_dir: Root directory for output datasets.
        target_fps: Target FPS for downsampling (default: 10Hz).
        image_size: (width, height) to resize images. Defaults to native resolution.
        use_videos: Encode camera streams as video instead of frames.
        resize_interpolation: OpenCV interpolation flag name.
        tolerance_s: LeRobot timestamp sync tolerance.
        rename: Camera renames as ``original=dataset_key`` (overrides info.video_map).
        push_to_hub: Upload to HuggingFace Hub after conversion.
        task: Fixed task description (overrides text.goal in data.json).
        robot_type: Backfill ``info.robot_type`` for legacy data.json.
        source_fps: Backfill ``info.fps`` for legacy data.json.
        state_output_key: Output field for low-dimensional state. Defaults to
            ``agent_pos`` for current RTPolicy configs. Use ``state`` for exact
            leaf-name output from ``observation.state``.
        drop_fields: Optional output fields to omit, e.g. ``endpose`` or a
            camera field. ``action`` and ``state_output_key`` are required.
    """
    rename_map = _parse_renames(rename)
    drop_field_set = parse_drop_fields(drop_fields)

    episode_dirs = discover_episodes(raw_dir)
    failed = [d for d in episode_dirs if d.name.endswith('_failed')]
    episode_dirs = [d for d in episode_dirs if not d.name.endswith('_failed')]
    if failed:
        print(f"Filtered out {len(failed)} failed episodes: "
              f"{[d.name for d in failed]}")
    if not episode_dirs:
        raise FileNotFoundError(f"No episodes found under {raw_dir}")
    print(f"Found {len(episode_dirs)} episodes")

    # Derive cfg from the first episode's data.json
    first_data = load_episode(episode_dirs[0])
    cfg = derive_config(
        first_data,
        target_fps=target_fps,
        image_size=image_size,
        use_videos=use_videos,
        resize_interpolation=resize_interpolation,
        tolerance_s=tolerance_s,
        rename=rename_map,
        robot_type_override=robot_type,
        source_fps_override=source_fps,
        state_output_key=state_output_key,
        drop_fields=drop_field_set,
    )
    print(f"Robot type: {cfg['robot_type']}")
    print(f"  {cfg['_state_output_key']}: {cfg['_arm_state_keys']} "
          f"(dim={len(cfg['_obs_state_names'])})")
    if cfg['_ee_state_keys']:
        print(f"  endpose:          {cfg['_ee_state_keys']} "
              f"(dim={len(cfg['_endpose_names'])})")
    print(f"  action:           aligned copy of {cfg['_state_output_key']} "
          f"(dim={len(cfg['_action_names'])})")
    print(f"  cameras:          {cfg['_camera_map']}")
    print(f"  fps: {cfg['source_fps']} -> {cfg['fps']}")

    # Detect image shapes from first episode
    first_images = load_episode_images(episode_dirs[0], first_data, cfg)
    image_shapes = {}
    for cam_name, img_list in first_images.items():
        if img_list:
            h, w = img_list[0].shape[:2]
            image_shapes[cam_name] = (h, w)
            print(f"  {cam_name}: {w}x{h}")

    # Create and populate
    print(f"Output: {output_dir / repo_id}")
    dataset = create_empty_dataset(
        repo_id, output_dir, cfg,
        image_shapes=image_shapes,
    )
    task_name = task or repo_id.rsplit('/', 1)[-1]
    dataset = populate_dataset(dataset, episode_dirs, cfg,
                               task_override=task_name)

    # Write modality.json for GR00T N1.7 compatibility
    modality = _build_modality_json(cfg)
    modality_path = output_dir / repo_id / 'meta' / 'modality.json'
    modality_path.parent.mkdir(parents=True, exist_ok=True)
    with open(modality_path, 'w') as f:
        json.dump(modality, f, indent=2)
    print(f"  Wrote {modality_path}")

    if push_to_hub:
        dataset.push_to_hub(upload_large_folder=True)

    print(f"Done. Dataset saved to {output_dir / repo_id}")


def main():
    tyro.cli(json_to_lerobot_vodpp)


if __name__ == '__main__':
    main()
