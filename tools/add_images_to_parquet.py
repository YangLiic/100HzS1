#!/usr/bin/env python3
"""Embed video frames as PNG bytes into LeRobot parquet files.

Reads existing parquet episodes, decodes matching camera videos, and adds
head_cam/left_cam/right_cam image columns without removing any existing data.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

CAMERA_KEYS = ("head_cam", "left_cam", "right_cam")
IMAGE_STRUCT = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])


def decode_video_ffmpeg(path: Path) -> list[np.ndarray]:
    probe = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        text=True,
    )
    info = json.loads(probe)["streams"][0]
    width, height = int(info["width"]), int(info["height"])
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    frames: list[np.ndarray] = []
    frame_size = width * height * 3
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        frames.append(np.frombuffer(raw, np.uint8).reshape(height, width, 3))
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path} (exit {proc.returncode})")
    return frames


def frame_to_png_bytes(frame: np.ndarray) -> bytes:
    # OpenCV expects BGR; ffmpeg gives RGB.
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode PNG frame")
    return encoded.tobytes()


def build_image_column(frames: list[np.ndarray]) -> pa.Array:
    values = [
        {"bytes": frame_to_png_bytes(frame), "path": f"frame_{idx:06d}.png"}
        for idx, frame in enumerate(frames)
    ]
    return pa.array(values, type=IMAGE_STRUCT)


def process_episode(dataset_root: Path, episode_name: str, chunk: str = "chunk-000") -> None:
    parquet_path = dataset_root / "data" / chunk / f"{episode_name}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    table = pq.read_table(parquet_path)
    if all(key in table.column_names for key in CAMERA_KEYS):
        return

    num_rows = len(table)
    camera_arrays: dict[str, pa.Array] = {}
    for cam_key in CAMERA_KEYS:
        video_path = (
            dataset_root / "videos" / chunk / cam_key / f"{episode_name}.mp4"
        )
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        frames = decode_video_ffmpeg(video_path)
        if len(frames) != num_rows:
            raise ValueError(
                f"{episode_name}/{cam_key}: video has {len(frames)} frames, "
                f"parquet has {num_rows} rows"
            )
        camera_arrays[cam_key] = build_image_column(frames)

    columns = []
    names = []
    for name in table.column_names:
        columns.append(table.column(name))
        names.append(name)
        if name == "endpose":
            for cam_key in CAMERA_KEYS:
                columns.append(camera_arrays[cam_key])
                names.append(cam_key)

    if "endpose" not in table.column_names:
        for cam_key in CAMERA_KEYS:
            columns.append(camera_arrays[cam_key])
            names.append(cam_key)

    new_table = pa.table(dict(zip(names, columns)))
    tmp_path = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(new_table, tmp_path)
    os.replace(tmp_path, parquet_path)


def _process_episode_worker(args: tuple[str, str, str]) -> str:
    dataset_root, chunk, episode_name = args
    process_episode(Path(dataset_root), episode_name, chunk)
    return episode_name


def _image_hwc_from_feature(feat: dict) -> list[int]:
    shape = feat.get("shape") or []
    if len(shape) == 3:
        a, b, c = (int(x) for x in shape)
        if a in (1, 3) and b >= 16 and c >= 16:
            return [b, c, a]
        if c in (1, 3) and a >= 16 and b >= 16:
            return [a, b, c]
    info = feat.get("info") or {}
    return [
        int(info.get("video.height") or 480),
        int(info.get("video.width") or 640),
        3,
    ]


def update_info_json(dataset_root: Path) -> None:
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)

    for cam_key in CAMERA_KEYS:
        feat = info.get("features", {}).get(cam_key, {})
        info["features"][cam_key] = {
            "dtype": "image",
            "shape": _image_hwc_from_feature(feat),
            "names": ["h", "w", "c"],
        }

    with info_path.open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="LeRobot dataset directory containing data/ and meta/",
    )
    parser.add_argument("--chunk", default="chunk-000")
    parser.add_argument("--episode", default=None, help="Process a single episode, e.g. episode_000000")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--skip-info-update", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    data_dir = args.dataset_root / "data" / args.chunk
    if args.episode:
        episodes = [args.episode]
    else:
        episodes = sorted(p.stem for p in data_dir.glob("episode_*.parquet"))
        episodes = episodes[args.start : args.end]

    pending = []
    for episode_name in episodes:
        parquet_path = data_dir / f"{episode_name}.parquet"
        existing_cols = pq.read_schema(parquet_path).names
        if all(key in existing_cols for key in CAMERA_KEYS):
            continue
        pending.append(episode_name)

    if not pending:
        print("All selected episodes already contain image columns.")
    elif args.workers <= 1 or len(pending) == 1:
        for episode_name in tqdm(pending, desc="episodes"):
            process_episode(args.dataset_root, episode_name, args.chunk)
    else:
        worker_args = [
            (str(args.dataset_root), args.chunk, episode_name) for episode_name in pending
        ]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_process_episode_worker, item) for item in worker_args]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="episodes"):
                _.result()

    if not args.skip_info_update and not args.episode:
        update_info_json(args.dataset_root)


if __name__ == "__main__":
    main()
