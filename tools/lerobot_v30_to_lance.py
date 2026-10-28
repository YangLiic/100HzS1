#!/usr/bin/env python3
"""Convert local LeRobot v3 parquet/images/MP4 data in the RTPolicy env.

--mode jpeg writes numeric columns + JPEG bytes to one Lance frame table.
--mode video writes numeric frames + original MP4 files in Lance blob columns.
Neither mode imports LeRobot or lerobot_lancedb. Writes are staged locally so
destinations that cannot perform Lance's atomic rename (e.g. EFS) still work.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import logging
from pathlib import Path
import re
import shutil
import sys
import tempfile

import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from PIL import Image

# Also supports invocation from outside the repo without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rtpolicy.datasets.lance_video import VideoDecoderCache


SCALARS = {"episode_index", "frame_index", "index", "timestamp", "task_index", "subtask_index"}
VISUAL_DTYPES = {"image", "video"}
LOG = logging.getLogger("lerobot_v30_to_lance")


def load_source(root):
    info = json.loads((root / "meta/info.json").read_text())
    if info.get("codebase_version") != "v3.0":
        raise ValueError("Expected LeRobot codebase_version=v3.0; upgrade v2 data first")
    if info["fps"] <= 0:
        raise ValueError("Dataset fps must be positive")
    shards = sorted(root.glob("data/**/*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No data parquet shards under {root}")
    episodes = []
    for path in sorted(root.glob("meta/episodes/**/*.parquet")):
        names = pq.read_schema(path).names
        wanted = ["episode_index", "dataset_from_index", "dataset_to_index"]
        wanted += [name for name in names if name.startswith("videos/")]
        episodes.extend(pq.read_table(path, columns=wanted).to_pylist())
    episodes.sort(key=lambda ep: ep["dataset_from_index"])
    if len(episodes) != info["total_episodes"] or not episodes:
        raise ValueError("Episode metadata count differs from info.json or dataset is empty")
    end = 0
    for ep in episodes:
        if ep["dataset_from_index"] != end or ep["dataset_to_index"] <= end:
            raise ValueError("Episode spans must be nonempty, contiguous and start at zero")
        end = ep["dataset_to_index"]
    if end != info["total_frames"]:
        raise ValueError("Episode spans do not cover info.json total_frames")
    for required in ("meta/tasks.parquet", "meta/stats.json"):
        if not (root / required).is_file():
            raise FileNotFoundError(root / required)
    return info, episodes, shards


def video_files(root, info, episodes):
    files = {}
    video_keys = [key for key, ft in info["features"].items() if ft["dtype"] == "video"]
    template = info.get("video_path")
    if video_keys and not template:
        raise ValueError("Video features require video_path in meta/info.json")
    for ep in episodes:
        for key in video_keys:
            prefix = f"videos/{key}/"
            identity = (key, int(ep[prefix + "chunk_index"]), int(ep[prefix + "file_index"]))
            path = root / template.format(video_key=key, chunk_index=identity[1], file_index=identity[2])
            if not path.is_file():
                raise FileNotFoundError(path)
            files[identity] = path
            if float(ep[prefix + "from_timestamp"]) < 0:
                raise ValueError(f"Negative video offset for episode {ep['episode_index']}, {key}")
    return files


def frame_schema(features, mode, jpeg_storage="binary"):
    if mode not in ("video", "jpeg"):
        raise ValueError(f"Unknown conversion mode: {mode}")
    fields = []
    mapped = {}
    for key, ft in features.items():
        name = key.replace(".", "_")
        if name in mapped.values():
            raise ValueError(f"Feature names collide after Lance normalization: {key}")
        mapped[key] = name
        if ft["dtype"] in VISUAL_DTYPES:
            if mode == "jpeg":
                fields.append(
                    pa.field(name, pa.large_binary(), metadata={b"lance-encoding:blob": b"true"})
                    if jpeg_storage == "blob" else pa.field(name, pa.binary())
                )
        elif key in SCALARS:
            fields.append(pa.field(name, pa.from_numpy_dtype(ft["dtype"])))
        else:
            try:
                dtype = pa.from_numpy_dtype(ft["dtype"])
            except (TypeError, pa.ArrowNotImplementedError) as exc:
                raise ValueError(f"Unsupported numeric feature {key}: {ft['dtype']}") from exc
            dim = int(np.prod(ft.get("shape", [1])))
            fields.append(pa.field(name, pa.list_(dtype, dim)))
    missing = (SCALARS - {"subtask_index"}) - set(features)
    if missing:
        raise ValueError(f"Missing frame columns: {sorted(missing)}")
    return pa.schema(fields), mapped


def encode_jpeg(image, quality, subsampling):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    elif isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    else:
        image = Image.open(image)
    with image:
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=quality, subsampling=subsampling)
        return buf.getvalue()


def frame_batches(
    root, info, episodes, shards, schema, mapped, files,
    *, batch_size, num_workers, quality, subsampling, tolerance_s,
):
    features = info["features"]
    ep_by_id = {int(ep["episode_index"]): ep for ep in episodes}
    starts = np.asarray([ep["dataset_from_index"] for ep in episodes])
    ep_ids = np.asarray([ep["episode_index"] for ep in episodes])
    decoder_cache = VideoDecoderCache(capacity=16, tolerance_s=tolerance_s)
    next_index = 0
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            for shard in shards:
                LOG.info("Reading %s", shard.relative_to(root))
                columns = [key for key in features if features[key]["dtype"] != "video"]
                for batch in pq.ParquetFile(shard).iter_batches(batch_size=batch_size, columns=columns):
                    source = pa.Table.from_batches([batch])
                    size = len(source)
                    indices = source["index"].to_numpy()
                    if not np.array_equal(indices, np.arange(next_index, next_index + size)):
                        raise ValueError(f"Frame indices are not contiguous/in order at {shard}")
                    expected_eps = ep_ids[np.searchsorted(starts, indices, side="right") - 1]
                    if not np.array_equal(source["episode_index"].to_numpy(), expected_eps):
                        raise ValueError(f"Frame episode ids disagree with metadata at {shard}")
                    arrays = []
                    for key, name in mapped.items():
                        if name not in schema.names:
                            continue
                        field = schema.field(name)
                        dtype = features[key]["dtype"]
                        if dtype == "video":
                            frames = [None] * size
                            plan = {}
                            frame_times = zip(source["episode_index"].to_pylist(), source["timestamp"].to_pylist())
                            for pos, (ep_id, ts) in enumerate(frame_times):
                                ep = ep_by_id[int(ep_id)]
                                prefix = f"videos/{key}/"
                                identity = (key, int(ep[prefix + "chunk_index"]), int(ep[prefix + "file_index"]))
                                plan.setdefault(identity, []).append((pos, float(ts) + float(ep[prefix + "from_timestamp"])))
                            for identity, queries in plan.items():
                                decoder = decoder_cache.get(identity, lambda: str(files[identity]))
                                for (pos, _), frame in zip(queries, decoder.decode([ts for _, ts in queries])):
                                    frames[pos] = frame
                            values = pool.map(lambda image: encode_jpeg(image, quality, subsampling), frames)
                            arrays.append(pa.array(list(values), type=field.type))
                        elif dtype == "image":
                            def encode(value):
                                if isinstance(value, dict):
                                    raw = value.get("bytes")
                                    if raw is None:
                                        raw = root / value["path"]
                                else:
                                    raw = value
                                return encode_jpeg(raw, quality, subsampling)
                            arrays.append(pa.array(list(pool.map(encode, source[key].to_pylist())), type=field.type))
                        elif key in SCALARS:
                            arrays.append(pc.cast(source[key].combine_chunks(), field.type))
                        else:
                            # Flatten all feature dimensions, as expected by the
                            # existing Lance training reader, preserving dtype.
                            values = np.asarray(source[key].to_pylist(), dtype=features[key]["dtype"])
                            dim = field.type.list_size
                            flat = pa.array(values.reshape(size, dim).reshape(-1), type=field.type.value_type)
                            arrays.append(pa.FixedSizeListArray.from_arrays(flat, dim))
                    yield pa.RecordBatch.from_arrays(arrays, schema=schema)
                    next_index += size
        if next_index != info["total_frames"]:
            raise ValueError(f"Read {next_index} frames, expected {info['total_frames']}")
    finally:
        decoder_cache.close()


def convert_dataset(
    root, output, *, mode="jpeg", repo_id=None, table_name=None,
    batch_size=256, num_workers=4, jpeg_quality=100, jpeg_subsampling=0,
    tolerance_s=1e-4, staging_dir=None, overwrite=False, dry_run=False,
    jpeg_storage="binary",
):
    root, output = Path(root).expanduser().resolve(), Path(output).expanduser().resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Source and output directories must be separate and must not contain one another")
    if mode not in ("video", "jpeg"):
        raise ValueError(f"Unknown conversion mode: {mode}")
    if jpeg_storage not in ("binary", "blob"):
        raise ValueError("jpeg_storage must be binary or blob")
    if mode == "video" and jpeg_storage != "binary":
        raise ValueError("jpeg_storage applies to jpeg mode")
    if batch_size < 1 or num_workers < 1 or tolerance_s <= 0:
        raise ValueError("batch_size, num_workers and tolerance_s must be positive")
    if not 1 <= jpeg_quality <= 100 or jpeg_subsampling not in (0, 1, 2):
        raise ValueError("JPEG quality must be 1..100; subsampling must be 0, 1 or 2")
    name = table_name or (repo_id or root.name).split("/")[-1]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name.endswith("_videos"):
        raise ValueError("table_name must use letters/digits/_/- and must not end in _videos")
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not overwrite:
        raise FileExistsError(f"Output is not empty: {output}; use --overwrite to replace it")
    if output.exists() and not output.is_dir():
        raise ValueError(f"Output must be a directory: {output}")
    info, episodes, shards = load_source(root)
    visual = {ft["dtype"] for ft in info["features"].values()} & VISUAL_DTYPES
    if mode == "video" and visual != {"video"}:
        raise ValueError("--mode video requires MP4-backed dtype=video cameras; use --mode jpeg for image data")
    files = video_files(root, info, episodes)
    schema, mapped = frame_schema(info["features"], mode, jpeg_storage)
    LOG.info("Source: %d episodes, %d frames, %d MP4 files; mode=%s, table=%s", len(episodes), info["total_frames"], len(files), mode, name)
    if dry_run:
        return output
    if staging_dir is not None:
        staging_dir = Path(staging_dir).expanduser().resolve()
        if staging_dir == root or root in staging_dir.parents or staging_dir == output or output in staging_dir.parents:
            raise ValueError("Staging directory must be outside the source and output")
        staging_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rtpolicy-lance-", dir=staging_dir) as work:
        staged = Path(work) / "dataset"
        staged.mkdir()
        db = lancedb.connect(str(staged))
        batches = frame_batches(
            root, info, episodes, shards, schema, mapped, files,
            batch_size=batch_size, num_workers=num_workers, quality=jpeg_quality,
            subsampling=jpeg_subsampling, tolerance_s=tolerance_s,
        )
        frames = db.create_table(name, data=pa.RecordBatchReader.from_batches(schema, batches), schema=schema)
        if frames.count_rows() != info["total_frames"]:
            raise ValueError("Converted frame count differs from source")
        if mode == "video":
            video_schema = pa.schema([
                pa.field("video_key", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("file_index", pa.int32()),
                pa.field("video_bytes", pa.large_binary(), metadata={b"lance-encoding:blob": b"true"}),
            ])
            def video_batches():
                for (key, chunk, file_index), path in sorted(files.items()):
                    LOG.info("Copying MP4 blob: %s", path.relative_to(root))
                    row = {"video_key": key, "chunk_index": chunk, "file_index": file_index, "video_bytes": path.read_bytes()}
                    yield pa.RecordBatch.from_pylist([row], schema=video_schema)
            videos = db.create_table(
                f"{name}_videos",
                data=pa.RecordBatchReader.from_batches(video_schema, video_batches()),
                schema=video_schema,
            )
            if videos.count_rows() != len(files):
                raise ValueError("Converted video file count differs from source")
        shutil.copytree(root / "meta", staged / "meta")
        manifest = {
            "storage_format": mode, "table_name": name, "source_codebase_version": "v3.0",
            "total_frames": info["total_frames"], "total_episodes": info["total_episodes"],
        }
        if mode == "jpeg":
            manifest["jpeg_storage"] = jpeg_storage
        (staged / "lance_format.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Source conversion and verification finish before replacing anything.
        # Copying (rather than renaming) also works on EFS.
        if output.exists() and overwrite:
            shutil.rmtree(output)
        shutil.copytree(staged, output, dirs_exist_ok=True)
    LOG.info("Converted dataset: %s", output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="Local LeRobot v3 source directory")
    parser.add_argument("--output", type=Path, required=True, help="Destination Lance directory")
    parser.add_argument("--repo-id", help="Optional dataset id/name; defaults to source directory name")
    parser.add_argument("--table-name", help="Override the frames table name")
    parser.add_argument(
        "--mode", choices=("video", "jpeg"), default="jpeg",
        help="video: retain MP4 blobs; jpeg: JPEG frame table",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4, help="JPEG encoding threads")
    parser.add_argument("--jpeg-quality", type=int, default=100)
    parser.add_argument("--jpeg-subsampling", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--jpeg-storage", choices=("binary", "blob"), default="binary",
                        help="JPEG column layout: binary (legacy) or blob (supports mounted range reads)")
    parser.add_argument("--tolerance-s", type=float, default=1e-4)
    parser.add_argument(
        "--staging-dir", type=Path,
        help="Local scratch parent with room for the complete output; defaults to TMPDIR or /tmp",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace output only after staged conversion succeeds")
    parser.add_argument("--dry-run", action="store_true", help="Validate metadata and paths without writing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        convert_dataset(**vars(args))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
