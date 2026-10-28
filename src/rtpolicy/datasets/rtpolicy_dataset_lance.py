"""Lance-backend RTPolicyDataset (self-contained, no lerobot_lancedb dependency).

The lerobot backend pins lerobot 0.3.3 while lerobot_lancedb requires
lerobot>=0.5, and the two cannot coexist in one environment -- so this module
reads Lance tables directly via lancedb/pylance and parses the v3.0 meta/
sidecar itself, running alongside the lerobot backend in the RTPolicy env.
Convert LeRobot v3 data with tools/lerobot_v30_to_lance.py. Both JPEG frame
tables and frames + MP4 blob tables are supported.

Known differences from the lerobot backend: no episode RAM cache (cache_*
keys), no automatic static-joint mask, no npy feature loading (those config
keys are ignored).
"""
import io
import json
from bisect import bisect_left, bisect_right
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, decode_jpeg

from rtpolicy.common.normalizer import LinearNormalizer
from rtpolicy.common.normalize_util import (
    get_identity_normalizer_from_stat,
    get_range_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
)

# Duplicated from rtpolicy_dataset.py: that module imports lerobot 0.3.3 API at
# module level, so this module deliberately avoids importing lerobot and must
# stay self-contained.
DEFAULT_KEYS = {"timestamp", "frame_indx", "episode_index", "index", "task_index", "frame_index"}
FILTER_KEYS = {"timestamp", "frame_indx", "index"}
ACTION_KEYS = ("action", "state", "delta_state", "endpose", "delta_endpose")
SCALAR_KEYS = ("episode_index", "frame_index", "index", "timestamp", "task_index")


def downsample_mask(mask, max_n, seed=0):
    if max_n is None or np.sum(mask) <= max_n:
        return mask

    curr_train_idxs = np.nonzero(mask)[0]
    rng = np.random.default_rng(seed=seed)
    train_idxs = curr_train_idxs[rng.choice(len(curr_train_idxs), size=int(max_n), replace=False)]

    new_mask = np.zeros_like(mask)
    new_mask[train_idxs] = True
    return new_mask


def get_val_mask(n_episodes, val_ratio, seed=0):
    if val_ratio <= 0:
        return np.zeros(n_episodes, dtype=bool)

    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes - 1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)

    val_mask = np.zeros(n_episodes, dtype=bool)
    val_mask[val_idxs] = True
    return val_mask


class _LanceMeta:
    """v3.0 meta/ sidecar parsing (info.json + tasks/episodes parquet + stats.json)."""

    def __init__(self, root):
        self.root = Path(root)
        info = json.loads((self.root / "meta/info.json").read_text())
        self.fps = info["fps"]
        self.features = info["features"]
        self.total_episodes = info["total_episodes"]
        self.image_keys = [k for k, ft in self.features.items() if ft["dtype"] in ("image", "video")]
        self.camera_keys = self.image_keys

        tasks = pq.read_table(self.root / "meta/tasks.parquet").to_pydict()
        if "task" not in tasks:
            tasks["task"] = tasks["__index_level_0__"]
        self.tasks = [None] * len(tasks["task"])
        for task, task_index in zip(tasks["task"], tasks["task_index"]):
            self.tasks[task_index] = task

        rows = []
        for path in sorted(self.root.glob("meta/episodes/**/*.parquet")):
            names = pq.read_schema(path).names
            columns = ["episode_index", "tasks", "dataset_from_index", "dataset_to_index"]
            columns += [name for name in names if name.startswith("videos/")]
            rows.extend(pq.read_table(path, columns=columns).to_pylist())
        # Per-episode [from, to) spans, sorted by from (for binary search and window clamping)
        self.episodes = sorted(rows, key=lambda ep: ep["dataset_from_index"])
        self.episodes_by_index = {ep["episode_index"]: ep for ep in self.episodes}

        stats = json.loads((self.root / "meta/stats.json").read_text())
        self.stats = {
            key: {name: np.asarray(value[name], dtype=np.float32) for name in ("min", "max", "mean", "std")}
            for key, value in stats.items()
        }


class _LanceReader:
    """Lance table reader: numeric columns stay in memory, image columns are
    randomly taken per observed frame.

    take_offsets collapses duplicate rows (permutation semantics), so _fetch
    deduplicates internally and gathers back in the original order.

    The tokio runtime inside lancedb does not survive fork() (workers get heap
    corruption; nearly always crashes at num_workers>=8, see the analysis in
    lerobot_lancedb/_spawn_compat.py), and with decode_device=cuda the parent
    process has already initialized CUDA so forked workers cannot use CUDA --
    both cases require DataLoader workers to use spawn.
    """

    _SPAWN_FORCED = False

    @staticmethod
    def resolve_table(root, table_name=None):
        if table_name is not None:
            if not (Path(root) / f"{table_name}.lance").is_dir():
                raise FileNotFoundError(f"Missing Lance table: {table_name!r} at {root}")
            return table_name
        candidates = sorted(p.stem for p in Path(root).glob("*.lance") if not p.stem.endswith("_videos"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one frames table at {root}, found {candidates}; set table_name explicitly")
        return candidates[0]

    def __init__(self, root, keys, image_keys, decode_device=None, table_name=None, image_cache_size_bytes=0, image_io_backend="auto"):
        self._ensure_spawn_workers()
        root = Path(root).resolve()
        table_name = self.resolve_table(root, table_name)
        table = lancedb.connect(str(root)).open_table(table_name)
        self._column_names = {
            key: key if key in table.schema.names else key.replace(".", "_")
            for key in dict.fromkeys([*keys, *SCALAR_KEYS])
        }
        missing = [key for key, name in self._column_names.items() if name not in table.schema.names]
        if missing:
            raise ValueError(f"Lance table {table_name!r} missing columns: {missing}")
        image_keys = set(image_keys)
        self._image_columns = [key for key in keys if key in image_keys]
        self._numeric_columns = [key for key in keys if key not in image_keys] + list(SCALAR_KEYS)
        self._lance_dataset = table.to_lance()
        self._table_uri = str(root / f"{table_name}.lance")
        numeric_table = self._lance_dataset.to_table(columns=[self._column_names[k] for k in self._numeric_columns])
        arrays = self._materialize_columns(numeric_table)
        self._numeric_arrays = {key: arrays[self._column_names[key]] for key in self._numeric_columns}
        self._decode_device = decode_device
        self._image_table = None
        self._image_batches = None
        self._image_batch_starts = None
        self._image_io_backend = image_io_backend
        self._blob_image_names = [
            self._column_names[key] for key in self._image_columns
            if (table.schema.field(self._column_names[key]).metadata or {}).get(b"lance-encoding:blob") == b"true"
        ]
        self._blob_images = None
        if image_io_backend in ("file", "mmap") and len(self._blob_image_names) != len(self._image_columns):
            raise ValueError(f"image_io_backend={image_io_backend} requires blob columns for all selected cameras")
        if self._blob_image_names:
            if image_cache_size_bytes:
                raise ValueError("JPEG blob columns use streaming; set image_cache_size_bytes=0")
            self._open_blob_images()
        if image_cache_size_bytes:
            batches = []
            cached_bytes = 0
            names = [self._column_names[key] for key in self._image_columns]
            for batch in self._lance_dataset.to_batches(columns=names, batch_size=128, scan_in_order=True):
                cached_bytes += batch.nbytes
                if cached_bytes > image_cache_size_bytes:
                    raise ValueError(
                        f"Encoded image cache exceeds its {image_cache_size_bytes} byte budget; "
                        "increase image_cache_size_bytes or set it to 0."
                    )
                batches.append(batch)
            self._image_table = pa.Table.from_batches(batches)
            self._image_batches = batches
            self._image_batch_starts = np.cumsum([0] + [batch.num_rows for batch in batches]).tolist()

    def __getstate__(self):
        if self._image_table is not None:
            raise RuntimeError("Encoded image RAM cache requires DataLoader num_workers=0")
        state = self.__dict__.copy()
        state["_lance_dataset"] = None
        return state

    def _ensure_open(self):
        if self._lance_dataset is None:
            import lance

            self._lance_dataset = lance.dataset(self._table_uri)
            if self._blob_images is None:
                self._open_blob_images()

    def _open_blob_images(self):
        if self._blob_image_names:
            from rtpolicy.datasets.lance_images import BlobImageReader

            self._blob_images = BlobImageReader(
                self._lance_dataset, self._table_uri, self._blob_image_names, self._image_io_backend,
            )

    @classmethod
    def _ensure_spawn_workers(cls):
        """Switch DataLoader worker start method from fork to spawn on Linux (idempotent).

        Each worker pays a one-off ~7s interpreter startup; in exchange we no
        longer gamble on fork heap integrity. Non-Linux processes and those
        that already chose a non-fork method are left untouched.
        """
        if cls._SPAWN_FORCED:
            return
        cls._SPAWN_FORCED = True

        import multiprocessing as mp
        import sys

        if sys.platform != "linux":
            return
        current = mp.get_start_method(allow_none=True)
        if current not in (None, "fork"):
            return
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass  # If fork cannot be switched away from, keep as is; num_workers=0 or low concurrency usually still works
        torch.multiprocessing.set_sharing_strategy("file_system")

    @staticmethod
    def _materialize_columns(batch):
        columns = {}
        for name in batch.schema.names:
            col = batch.column(name)
            if pa.types.is_binary(col.type) or pa.types.is_large_binary(col.type):
                columns[name] = col.to_pylist()
            elif pa.types.is_fixed_size_list(col.type):
                flat = np.asarray(pc.list_flatten(col).to_numpy(zero_copy_only=False))
                dim = col.type.list_size
                columns[name] = flat.reshape(len(col), dim)
            else:
                columns[name] = np.asarray(col.to_numpy(zero_copy_only=False))
        return columns

    def _fetch(self, rows, selected_columns):
        """Random take of selected columns, preserving repeated-row order."""
        unique = sorted(set(rows))
        names = [self._column_names[k] for k in selected_columns]
        if self._image_table is not None:
            # Table.take concatenates binary chunks first (and overflows at
            # 2 GiB per column). Gather only the requested rows of each chunk.
            selected = []
            position = 0
            while position < len(unique):
                chunk = bisect_right(self._image_batch_starts, unique[position]) - 1
                end = position + 1
                while end < len(unique) and unique[end] < self._image_batch_starts[chunk + 1]:
                    end += 1
                local = [row - self._image_batch_starts[chunk] for row in unique[position:end]]
                selected.append(self._image_batches[chunk].select(names).take(pa.array(local, type=pa.int64())))
                position = end
            batch = pa.Table.from_batches(selected, schema=self._image_table.select(names).schema)
        else:
            if self._blob_images is None:
                self._ensure_open()
            if self._blob_images is not None:
                blob_names = [name for name in names if name in self._blob_images.names]
                regular_names = [name for name in names if name not in self._blob_images.names]
                arrays = self._blob_images.take(unique, blob_names)
                if regular_names:
                    self._ensure_open()
                    arrays.update(self._materialize_columns(self._lance_dataset.take(unique, columns=regular_names)))
                offsets = [bisect_left(unique, row) for row in rows]
                return {key: [arrays[self._column_names[key]][i] for i in offsets] for key in selected_columns}
            batch = self._lance_dataset.take(unique, columns=names)
        arrays = self._materialize_columns(batch)
        gathered = {key: arrays[self._column_names[key]] for key in selected_columns}
        offsets = [bisect_left(unique, row) for row in rows]
        return {
            name: [values[i] for i in offsets] if isinstance(values, list) else values[offsets]
            for name, values in gathered.items()
        }

    def fetch(self, rows):
        return {
            name: [values[row] for row in rows] if isinstance(values, list) else values[rows]
            for name, values in self._numeric_arrays.items()
        }

    def fetch_images(self, rows):
        if not rows or not self._image_columns:
            return {}
        return self._fetch(rows, self._image_columns)

    def decode_images(self, blobs):
        """JPEG bytes -> normalized CHW frames, allowing different camera sizes."""
        if not blobs:
            return torch.empty(0)
        try:
            tensors = [torch.frombuffer(bytearray(b), dtype=torch.uint8) for b in blobs]
            images = decode_jpeg(tensors, mode=ImageReadMode.RGB, device=self._decode_device or "cpu")
        except (RuntimeError, TypeError):
            from PIL import Image

            arrays = [np.asarray(Image.open(io.BytesIO(b)).convert("RGB")).transpose(2, 0, 1) for b in blobs]
            images = [torch.from_numpy(arr.copy()) for arr in arrays]
        return self._normalize_images(images)

    @staticmethod
    def _normalize_images(images):
        if not images:
            return torch.empty(0)
        if all(img.shape == images[0].shape for img in images):
            # Preserve the original vectorized normalization for equal-sized
            # cameras, which is the common training configuration.
            return torch.stack(images).float().div_(255.0)
        return [img.float().div_(255.0) for img in images]


class _LanceVideoReader(_LanceReader):
    """Numeric frame cache + per-worker, bounded MP4 decoder cache."""

    def __init__(
        self, root, keys, meta, table_name, cache_size=16, tolerance_s=1e-4,
        decode_device="cpu", video_backend="auto", video_seek_mode="approximate",
        video_seek_gap_s=None, video_cache_size_bytes=0, video_read_buffer_size=1024 * 1024,
        video_io_backend="lance",
    ):
        import lance

        image_keys = [key for key in keys if key in meta.image_keys]
        super().__init__(root, [key for key in keys if key not in image_keys], [], table_name=table_name)
        self._image_columns = image_keys
        self._videos_uri = str(Path(root).resolve() / f"{table_name}_videos.lance")
        self._videos_dataset = lance.dataset(self._videos_uri)
        identities = self._videos_dataset.to_table(columns=["video_key", "chunk_index", "file_index"]).to_pylist()
        self._file_rows = {
            (r["video_key"], int(r["chunk_index"]), int(r["file_index"])): i
            for i, r in enumerate(identities)
        }
        self._episode_files = {}
        for ep in meta.episodes:
            for key in image_keys:
                prefix = f"videos/{key}/"
                identity = (key, int(ep[prefix + "chunk_index"]), int(ep[prefix + "file_index"]))
                if identity not in self._file_rows:
                    raise ValueError(f"Missing video blob for {identity}")
                self._episode_files[(int(ep["episode_index"]), key)] = (identity, float(ep[prefix + "from_timestamp"]))
        self._cache_size = cache_size
        self._tolerance_s = tolerance_s
        self._decode_device = decode_device
        self._video_backend = video_backend
        self._video_seek_mode = video_seek_mode
        self._video_seek_gap_s = video_seek_gap_s
        self._video_read_buffer_size = video_read_buffer_size
        self._video_slices = None
        self._file_pool = None
        if video_io_backend in ("auto", "file"):
            from rtpolicy.datasets.lance_video import mounted_video_slices

            self._video_slices = mounted_video_slices(self._videos_dataset, self._videos_uri)
            if self._video_slices is None and video_io_backend == "file":
                raise ValueError("video_io_backend=file requires mounted Lance blob v1 single-file fragments")
        self._decoder_cache = self._new_decoder_cache()
        self._video_bytes_cache = {}
        if video_cache_size_bytes:
            cached_bytes = 0
            for identity, row in self._file_rows.items():
                if identity[0] not in self._image_columns:
                    continue
                with self._videos_dataset.take_blobs(blob_column="video_bytes", indices=[row])[0] as blob:
                    size = blob.seek(0, io.SEEK_END)
                    if cached_bytes + size > video_cache_size_bytes:
                        raise ValueError(
                            f"Encoded video cache exceeds its {video_cache_size_bytes} byte budget; "
                            "increase video_cache_size_bytes or set it to 0."
                        )
                    blob.seek(0)
                    data = blob.read()
                self._video_bytes_cache[identity] = data
                cached_bytes += len(data)

    def _open_video_source(self, identity):
        from rtpolicy.datasets.lance_video import buffered_video_blob

        if identity in self._video_bytes_cache:
            return io.BytesIO(self._video_bytes_cache[identity])
        if self._video_slices is not None:
            from rtpolicy.datasets.lance_video import FileSlice, MountedFilePool

            if self._file_pool is None:
                self._file_pool = MountedFilePool(self._cache_size)
            return io.BufferedReader(
                FileSlice(*self._video_slices[identity], pool=self._file_pool), self._video_read_buffer_size,
            )
        return buffered_video_blob(
            self._videos_dataset.take_blobs(blob_column="video_bytes", indices=[self._file_rows[identity]])[0],
            buffer_size=self._video_read_buffer_size,
        )

    def _new_decoder_cache(self):
        from rtpolicy.datasets.lance_video import VideoDecoderCache

        return VideoDecoderCache(
            self._cache_size, self._tolerance_s, self._decode_device,
            self._video_backend, self._video_seek_mode,
            self._video_seek_gap_s,
        )

    def __getstate__(self):
        state = super().__getstate__()
        state["_videos_dataset"] = None
        state["_decoder_cache"] = None
        state["_file_pool"] = None
        return state

    def fetch_images(self, rows):
        if not rows or not self._image_columns:
            return {}
        if self._videos_dataset is None:
            import lance
            self._videos_dataset = lance.dataset(self._videos_uri)
            self._decoder_cache = self._new_decoder_cache()
        results = {key: [None] * len(rows) for key in self._image_columns}
        plan = {}
        for pos, row in enumerate(rows):
            ep = int(self._numeric_arrays["episode_index"][row])
            for key in self._image_columns:
                identity, offset = self._episode_files[(ep, key)]
                timestamp = float(self._numeric_arrays["timestamp"][row]) + offset
                plan.setdefault(identity, []).append((pos, timestamp))
        for identity, queries in plan.items():
            decoder = self._decoder_cache.get(
                identity,
                lambda: self._open_video_source(identity),
            )
            frames = decoder.decode([ts for _, ts in queries])
            for (pos, _), frame in zip(queries, frames):
                results[identity[0]][pos] = frame
        return results

    def decode_images(self, frames):
        tensors = [frame if torch.is_tensor(frame) else torch.from_numpy(frame.transpose(2, 0, 1).copy()) for frame in frames]
        return self._normalize_images(tensors)

    def __del__(self):
        cache = getattr(self, "_decoder_cache", None)
        if cache is not None:
            cache.close()
        pool = getattr(self, "_file_pool", None)
        if pool is not None:
            pool.close()


class RTPolicyDataset(Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        task_list: list[str] | None = None,
        task_index_list: list[int] | None = None,
        black_index: list[int] | None = None,
        horizon: int = 8,
        n_obs_steps: int | None = None,
        obs_image_frame_indices: list[int] | None = None,
        rgb_key: list[str] | str | None = None,
        pad_before: int | None = None,
        pad_after: int = 0,
        npy_feature_keys: list[str] | None = None,
        seed: int = 0,
        val_ratio: float = 0.0,
        max_train_episodes=None,
        decode_device: str | None = "cpu",
        storage_format: str = "auto",
        table_name: str | None = None,
        video_decoder_cache_size: int = 16,
        tolerance_s: float = 1e-4,
        video_backend: str = "auto",
        video_seek_mode: str = "approximate",
        video_seek_gap_s: float | None = None,
        image_cache_size_bytes: int = 0,
        image_io_backend: str = "auto",
        video_cache_size_bytes: int = 0,
        video_read_buffer_size: int = 1024 * 1024,
        video_io_backend: str = "lance",
        pin_memory: bool | None = None,
        **kwargs,
    ):
        super().__init__()

        self.repo_id = repo_id
        self.root = root
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps if n_obs_steps is not None else horizon
        if obs_image_frame_indices is None:
            obs_image_frame_indices = list(range(self.n_obs_steps))
        self.obs_image_frame_indices = list(obs_image_frame_indices)
        self.rgb_key = [rgb_key] if isinstance(rgb_key, str) else list(rgb_key) if rgb_key is not None else None
        self.pad_before = self.n_obs_steps if pad_before is None else pad_before
        self.pad_after = pad_after
        self.seed = seed
        self.val_ratio = val_ratio
        self.max_train_episodes = max_train_episodes
        self.npy_feature_keys = npy_feature_keys or []
        self.task_list = task_list
        self.task_index_list = task_index_list
        self.black_index = black_index
        self.detail_item = kwargs.pop("detail_item", False)
        if "dataloader_num_workers" in kwargs:
            raise ValueError("Set the top-level dataloader_num_workers; dataset-level worker overrides are not supported")
        self.table_name = _LanceReader.resolve_table(root, table_name)
        has_videos = (Path(root) / f"{self.table_name}_videos.lance").is_dir()
        if storage_format not in ("auto", "jpeg", "video"):
            raise ValueError(f"Unknown Lance storage_format: {storage_format!r}")
        self.storage_format = "video" if storage_format == "auto" and has_videos else storage_format
        if self.storage_format == "auto":
            self.storage_format = "jpeg"
        self.video_decoder_cache_size = video_decoder_cache_size
        self.tolerance_s = tolerance_s
        self.video_backend = video_backend
        self.video_seek_mode = video_seek_mode
        self.video_seek_gap_s = video_seek_gap_s
        self.image_cache_size_bytes = int(image_cache_size_bytes)
        if image_io_backend not in ("auto", "file", "mmap", "lance"):
            raise ValueError(f"Unknown image_io_backend: {image_io_backend}")
        self.image_io_backend = image_io_backend
        if self.image_cache_size_bytes < 0:
            raise ValueError("image_cache_size_bytes must be nonnegative")
        if self.image_cache_size_bytes and self.storage_format == "video":
            raise ValueError("image_cache_size_bytes applies to JPEG tables; video has a decoder cache")
        self.video_cache_size_bytes = int(video_cache_size_bytes)
        if self.video_cache_size_bytes < 0:
            raise ValueError("video_cache_size_bytes must be nonnegative")
        if self.video_cache_size_bytes and self.storage_format != "video":
            raise ValueError("video_cache_size_bytes applies to video tables")
        self.video_read_buffer_size = int(video_read_buffer_size)
        if video_io_backend not in ("auto", "lance", "file"):
            raise ValueError(f"Unknown video_io_backend: {video_io_backend}")
        self.video_io_backend = video_io_backend
        if self.video_read_buffer_size < 1:
            raise ValueError("video_read_buffer_size must be positive")
        if self.storage_format == "video":
            if not has_videos:
                raise FileNotFoundError(f"Missing {self.table_name}_videos.lance at {root}")
        if decode_device == "auto":
            decode_device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        decode_device = decode_device or "cpu"
        if str(decode_device).startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA decoding requested but CUDA is unavailable")
            device = torch.device(decode_device)
            decode_device = f"cuda:{device.index if device.index is not None else torch.cuda.current_device()}"
        self.decode_device = decode_device
        cuda_decode = str(decode_device).startswith("cuda")
        self.dataloader_num_workers = 0 if cuda_decode else None
        if self.image_cache_size_bytes:
            self.dataloader_num_workers = 0
        self.pin_memory = False if cuda_decode else (True if pin_memory is None else pin_memory)

        self.dataset_meta = self.meta = _LanceMeta(root)
        self.delta_timestamps = self._create_delta_timestamps()
        self._delta_indices = {
            key: [round(t * self.dataset_meta.fps) for t in ts] for key, ts in self.delta_timestamps.items()
        }
        self._keys = [
            key
            for key in self.dataset_meta.features
            if key not in DEFAULT_KEYS
            and not (
                self.dataset_meta.features[key]["dtype"] in ("image", "video")
                and self.rgb_key is not None
                and key not in self.rgb_key
            )
        ]
        if self.storage_format == "video":
            self._reader = _LanceVideoReader(
                root, self._keys, self.meta, self.table_name, video_decoder_cache_size, tolerance_s,
                decode_device, video_backend, video_seek_mode,
                video_seek_gap_s,
                self.video_cache_size_bytes,
                self.video_read_buffer_size,
                self.video_io_backend,
            )
        else:
            self._reader = _LanceReader(root, self._keys, self.dataset_meta.image_keys, decode_device, self.table_name, self.image_cache_size_bytes, self.image_io_backend)

        # [from, to) spans of all episodes, sorted by from, for window clamping
        self._ep_starts = np.array([ep["dataset_from_index"] for ep in self.dataset_meta.episodes], dtype=np.int64)
        self._ep_ends = np.array([ep["dataset_to_index"] for ep in self.dataset_meta.episodes], dtype=np.int64)

        if episodes is None:
            episodes = self._setup_train_val_split()
        self.episodes = [int(ep) for ep in episodes]
        self.ep_idx_to_arr_idx = {ep_idx: arr_idx for arr_idx, ep_idx in enumerate(self.episodes)}
        self.frame_indices = self._build_frame_indices(self.episodes)

    @property
    def fps(self):
        return self.dataset_meta.fps

    @property
    def features(self):
        return self.dataset_meta.features

    @property
    def num_frames(self):
        return len(self.frame_indices)

    @property
    def num_episodes(self):
        return len(self.episodes)

    def __len__(self):
        return len(self.frame_indices)

    def _create_delta_timestamps(self):
        timestamps = {}
        for key, feature in self.dataset_meta.features.items():
            if key in DEFAULT_KEYS:
                continue
            is_visual = feature["dtype"] in ("image", "video")
            if is_visual and self.rgb_key is not None and key not in self.rgb_key:
                continue
            if is_visual:
                frame_indices = [t - self.pad_before for t in self.obs_image_frame_indices]
            else:
                frame_indices = range(-self.pad_before, -self.pad_before + self.horizon)
            timestamps[key] = [t / self.dataset_meta.fps for t in frame_indices]
        return timestamps

    def _setup_train_val_split(self):
        total_episodes = self.dataset_meta.total_episodes
        available_mask = np.zeros(total_episodes, dtype=bool)
        self.train_mask = np.zeros(total_episodes, dtype=bool)
        self.val_mask = None

        if self.task_index_list is not None:
            if self.task_list is None:
                self.task_list = []
            self.task_list += [self.dataset_meta.tasks[ind] for ind in self.task_index_list]
        if self.task_list is None:
            self.task_list = list(self.dataset_meta.tasks)

        task_set = set(self.task_list)
        for ep in self.dataset_meta.episodes:
            available_mask[int(ep["episode_index"])] = bool(set(ep["tasks"]) & task_set)

        if self.black_index is not None:
            available_mask[self.black_index] = False

        val_mask = get_val_mask(int(available_mask.sum()), self.val_ratio, self.seed)
        if val_mask.sum() > 0:
            self.val_mask = np.zeros(total_episodes, dtype=bool)
            self.val_mask[available_mask] = val_mask

        train_mask = downsample_mask(~val_mask, self.max_train_episodes, self.seed)
        self.train_mask[available_mask] = train_mask
        return np.nonzero(self.train_mask)[0].tolist()

    def _build_frame_indices(self, episodes):
        frame_indices = []
        for ep_idx in episodes:
            ep = self.dataset_meta.episodes_by_index[int(ep_idx)]
            frame_indices.append(np.arange(ep["dataset_from_index"], ep["dataset_to_index"], dtype=np.int64))
        return np.concatenate(frame_indices)

    def get_validation_dataset(self):
        if self.val_mask is None:
            return None

        validation = self.__class__(
            repo_id=self.repo_id,
            root=self.root,
            episodes=np.nonzero(self.val_mask)[0].tolist(),
            horizon=self.horizon,
            n_obs_steps=self.n_obs_steps,
            obs_image_frame_indices=self.obs_image_frame_indices,
            rgb_key=self.rgb_key,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            npy_feature_keys=self.npy_feature_keys,
            seed=self.seed,
            val_ratio=0.0,
            max_train_episodes=None,
            decode_device=self.decode_device,
            storage_format=self.storage_format,
            table_name=self.table_name,
            video_decoder_cache_size=self.video_decoder_cache_size,
            tolerance_s=self.tolerance_s,
            video_backend=self.video_backend,
            video_seek_mode=self.video_seek_mode,
            video_seek_gap_s=self.video_seek_gap_s,
            video_read_buffer_size=self.video_read_buffer_size,
            image_io_backend=self.image_io_backend,
            video_io_backend=self.video_io_backend,
            pin_memory=self.pin_memory,
            detail_item=self.detail_item,
        )
        if self.image_cache_size_bytes:
            validation._reader._image_table = self._reader._image_table
            validation._reader._image_batches = self._reader._image_batches
            validation._reader._image_batch_starts = self._reader._image_batch_starts
            validation.image_cache_size_bytes = self.image_cache_size_bytes
        if self.video_cache_size_bytes:
            validation._reader._video_bytes_cache = self._reader._video_bytes_cache
            validation.video_cache_size_bytes = self.video_cache_size_bytes
        return validation

    def get_normalizer(self, **kwargs):
        normalizer = LinearNormalizer()

        if "endpose" in self.dataset_meta.features:
            normalizer["delta_endpose"] = self._compute_delta_stats("delta_endpose")
        if "state" in self.dataset_meta.features:
            normalizer["delta_state"] = self._compute_delta_stats("delta_state")

        for key in self.npy_feature_keys:
            normalizer[key] = self._create_identity_normalizer()

        for key, feature in self.dataset_meta.features.items():
            if key in DEFAULT_KEYS or feature["dtype"] in ("image", "video"):
                continue
            normalizer[key] = self._create_feature_normalizer(key, feature)

        return normalizer

    def _compute_delta_stats(self, key, num_samples=1):
        sample_size = min(len(self), num_samples)
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(len(self), size=sample_size, replace=False)
        all_data = np.stack([self[int(idx)]["action"][key].detach().cpu().numpy() for idx in indices], axis=0)

        stat = {
            "min": np.min(all_data, axis=0).astype(np.float32),
            "max": np.max(all_data, axis=0).astype(np.float32),
            "mean": np.mean(all_data, axis=0).astype(np.float32),
            "std": np.clip(np.std(all_data, axis=0).astype(np.float32), 1e-4, None),
        }

        if "endpose" in key:
            return robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
        return get_range_normalizer_from_stat(stat)

    def _create_identity_normalizer(self):
        stat = {
            "min": np.array([-1e8], dtype=np.float32),
            "max": np.array([1e8], dtype=np.float32),
            "mean": np.array([0], dtype=np.float32),
            "std": np.array([1e8], dtype=np.float32),
        }
        return get_identity_normalizer_from_stat(stat)

    def _create_feature_normalizer(self, key, feature):
        if feature["dtype"] == "string":
            return self._create_identity_normalizer()

        stats = self.dataset_meta.stats[key]
        if "endpose" in key:
            return robomimic_abs_action_only_dual_arm_normalizer_from_stat(stats)
        return get_range_normalizer_from_stat(stats)

    def _window_rows(self, abs_idx):
        """Map an absolute frame index to per-key window row indices (clamped to the episode)."""
        pos = bisect_left(self._ep_starts, abs_idx + 1) - 1
        start, end = int(self._ep_starts[pos]), int(self._ep_ends[pos])
        return {
            key: [min(end - 1, max(start, abs_idx + delta)) for delta in deltas]
            for key, deltas in self._delta_indices.items()
        }

    def __getitem__(self, idx):
        return self.__getitems__([idx])[0]

    def __getitems__(self, indices):
        layouts = []
        rows = []
        image_rows = []
        image_keys = set(self.dataset_meta.image_keys)
        for idx in indices:
            abs_idx = int(self.frame_indices[int(idx)])
            windows = self._window_rows(abs_idx)
            base_pos = len(rows)
            rows.append(abs_idx)
            spans = {}
            for key, w in windows.items():
                # Each span indexes its key's column batch: image_rows or rows.
                if key in image_keys:
                    spans[key] = (len(image_rows), len(w))
                    image_rows.extend(w)
                else:
                    spans[key] = (len(rows), len(w))
                    rows.extend(w)
            layouts.append((base_pos, spans))

        columns = self._reader.fetch(rows)
        columns.update(self._reader.fetch_images(image_rows))

        # Decode images as one batch: the per-call overhead of decode_jpeg far
        # exceeds the per-image decode, so concatenate the whole batch into a
        # single call and slice back.
        image_spans = [[] for _ in layouts]  # per-sample [(key, flat_start, length)]
        blobs = []
        for sample_i, (_, spans) in enumerate(layouts):
            for key in image_keys & spans.keys():
                start, length = spans[key]
                image_spans[sample_i].append((key, len(blobs), length))
                blobs.extend(columns[key][start : start + length])
        decoded = self._reader.decode_images(blobs)

        items = []
        for sample_i, (base_pos, spans) in enumerate(layouts):
            item = {key: torch.as_tensor(columns[key][base_pos]) for key in SCALAR_KEYS}
            for key, (start, length) in spans.items():
                if key not in image_keys:
                    item[key] = torch.from_numpy(np.asarray(columns[key][start : start + length]))
            for key, flat_start, length in image_spans[sample_i]:
                window = decoded[flat_start : flat_start + length]
                item[key] = window if isinstance(window, torch.Tensor) else torch.stack(window)
            items.append(self._postprocess_item(item))
        return items

    def _postprocess_item(self, item):
        if "endpose" in item:
            item["delta_endpose"] = item["endpose"] - item["endpose"][0:1]
            item["delta_endpose"][0:1] = item["endpose"][0:1]
        if "state" in item:
            item["delta_state"] = item["state"] - item["state"][0:1]
            item["delta_state"][0:1] = item["state"][0:1]

        return self._organize_item_data(item)

    def _organize_item_data(self, item):
        data = {"obs": {}, "action": {}}

        for key in ACTION_KEYS:
            if key in item:
                data["action"][key] = item[key]

        filter_keys = set(FILTER_KEYS)
        if not self.detail_item:
            filter_keys.update(DEFAULT_KEYS)

        valid_keys = {*self.npy_feature_keys, *self.dataset_meta.features.keys(), "delta_endpose", "delta_state"}
        for key, value in item.items():
            if key in valid_keys and key not in {*filter_keys, "action"}:
                data["obs"][key] = value

        return data
