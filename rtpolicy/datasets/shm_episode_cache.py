import argparse
import array
import ctypes
import mmap
import os
import pickle
import socket
import socketserver
import struct
import threading
import time
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from rtpolicy.datasets.episode_cache import estimate_item_bytes, register_omegaconf_resolvers


def _memfd_create(name):
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, flags=0)
    libc = ctypes.CDLL(None, use_errno=True)
    syscall_number = 319  # Linux x86_64 memfd_create
    fd = libc.syscall(syscall_number, name.encode(), 0)
    if fd < 0:
        errno_value = ctypes.get_errno()
        raise OSError(errno_value, os.strerror(errno_value))
    return int(fd)


def _send_message(sock, message):
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(payload)))
    sock.sendall(payload)


def _recv_exact(sock, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("socket closed while reading message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_message(sock):
    header = _recv_exact(sock, 8)
    size = struct.unpack("!Q", header)[0]
    return pickle.loads(_recv_exact(sock, size))


def _send_fd_message(sock, fd, message):
    payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    frame = struct.pack("!Q", len(payload)) + payload
    fds = array.array("i", [fd])
    sock.sendmsg([frame], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds)])


def _recv_fd_message(sock):
    msg, ancdata, flags, addr = sock.recvmsg(8, socket.CMSG_SPACE(array.array("i", [0]).itemsize))
    if len(msg) < 8:
        msg += _recv_exact(sock, 8 - len(msg))
    size = struct.unpack("!Q", msg[:8])[0]
    fd = None
    for level, ctype, data in ancdata:
        if level == socket.SOL_SOCKET and ctype == socket.SCM_RIGHTS:
            fds = array.array("i")
            fds.frombytes(data[: fds.itemsize])
            fd = int(fds[0])
            break
    payload = msg[8:]
    if len(payload) < size:
        payload += _recv_exact(sock, size - len(payload))
    if fd is None:
        raise RuntimeError("shared memory attach response did not include a file descriptor")
    return fd, pickle.loads(payload)


def _value_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _ensure_contiguous_array(value, dtype=None):
    arr = np.asarray(value, dtype=dtype)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


class ShmColumnStore:
    def __init__(self, metadata, mm):
        self.metadata = metadata
        self.mm = mm
        self.arrays = {}
        for key, spec in metadata["columns"].items():
            self.arrays[key] = np.ndarray(
                shape=tuple(spec["shape"]),
                dtype=np.dtype(spec["dtype"]),
                buffer=mm,
                offset=int(spec["offset"]),
            )
        ranges = [tuple(map(int, item)) for item in metadata.get("cached_ranges", [])]
        ranges.sort(key=lambda item: item[0])
        self.cached_ranges = ranges
        self.cached_starts = [start for start, _ in ranges]
        self.cached_ends = [end for _, end in ranges]
        self.misses = 0

    def _require_cached_idx(self, idx):
        idx = int(idx)
        range_idx = bisect_right(self.cached_starts, idx) - 1
        if range_idx < 0 or idx >= self.cached_ends[range_idx]:
            self.misses += 1
            raise KeyError(f"Dataset index {idx} is not cached in shared memory")
        return idx

    def get_row(self, idx, include_images=True, image_keys=None):
        idx = self._require_cached_idx(idx)
        image_keys = set(image_keys or [])
        row = {}
        for key, arr in self.arrays.items():
            if include_images or key not in image_keys:
                row[key] = torch.from_numpy(np.asarray(arr[idx]))
        return row

    def get_rows(self, query_indices):
        result = {}
        for key, indices in query_indices.items():
            if key not in self.arrays:
                raise KeyError(key)
            arr = self.arrays[key]
            checked_indices = [self._require_cached_idx(idx) for idx in indices]
            result[key] = [torch.from_numpy(np.asarray(arr[idx])) for idx in checked_indices]
        return result

class ShmEpisodeCacheBuilder:
    def __init__(self, dataset, report_interval=1000, max_episodes=None, build_num_workers=1):
        self.dataset = dataset
        self.report_interval = report_interval
        self.max_episodes = max_episodes
        self.build_num_workers = max(1, int(build_num_workers or 1))

    def _cache_episode_ids(self):
        episodes = list(self.dataset.selected_episodes)
        if self.max_episodes is not None:
            episodes = episodes[: max(0, int(self.max_episodes))]
        return episodes

    def _infer_layout(self, first_episode, cache_episode_ids):
        num_rows = 0
        cached_ranges = []
        cached_episode_indices = []
        for ep_idx in cache_episode_ids:
            start, end = self.dataset._episode_ranges[int(ep_idx)]
            start = int(start)
            end = int(end)
            cached_ranges.append([start, end])
            cached_episode_indices.append(int(ep_idx))
            num_rows = max(num_rows, end)

        metadata = {
            "backend": "shm",
            "num_rows": num_rows,
            "columns": {},
            "estimated_gib": 0.0,
            "cached_ranges": cached_ranges,
            "cached_episode_indices": cached_episode_indices,
            "cached_episodes": len(cache_episode_ids),
            "total_episodes": len(self.dataset.selected_episodes),
            "max_episodes": self.max_episodes,
            "build_num_workers": self.build_num_workers,
        }
        offset = 0
        for key, values in first_episode["rows"].items():
            if not values:
                continue
            sample = _ensure_contiguous_array(_value_to_numpy(values[0]))
            shape = [num_rows] + list(sample.shape)
            nbytes = int(np.prod(shape, dtype=np.int64)) * int(sample.dtype.itemsize)
            metadata["columns"][key] = {
                "dtype": str(sample.dtype),
                "shape": shape,
                "offset": offset,
                "nbytes": nbytes,
            }
            offset += nbytes
        metadata["estimated_gib"] = offset / 1024**3
        return metadata, offset

    def _open_column_arrays(self, metadata, mm):
        arrays = {}
        for key, spec in metadata["columns"].items():
            arrays[key] = np.ndarray(
                shape=tuple(spec["shape"]),
                dtype=np.dtype(spec["dtype"]),
                buffer=mm,
                offset=int(spec["offset"]),
            )
        return arrays

    def _write_episode(self, episode, arrays):
        start = int(episode["start"])
        end = int(episode["end"])
        row_count = end - start
        for key, arr in arrays.items():
            values = episode["rows"].get(key)
            if values is None:
                raise KeyError(f"Column {key!r} is missing from an episode during shared memory cache build")
            if len(values) != row_count:
                raise ValueError(
                    f"Column {key!r} has {len(values)} rows for episode range [{start}, {end})"
                )
            if row_count == 0:
                continue
            try:
                chunk = np.stack([_value_to_numpy(value) for value in values], axis=0)
            except ValueError as exc:
                raise ValueError(f"Column {key!r} cannot be stacked into shared memory") from exc
            chunk = _ensure_contiguous_array(chunk, dtype=arr.dtype)
            expected_shape = arr[start:end].shape
            if chunk.shape != expected_shape:
                raise ValueError(
                    f"Column {key!r} has shape {chunk.shape}, expected {expected_shape} "
                    f"for episode range [{start}, {end})"
                )
            arr[start:end] = chunk

    def _iter_loaded_episodes(self, cache_episode_ids, first_ep_idx, first_episode):
        yield 1, first_ep_idx, first_episode
        remaining = [ep_idx for ep_idx in cache_episode_ids if int(ep_idx) != int(first_ep_idx)]
        if not remaining:
            return
        if self.build_num_workers <= 1:
            for count, ep_idx in enumerate(remaining, start=2):
                yield count, ep_idx, self.dataset._load_episode_to_ram(ep_idx)
            return
        with ThreadPoolExecutor(max_workers=self.build_num_workers) as executor:
            for count, (ep_idx, episode) in enumerate(
                zip(remaining, executor.map(self.dataset._load_episode_to_ram, remaining)),
                start=2,
            ):
                yield count, ep_idx, episode

    def build(self):
        start_time = time.monotonic()
        cache_episode_ids = self._cache_episode_ids()
        total_episodes = len(self.dataset.selected_episodes)
        print(
            f"Shm episode RAM cache build start: episodes={len(cache_episode_ids)}/{total_episodes}, "
            f"build_num_workers={self.build_num_workers}",
            flush=True,
        )
        if not cache_episode_ids:
            raise ValueError("Cannot build shared memory episode cache with no selected episodes")

        first_ep_idx = int(cache_episode_ids[0])
        first_episode = self.dataset._load_episode_to_ram(first_ep_idx)
        metadata, total_nbytes = self._infer_layout(first_episode, cache_episode_ids)
        if total_nbytes <= 0:
            raise ValueError("Shared memory episode cache has no cacheable columns")

        fd = _memfd_create("rtpolicy_shm_episode_cache")
        try:
            os.ftruncate(fd, total_nbytes)
            mm = mmap.mmap(fd, total_nbytes, access=mmap.ACCESS_WRITE)
            try:
                arrays = self._open_column_arrays(metadata, mm)
                for count, ep_idx, episode in self._iter_loaded_episodes(cache_episode_ids, first_ep_idx, first_episode):
                    self._write_episode(episode, arrays)
                    if self.report_interval and count % self.report_interval == 0:
                        elapsed = time.monotonic() - start_time
                        print(
                            f"Shm episode RAM cache build: {count}/{len(cache_episode_ids)} episodes, elapsed={elapsed:.1f}s",
                            flush=True,
                        )
                mm.flush()
            except Exception:
                mm.close()
                raise
        except Exception:
            os.close(fd)
            raise

        elapsed = time.monotonic() - start_time
        print(
            f"Shm episode RAM cache build done: rows={metadata['num_rows']}, "
            f"cached_episodes={metadata['cached_episodes']}/{metadata['total_episodes']}, "
            f"columns={len(metadata['columns'])}, estimated_size={metadata['estimated_gib']:.2f} GiB, "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        return fd, mm, metadata

class ShmEpisodeCacheClient:
    def __init__(self, socket_path, timeout=600.0):
        self.socket_path = str(socket_path)
        self.timeout = float(timeout)
        self.fd = None
        self.mm = None
        self.metadata = None
        self.store = None

    def _request(self, message):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout)
            sock.connect(self.socket_path)
            _send_message(sock, message)
            response = _recv_message(sock)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "shm episode cache request failed"))
        return response

    def _attach(self):
        if self.store is not None:
            return
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout)
            sock.connect(self.socket_path)
            _send_message(sock, {"cmd": "attach"})
            fd, response = _recv_fd_message(sock)
        if not response.get("ok"):
            os.close(fd)
            raise RuntimeError(response.get("error", "shm episode cache attach failed"))
        self.fd = fd
        self.metadata = response["metadata"]
        total_nbytes = sum(int(spec["nbytes"]) for spec in self.metadata["columns"].values())
        self.mm = mmap.mmap(fd, total_nbytes, access=mmap.ACCESS_WRITE)
        self.store = ShmColumnStore(self.metadata, self.mm)

    def get_row(self, idx, include_images=True, image_keys=None):
        self._attach()
        return self.store.get_row(idx, include_images=include_images, image_keys=image_keys)

    def get_rows(self, query_indices):
        self._attach()
        return self.store.get_rows(query_indices)

    def stats(self):
        stats = self._request({"cmd": "stats"})["stats"]
        if self.store is not None:
            stats["misses"] = int(getattr(self.store, "misses", 0))
        else:
            stats.setdefault("misses", 0)
        return stats

    def ping(self):
        return self._request({"cmd": "ping"})["status"] == "ok"

    def stop(self):
        return self._request({"cmd": "stop"})

    def close(self):
        if self.mm is not None:
            self.mm.close()
            self.mm = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.store = None


class _ThreadingUnixStreamServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    request_queue_size = 4096


class _ShmRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            message = _recv_message(self.request)
            cmd = message.get("cmd")
            if cmd == "attach":
                self.server.requests += 1
                _send_fd_message(self.request, self.server.cache_fd, {"ok": True, "metadata": self.server.metadata})
            elif cmd == "stats":
                _send_message(self.request, {"ok": True, "stats": self.server.get_stats()})
            elif cmd == "ping":
                _send_message(self.request, {"ok": True, "status": "ok"})
            elif cmd == "stop":
                _send_message(self.request, {"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
            else:
                _send_message(self.request, {"ok": False, "error": f"unknown command: {cmd}"})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                _send_message(self.request, {"ok": False, "error": repr(exc)})
            except (BrokenPipeError, ConnectionResetError):
                return


class ShmEpisodeCacheServer:
    def __init__(self, dataset, socket_path, report_interval=1000, max_episodes=None, build_num_workers=1):
        self.dataset = dataset
        self.socket_path = str(socket_path)
        self.report_interval = report_interval
        self.max_episodes = max_episodes
        self.build_num_workers = build_num_workers
        self.cache_fd = None
        self.cache_mmap = None
        self.metadata = None
        self.server = None
        self.thread = None

    def start(self):
        builder = ShmEpisodeCacheBuilder(
            self.dataset,
            report_interval=self.report_interval,
            max_episodes=self.max_episodes,
            build_num_workers=self.build_num_workers,
        )
        self.cache_fd, self.cache_mmap, self.metadata = builder.build()
        Path(self.socket_path).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self.server = _ThreadingUnixStreamServer(self.socket_path, _ShmRequestHandler)
        self.server.cache_fd = self.cache_fd
        self.server.metadata = self.metadata
        self.server.requests = 0
        self.server.started_at = time.monotonic()
        self.server.get_stats = self._get_stats
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def _get_stats(self):
        return {
            "backend": "shm",
            "requests": int(getattr(self.server, "requests", 0)),
            "num_rows": int(self.metadata.get("num_rows", 0)),
            "columns": len(self.metadata.get("columns", {})),
            "estimated_gib": float(self.metadata.get("estimated_gib", 0.0)),
            "cached_episodes": int(self.metadata.get("cached_episodes", 0)),
            "total_episodes": int(self.metadata.get("total_episodes", 0)),
            "max_episodes": self.metadata.get("max_episodes"),
            "build_num_workers": int(self.metadata.get("build_num_workers", 1)),
            "misses": 0,
        }

    def serve_forever(self):
        self.start()
        try:
            self.thread.join()
        finally:
            self.close()

    def stop(self):
        try:
            ShmEpisodeCacheClient(self.socket_path, timeout=5.0).stop()
        finally:
            if self.thread is not None:
                self.thread.join(timeout=5.0)
            self.close()

    def close(self):
        if self.server is not None:
            self.server.server_close()
            self.server = None
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        if self.cache_mmap is not None:
            self.cache_mmap.close()
            self.cache_mmap = None
        if self.cache_fd is not None:
            os.close(self.cache_fd)
            self.cache_fd = None


def main():
    parser = argparse.ArgumentParser(description="Run a shared-memory RTPolicy episode cache server.")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--cache-report-interval", type=int, default=1000)
    parser.add_argument("--cache-max-episodes", type=int, default=None)
    parser.add_argument("--build-num-workers", type=int, default=1)
    parser.add_argument("--ready-file", default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    register_omegaconf_resolvers()
    config_dir = str(Path(args.config_path).resolve())
    config_name = args.config_name[:-5] if args.config_name.endswith(".yaml") else args.config_name
    overrides = list(args.overrides)
    overrides.extend([
        "++train_dataset.cache_to_ram=false",
        "++train_dataset.cache_backend=local",
    ])
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    dataset = hydra.utils.instantiate(cfg.train_dataset)
    server = ShmEpisodeCacheServer(
        dataset=dataset,
        socket_path=args.socket,
        report_interval=args.cache_report_interval,
        max_episodes=args.cache_max_episodes,
        build_num_workers=args.build_num_workers,
    )
    print(f"Shm episode RAM cache server starting: socket={args.socket}", flush=True)
    server.start()
    if args.ready_file:
        Path(args.ready_file).parent.mkdir(parents=True, exist_ok=True)
        Path(args.ready_file).write_text("ready\n")
    try:
        server.thread.join()
    finally:
        server.close()


if __name__ == "__main__":
    main()
