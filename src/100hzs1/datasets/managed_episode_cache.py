"""Start one node-local episode cache for ranks sharing a training run."""

import atexit
import fcntl
import hashlib
import json
import os
from pathlib import Path

from .shm_episode_cache import ShmEpisodeCacheClient, ShmEpisodeCacheServer


_SERVERS = {}


def _socket_path(dataset):
    # torchrun/accelerate give local ranks the same rendezvous identity. /tmp
    # stays node-local even when TMPDIR points at a shared network filesystem.
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        run = [os.environ.get(key, "") for key in (
            "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID", "GROUP_RANK",
        )]
        if not run[0] or not run[1]:
            raise RuntimeError("Automatic multi-rank caching requires MASTER_ADDR and MASTER_PORT")
    else:
        run = [str(os.getpid())]
    identity = {
        "run": run,
        "uid": os.getuid(),
        "class": f"{type(dataset).__module__}.{type(dataset).__qualname__}",
        "root": str(Path(dataset.root).resolve()),
        "episodes": list(dataset.selected_episodes),
        "ranges": dataset._episode_ranges,
        "cameras": [key for key in dataset.meta.camera_keys if dataset._is_selected_visual_key(key)],
        "features": dataset.meta.features,
        "decoded_video": dataset.cache_decoded_video,
        "max_episodes": dataset.cache_max_episodes,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:32]
    return f"/tmp/100hzs1_episode_{digest}.sock"


def ensure_episode_cache(dataset):
    """Build once under a file lock; all other local ranks attach to its memfd."""
    path = _socket_path(dataset)
    client = ShmEpisodeCacheClient(path, timeout=5)
    with open(path + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if client.ping():
                return path
        except OSError:
            pass  # No live server, or a socket left by an interrupted run.
        server = ShmEpisodeCacheServer(
            dataset,
            path,
            report_interval=dataset.cache_report_interval,
            max_episodes=dataset.cache_max_episodes,
            build_num_workers=dataset.cache_build_num_workers,
        )
        try:
            server.start()
        except BaseException:
            server.close()
            raise
        # Retain only the memfd and socket service after the build, not another
        # reference to the HF dataset or any temporary decoded episodes.
        server.dataset = None
        _SERVERS[(os.getpid(), path)] = server
        print(f"100HzS1 shared episode cache ready: socket={path}, "
              f"estimated_size={server.metadata['estimated_gib']:.2f} GiB", flush=True)
    return path


def close_managed_episode_caches():
    """Only the creating process stops its services; forked workers never do."""
    for key, server in list(_SERVERS.items()):
        if key[0] == os.getpid():
            server.stop()
            del _SERVERS[key]


atexit.register(close_managed_episode_caches)
