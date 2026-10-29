import sys
import time
from collections import OrderedDict


def register_omegaconf_resolvers():
    from omegaconf import ListConfig, OmegaConf

    OmegaConf.register_new_resolver(
        "eval",
        lambda expr: eval(expr, {"ListConfig": ListConfig, "list": list, "range": range}),
        replace=True,
    )


def estimate_item_bytes(value, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(value)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    if hasattr(value, "element_size") and hasattr(value, "nelement"):
        return int(value.element_size() * value.nelement())
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(
            estimate_item_bytes(k, seen) + estimate_item_bytes(v, seen)
            for k, v in value.items()
        )
    if isinstance(value, (list, tuple, set)):
        return sys.getsizeof(value) + sum(estimate_item_bytes(v, seen) for v in value)
    return sys.getsizeof(value)


class LocalEpisodeCacheBackend:
    def __init__(
        self,
        dataset,
        mode="lazy",
        max_episodes=None,
        report_interval=1000,
        in_worker_fn=None,
    ):
        self.dataset = dataset
        self.mode = mode
        self.max_episodes = max_episodes
        self.report_interval = report_interval
        self.in_worker_fn = in_worker_fn or (lambda: False)
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._reported_requests = 0

        if self.mode not in {"lazy", "eager"}:
            raise ValueError(f"Unsupported cache_mode: {self.mode}")
        if self.mode == "eager":
            self.preload()

    def capacity(self):
        if self.max_episodes is None:
            return None
        return max(0, int(self.max_episodes))

    def preload(self):
        capacity = self.capacity()
        episode_ids = list(self.dataset.selected_episodes)
        if capacity is not None:
            episode_ids = episode_ids[:capacity]
        start_time = time.monotonic()
        print(
            f"S1Dataset episode RAM cache eager fill start: "
            f"episodes={len(episode_ids)}/{len(self.dataset.selected_episodes)}",
            flush=True,
        )
        for count, ep_idx in enumerate(episode_ids, start=1):
            self.put(ep_idx, self.dataset._load_episode_to_ram(ep_idx))
            if self.report_interval and count % self.report_interval == 0:
                elapsed = time.monotonic() - start_time
                print(
                    f"S1Dataset episode RAM cache eager fill: {count}/{len(episode_ids)} episodes, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
        bytes_estimate = sum(estimate_item_bytes(item) for item in self.cache.values())
        elapsed = time.monotonic() - start_time
        print(
            f"S1Dataset episode RAM cache eager fill done: "
            f"cached={len(self.cache)}/{len(self.dataset.selected_episodes)}, "
            f"estimated_size={bytes_estimate / 1024**3:.2f} GiB, "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    def get(self, ep_idx):
        ep_idx = int(ep_idx)
        if ep_idx not in self.cache:
            if not self.in_worker_fn():
                self.misses += 1
            if self.in_worker_fn():
                return None
            episode = self.dataset._load_episode_to_ram(ep_idx)
            self.put(ep_idx, episode)
            self._maybe_report()
            return episode

        if self.in_worker_fn():
            return self.cache[ep_idx]
        self.hits += 1
        episode = self.cache.pop(ep_idx)
        self.cache[ep_idx] = episode
        self._maybe_report()
        return episode

    def put(self, ep_idx, episode):
        capacity = self.capacity()
        if capacity == 0:
            return
        ep_idx = int(ep_idx)
        if ep_idx in self.cache:
            self.cache.pop(ep_idx)
        self.cache[ep_idx] = episode
        while capacity is not None and len(self.cache) > capacity:
            self.cache.popitem(last=False)
            self.evictions += 1

    def stats(self):
        return {
            "backend": "local",
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }

    def _maybe_report(self):
        if not self.report_interval:
            return
        requests = self.hits + self.misses
        if requests <= 0 or requests == self._reported_requests:
            return
        if requests % self.report_interval != 0:
            return
        self._reported_requests = requests
        stats = self.stats()
        print(
            f"S1Dataset episode RAM cache: size={stats['size']}, "
            f"hits={stats['hits']}, misses={stats['misses']}, "
            f"evictions={stats['evictions']}",
            flush=True,
        )
