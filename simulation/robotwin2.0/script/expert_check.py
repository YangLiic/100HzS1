"""Shared expert-only scene validity check; independent of learned policies."""

import time


def check_scene(environment, seed, index, args, attempts=3, unstable_errors=()):
    if attempts < 1:
        raise ValueError("attempts must be positive")
    records = []
    for attempt in range(1, attempts + 1):
        started = time.monotonic()
        info = None
        record = {"attempt": attempt, "plan_success": False, "task_success": False}
        try:
            environment.setup_demo(now_ep_num=index, seed=seed, is_test=True, **args)
            info = environment.play_once()
            record["plan_success"] = bool(environment.plan_success)
            record["task_success"] = bool(environment.check_success())
        except unstable_errors as error:
            record["scene_error"] = f"{type(error).__name__}: {error}"
        finally:
            environment.close_env(clear_cache=True)
        record["seconds"] = time.monotonic() - started
        record["success"] = record["plan_success"] and record["task_success"]
        records.append(record)
        if record["success"]:
            return info, records
        print(f"EXPERT_RETRY seed={seed} attempt={attempt}", flush=True)
    return None, records


def accepted_seeds(records, start_seed):
    seeds = []
    for index, record in enumerate(records):
        if record["seed"] != start_seed + index or type(record["success"]) is not bool:
            raise ValueError(
                "Scene records must be contiguous and have boolean results"
            )
        if record["success"]:
            seeds.append(record["seed"])
    return seeds
