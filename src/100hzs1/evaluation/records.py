"""Durable per-worker journals and one atomic, live evaluation summary."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def append_event(path, event, **values):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(dict(event=event, at=now(), **values), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_events(path):
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError:
        return []
    # A worker may be appending its last line. Never consume an incomplete row.
    return [json.loads(line) for line in data.split(b"\n")[:-1] if line.strip()]


def collect(output, groups, gpus, require_complete=False):
    """Reject duplicates, wrong assignments and incomplete 'successful' exits."""
    output = Path(output)
    records, workers = [], {}
    for gpu, expected in zip(gpus, groups):
        rows = read_events(output / "logs" / f"gpu{gpu}.jsonl")
        local = []
        state = dict(status="starting", planned=len(expected), completed=0, successes=0)
        completed = False
        for event in rows:
            kind = event["event"]
            if completed:
                raise ValueError(f"GPU{gpu}: events after completion")
            state["updated_at"] = event["at"]
            if kind == "runtime":
                state.update(status="running", runtime=event["runtime"])
                state.pop("error", None)
            elif kind == "started":
                if event["seed"] not in expected:
                    raise ValueError(f"GPU{gpu}: unexpected seed")
                state.update(current_seed=event["seed"], status="running")
            elif kind == "episode":
                record = event["record"]
                if type(record.get("success")) is not bool:
                    raise ValueError("Every episode needs a boolean outcome")
                if record["seed"] not in expected or record["seed"] in [r["seed"] for r in local]:
                    raise ValueError(f"GPU{gpu}: duplicate or unexpected seed")
                if record["seed"] != expected[len(local)]:
                    raise ValueError(f"GPU{gpu}: episode order changed")
                local.append(dict(record, gpu=gpu))
                state["current_seed"] = None
            elif kind == "completed":
                if [r["seed"] for r in local] != expected:
                    raise RuntimeError(f"GPU{gpu}: completion with missing seeds")
                completed = True
                state.update(status="completed", current_seed=None, **event.get("graph", {}))
            elif kind == "failed":
                state.update(status="failed", error=event["error"])
            elif kind == "retry":
                state.update(status="retrying", restarts=event["attempt"])
                state.pop("error", None)
            elif kind == "resume":
                state.update(status="starting", resumed_from=event["source"])
            else:
                raise ValueError(f"Unknown worker event: {kind}")
        state.update(completed=len(local), successes=sum(r["success"] for r in local))
        if require_complete and not completed:
            raise RuntimeError(f"GPU{gpu} did not complete its exact seed assignment")
        if require_complete:
            for record in local:
                if record.get("video"):
                    video = output / record["video"]
                    if not video.is_file() or video.stat().st_size == 0:
                        raise RuntimeError(f"Missing episode video: {video}")
        records.extend(local)
        workers[str(gpu)] = state
    expected_all = [s for group in groups for s in group]
    if len(set(expected_all)) != len(expected_all) or len({r["seed"] for r in records}) != len(records):
        raise ValueError("Overlapping seed assignments/results")
    records.sort(key=lambda row: expected_all.index(row["seed"]))
    return records, workers


def snapshot(manifest, output, groups, gpus, status="running", error=None, final=False):
    records, workers = collect(output, groups, gpus, require_complete=final)
    successes = sum(r["success"] for r in records)
    value = dict(manifest, status=status, updated_at=now(),
                 planned_episodes=sum(map(len, groups)), completed_episodes=len(records),
                 successes=successes, failures=len(records)-successes,
                 success_rate=successes/len(records) if records else None,
                 success_rate_denominator="completed_episodes",
                 workers=workers, results=records)
    if error is not None:
        value["error"] = str(error)
    if final:
        value["finished_at"] = now()
    atomic_json(Path(output) / "result.json", value)
    return value

def restore_progress(source, output, manifest, groups, gpus):
    """Copy completed episodes into a fresh run; never reevaluate or replace their outcomes."""
    import shutil
    source, output = Path(source).resolve(), Path(output).resolve()
    old = json.loads((source / "result.json").read_text())
    if source == output or old["status"] not in {"failed", "interrupted"}:
        raise ValueError("Resume requires a failed/interrupted source and a fresh destination")
    for key in ("checkpoint", "checkpoint_config_sha256", "task", "task_config", "seed_groups",
                "gpus", "nfe", "use_cuda_graph", "use_kv_cache", "precision", "channels"):
        if old.get(key) != manifest.get(key):
            raise ValueError(f"Cannot resume after changing {key}")
    if old.get("backend", "eager") != manifest.get("backend", "eager"):
        raise ValueError("Cannot resume after changing backend")
    rows, states = collect(source, groups, gpus)
    if rows != old["results"]:
        raise ValueError("Source summary and journals disagree")
    for gpu, seeds in zip(gpus, groups):
        local = [row for row in rows if row["gpu"] == gpu]
        journal = output / "logs" / f"gpu{gpu}.jsonl"
        append_event(journal, "resume", source=str(source))
        for row in local:
            row = dict(row)
            row.pop("gpu")
            if row.get("video"):
                expected = f"videos/seed_{row['seed']}.mp4"
                if row["video"] != expected:
                    raise ValueError("Unexpected source video path")
                src = source / expected
                if not src.is_file() or not src.stat().st_size:
                    raise ValueError(f"Missing source video: {src}")
                shutil.copy2(src, output / expected)
            append_event(journal, "episode", record=row)
        if len(local) == len(seeds):
            state = states[str(gpu)]
            append_event(journal, "runtime", runtime=state.get("runtime", {}))
            append_event(journal, "completed",
                         graph={k: state[k] for k in ("graph_captures", "graph_replays") if k in state})
    manifest.update(resumed_from=str(source), preserved_episodes=len(rows))
