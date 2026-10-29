#!/usr/bin/env python3
"""Materialize a new LeRobot v2.1 dataset from subsets/merges of existing ones.

Read-only on sources. Re-indexes episodes (0..N-1) and the global frame index,
writes data/chunk-000 parquets + meta/{info.json, episodes.jsonl,
episodes_stats.jsonl, tasks.jsonl}. Images stay as parquet PNG bytes
(use_videos=False layout).

Example:
  python tools/assemble_lerobot_subset.py \
    --out /path/new_dataset \
    --src /path/dsA:0,2,4 --src /path/dsB:1,3
"""
from __future__ import annotations

import argparse, json, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BOOKKEEP = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def parse_src(spec: str):
    root, _, idx = spec.partition(":")
    indices = [int(x) for x in idx.split(",") if x != ""]
    return Path(root), indices


def episode_stats_recompute(df: pd.DataFrame) -> dict:
    """Recompute per-episode stats for bookkeeping columns from the new table."""
    stats = {}
    for col in BOOKKEEP:
        v = np.stack(df[col].values).astype(np.float64).reshape(len(df), -1)
        stats[col] = {
            "min": v.min(0).tolist(), "max": v.max(0).tolist(),
            "mean": v.mean(0).tolist(),
            "std": (v.std(0) if len(v) > 1 else np.zeros(v.shape[1])).tolist(),
            "count": [len(df)],
        }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", action="append", required=True,
                    help="root:idx1,idx2,... (repeatable; concatenated in order)")
    args = ap.parse_args()

    out = Path(args.out)
    assert not out.exists(), f"output exists, refusing to overwrite: {out}"

    # template meta from the first source
    src0, _ = parse_src(args.src[0])
    info = json.load(open(src0 / "meta/info.json"))
    chunks_size = info["chunks_size"]
    src_episodes = {
        int(json.loads(l)["episode_index"]): json.loads(l)
        for l in open(src0 / "meta/episodes.jsonl")
    }
    src_stats = {}
    for spec in args.src:
        root, _ = parse_src(spec)
        for l in open(root / "meta/episodes_stats.jsonl"):
            d = json.loads(l)
            src_stats[(str(root), int(d["episode_index"]))] = d["stats"]

    # Global task table across all sources (each source numbers its own tasks
    # from 0, so a merged dataset must remap task_index into a shared space).
    global_tasks: list = []
    src_task_remap: dict = {}
    for spec in args.src:
        root, _ = parse_src(spec)
        remap = {}
        for l in open(root / "meta/tasks.jsonl"):
            d = json.loads(l)
            if d["task"] not in global_tasks:
                global_tasks.append(d["task"])
            remap[int(d["task_index"])] = global_tasks.index(d["task"])
        src_task_remap[str(root)] = remap

    (out / "data").mkdir(parents=True)
    (out / "meta").mkdir(parents=True)

    new_episodes, new_stats = [], []
    gindex = 0
    new_idx = 0
    for spec in args.src:
        root, indices = parse_src(spec)
        for old_idx in indices:
            df = pq.read_table(root / f"data/chunk-000/episode_{old_idx:06d}.parquet").to_pandas()
            # The huggingface features metadata (declares Image dtype etc.) is
            # what lets lerobot/datasets decode image columns on read; a plain
            # pandas roundtrip drops it and training dies with
            # "Could not infer dtype of dict". Carry it over verbatim.
            src_meta = pq.read_schema(root / f"data/chunk-000/episode_{old_idx:06d}.parquet").metadata
            n = len(df)
            df = df.copy()
            df["episode_index"] = new_idx
            df["index"] = np.arange(gindex, gindex + n)
            df["frame_index"] = np.arange(n)
            remap = src_task_remap[str(root)]
            df["task_index"] = df["task_index"].map(lambda t: remap[int(t)])
            import pyarrow as pa
            table = pa.Table.from_pandas(df, preserve_index=False)
            if src_meta and b"huggingface" in src_meta:
                table = table.replace_schema_metadata({b"huggingface": src_meta[b"huggingface"]})
            ep_chunk = new_idx // chunks_size
            chunk_dir = out / f"data/chunk-{ep_chunk:03d}"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, chunk_dir / f"episode_{new_idx:06d}.parquet")

            ep_meta = src_episodes[old_idx] if str(root) == str(src0) else\
                json.loads([l for l in open(root / "meta/episodes.jsonl")
                            if json.loads(l)["episode_index"] == old_idx][0])
            new_episodes.append({"episode_index": new_idx,
                                 "tasks": ep_meta["tasks"], "length": n})
            st = dict(src_stats[(str(root), old_idx)])
            for k in BOOKKEEP:
                st.pop(k, None)
            st.update(episode_stats_recompute(df))
            new_stats.append({"episode_index": new_idx, "stats": st})

            gindex += n
            new_idx += 1

    with open(out / "meta/episodes.jsonl", "w") as f:
        for e in new_episodes:
            f.write(json.dumps(e) + "\n")
    with open(out / "meta/episodes_stats.jsonl", "w") as f:
        for s in new_stats:
            f.write(json.dumps(s) + "\n")
    with open(out / "meta/tasks.jsonl", "w") as f:
        for i, t in enumerate(global_tasks):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")

    info["total_episodes"] = new_idx
    info["total_frames"] = gindex
    info["total_videos"] = 0
    info["total_tasks"] = len(global_tasks)
    info["total_chunks"] = (new_idx + chunks_size - 1) // chunks_size
    info["splits"] = {"train": f"0:{new_idx}"}
    with open(out / "meta/info.json", "w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")

    print(f"wrote {out}: {new_idx} episodes, {gindex} frames")


if __name__ == "__main__":
    main()
