#!/usr/bin/env python3
"""Merge two structurally identical lerobot v2.1 video datasets (videos/ layout).

Only rewrites meta + data parquet (episode_index/index/task remapping); video
mp4s are copied/linked into new chunk dirs under the new episode numbers.
Sources are read-only.

Usage: python tools/merge_lerobot_videos.py --out <dst> --src <rootA> --src <rootB>
(episode order = the order of --src on the command line)
"""
import argparse, json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BOOKKEEP = ("timestamp", "frame_index", "episode_index", "index", "task_index")

def load_eps(root):
    return {int(json.loads(l)["episode_index"]): json.loads(l)
            for l in open(root / "meta/episodes.jsonl")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", action="append", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    assert not out.exists(), f"refusing: {out} exists"

    roots = [Path(s) for s in args.src]
    info0 = json.load(open(roots[0] / "meta/info.json"))
    chunks_size = info0["chunks_size"]
    feats = info0["features"]

    # Merge and remap the task table
    global_tasks, remaps = [], {}
    for r in roots:
        m = {}
        for l in open(r / "meta/tasks.jsonl"):
            t = json.loads(l)["task"]
            if t not in global_tasks:
                global_tasks.append(t)
            m[json.loads(l)["task_index"]] = global_tasks.index(t)
        remaps[str(r)] = m

    (out / "meta").mkdir(parents=True)
    (out / "data").mkdir(parents=True)
    (out / "videos").mkdir(parents=True)

    new_eps, new_stats = [], []
    gidx = 0
    nidx = 0
    for r in roots:
        eps = load_eps(r)
        for old_idx in sorted(eps):
            pf = r / f"data/chunk-{old_idx // chunks_size:03d}/episode_{old_idx:06d}.parquet"
            df = pq.read_table(pf).to_pandas()
            schema_meta = pq.read_schema(pf).metadata
            n = len(df)
            df = df.copy()
            df["episode_index"] = nidx
            df["index"] = np.arange(gidx, gidx + n)
            df["frame_index"] = np.arange(n)
            m = remaps[str(r)]
            df["task_index"] = df["task_index"].map(lambda t: m[int(t)])
            table = pa.Table.from_pandas(df, preserve_index=False)
            if schema_meta and b"huggingface" in schema_meta:
                table = table.replace_schema_metadata({b"huggingface": schema_meta[b"huggingface"]})
            dchunk = out / f"data/chunk-{nidx // chunks_size:03d}"
            dchunk.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, dchunk / f"episode_{nidx:06d}.parquet")

            # Move videos (into the same chunk, under the new episode number)
            vchunk = out / f"videos/chunk-{nidx // chunks_size:03d}"
            for key in feats:
                if feats[key].get("dtype") != "video":
                    continue
                src_v = r / f"videos/chunk-{old_idx // chunks_size:03d}/{key}/episode_{old_idx:06d}.mp4"
                dst_v = vchunk / key
                dst_v.mkdir(parents=True, exist_ok=True)
                if src_v.exists():
                    (dst_v / f"episode_{nidx:06d}.mp4").write_bytes(src_v.read_bytes())  # EFS does not support rename; read/write is safe

            # meta
            old_ep = eps[old_idx]
            new_eps.append({"episode_index": nidx, "tasks": old_ep["tasks"], "length": n})
            # stats: reuse the source row, just change episode_index
            stats_line = [l for l in open(r / "meta/episodes_stats.jsonl")
                          if json.loads(l)["episode_index"] == old_idx]
            if stats_line:
                d = json.loads(stats_line[0])
                d["episode_index"] = nidx
                new_stats.append(d)
            gidx += n
            nidx += 1
            if nidx % 20 == 0:
                print(f"  {nidx} episodes merged", flush=True)

    (out / "meta/info.json").write_text(json.dumps(
        {**info0, "total_episodes": nidx, "total_frames": gidx}, indent=4))
    with open(out / "meta/episodes.jsonl", "w") as f:
        for e in new_eps:
            f.write(json.dumps(e) + "\n")
    with open(out / "meta/episodes_stats.jsonl", "w") as f:
        for s in new_stats:
            f.write(json.dumps(s) + "\n")
    with open(out / "meta/tasks.jsonl", "w") as f:
        for i, t in enumerate(global_tasks):
            f.write(json.dumps({"task_index": i, "task": t}) + "\n")
    print(f"DONE: {out} — {nidx} episodes, {gidx} frames")

if __name__ == "__main__":
    main()
