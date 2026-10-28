#!/usr/bin/env python3
"""Copy a real-robot dataset and physically rename keys to training-standard names
(zero code intrusion).

observation.state            -> state
observation.images.head_camera      -> head_cam
observation.images.left_wrist_camera  -> left_cam
observation.images.right_wrist_camera -> right_cam
Other keys (action/endpose/effort/velocity etc.) keep their names (training only
uses the 4 above + action).

Touched: meta/info.json, data/*.parquet column names, videos/ dir names.
Usage: python tools/rename_real_keys.py <src_root> <dst_root>
"""
import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

KEY_MAP = {
    "observation.state": "state",
    "observation.images.head_camera": "head_cam",
    "observation.images.left_wrist_camera": "left_cam",
    "observation.images.right_wrist_camera": "right_cam",
}


def main():
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2])
    assert src != dst
    if dst.exists():
        print(f"dst exists, refusing: {dst}")
        sys.exit(1)

    # 1) Copy the whole tree (EFS->EFS same-disk cp; hard links to save space do not work across dirs, so plain copy)
    print("copying dataset ...")
    shutil.copytree(src, dst)

    # 2) meta/info.json: feature key names
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    feats = info["features"]
    new_feats = {}
    for k, v in feats.items():
        new_feats[KEY_MAP.get(k, k)] = v
    info["features"] = new_feats
    info_path.write_text(json.dumps(info, indent=4))
    print("info.json keys renamed")

    # 3) videos dir names (EFS does not support rename: create the new dir and move files one by one)
    for old, new in KEY_MAP.items():
        old_dir = dst / "videos" / "chunk-000" / old
        new_dir = dst / "videos" / "chunk-000" / new
        if old_dir.exists():
            new_dir.mkdir(parents=True, exist_ok=True)
            for f in old_dir.iterdir():
                shutil.move(str(f), str(new_dir / f.name))
            old_dir.rmdir()
            print(f"video dir renamed: {old} -> {new}")

    # 4) meta/episodes_stats.jsonl: stats key names
    stats_path = dst / "meta" / "episodes_stats.jsonl"
    if stats_path.exists():
        lines = []
        for line in stats_path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            d["stats"] = {KEY_MAP.get(k, k): v for k, v in d["stats"].items()}
            lines.append(json.dumps(d))
        stats_path.write_text("\n".join(lines) + "\n")
        print("episodes_stats.jsonl keys renamed")

    # 5) meta/modality.json: video section key names
    mod_path = dst / "meta" / "modality.json"
    if mod_path.exists():
        m = json.loads(mod_path.read_text())
        if "video" in m and isinstance(m["video"], dict):
            m["video"] = {KEY_MAP.get(k, k): v for k, v in m["video"].items()}
        mod_path.write_text(json.dumps(m, indent=2))
        print("modality.json keys renamed")

    # 6) data/*.parquet column names
    n = 0
    for pqf in sorted((dst / "data").rglob("*.parquet")):
        t = pq.read_table(pqf)
        cols = {c: KEY_MAP.get(c, c) for c in t.column_names}
        if cols == dict(zip(t.column_names, t.column_names)):
            continue
        import pyarrow.parquet as pq_mod
        pq_mod.write_table(t.rename_columns([cols[c] for c in t.column_names]), pqf, compression="zstd")
        n += 1
    print(f"parquet columns renamed in {n} files")

    print("DONE:", dst)


if __name__ == "__main__":
    main()
