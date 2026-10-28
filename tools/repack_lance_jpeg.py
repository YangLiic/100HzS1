#!/usr/bin/env python3
"""Repack a JPEG Lance frame table into blob columns without re-encoding."""
import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile

import lance
import lancedb
import pyarrow as pa


def repack_dataset(root, output, *, table_name=None, batch_size=64, staging_dir=None, dry_run=False):
    root, output = Path(root).expanduser().resolve(), Path(output).expanduser().resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Source and output must be separate directories and must not contain one another")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    manifest_path = root / "lance_format.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if table_name is None:
        table_name = manifest.get("table_name")
    if table_name is None:
        candidates = sorted(p.stem for p in root.glob("*.lance") if not p.stem.endswith("_videos"))
        if len(candidates) != 1:
            raise ValueError("Set table_name when the frames table is ambiguous")
        table_name = candidates[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]+", table_name) or table_name.endswith("_videos"):
        raise ValueError("Invalid frames table_name")
    info = json.loads((root / "meta/info.json").read_text())
    dataset = lance.dataset(root / f"{table_name}.lance")
    cameras = [key if key in dataset.schema.names else key.replace(".", "_")
               for key, feature in info["features"].items() if feature["dtype"] in ("image", "video")]
    if not cameras or any(name not in dataset.schema.names for name in cameras):
        raise ValueError("Source must contain JPEG camera columns")
    if any(not (pa.types.is_binary(dataset.schema.field(name).type) or pa.types.is_large_binary(dataset.schema.field(name).type)) for name in cameras):
        raise ValueError("Camera columns must contain encoded JPEG bytes")
    count = dataset.count_rows()
    if count != info["total_frames"]:
        raise ValueError("Source row count differs from metadata")
    schema = pa.schema([
        pa.field(field.name, pa.large_binary(), metadata={b"lance-encoding:blob": b"true"})
        if field.name in cameras else field for field in dataset.schema
    ], metadata=dataset.schema.metadata)
    if dry_run:
        return output
    if staging_dir is not None:
        staging_dir = Path(staging_dir).expanduser().resolve()
        if staging_dir == root or root in staging_dir.parents or staging_dir == output or output in staging_dir.parents:
            raise ValueError("Staging directory must be outside source and output")
        staging_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lance-jpeg-blob-", dir=staging_dir) as work:
        staged = Path(work) / "dataset"
        rows = 0

        def batches():
            nonlocal rows
            for batch in dataset.to_batches(batch_size=batch_size, scan_in_order=True, blob_handling="all_binary"):
                if not batch.num_rows:
                    continue
                for name in cameras:
                    for scalar in batch.column(batch.schema.get_field_index(name)):
                        value = scalar.as_py()
                        if not value or not value.startswith(b"\xff\xd8"):
                            raise ValueError(f"Expected JPEG bytes in {name}")
                rows += batch.num_rows
                yield pa.RecordBatch.from_arrays(
                    [batch.column(i).cast(schema.field(i).type) for i in range(len(schema))], schema=schema,
                )

        table = lancedb.connect(str(staged)).create_table(
            table_name, data=pa.RecordBatchReader.from_batches(schema, batches()), schema=schema,
        )
        if rows != count or table.count_rows() != count:
            raise ValueError("Repacked row count differs from source")
        shutil.copytree(root / "meta", staged / "meta")
        manifest.update(storage_format="jpeg", table_name=table_name, jpeg_storage="blob",
                        total_frames=count, total_episodes=info["total_episodes"],
                        repacked_from=str(root), jpeg_reencoded=False)
        (staged / "lance_format.json").write_text(json.dumps(manifest, indent=2) + "\n")
        shutil.copytree(staged, output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table-name")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--staging-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        print(repack_dataset(**vars(args)))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
