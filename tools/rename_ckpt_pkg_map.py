#!/usr/bin/env python3
"""Migrate checkpoint config metadata to the rtpolicy namespace.

Package and supported class names are updated in config.json. Weights are left
unchanged by default. The older vggt_encoder -> backbone weight-key migration is
available only with --rewrite-legacy-weights and creates a backup before writing.

Usage:
    python tools/rename_ckpt_pkg_map.py <checkpoint_dir>...
    python tools/rename_ckpt_pkg_map.py --root <directory> --dry-run
"""
from __future__ import annotations

import argparse
import json
import os

# Import the lightweight metadata helper even before editable installation.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rtpolicy.common.checkpoint import normalize_checkpoint_metadata


LEGACY_EMVIS_KEYS = {
    "vggt_target": "backbone_target",
    "vggt_model_path": "backbone_path",
    "load_vggt_pretrain": "load_backbone_pretrain",
    "vggt_config": "backbone_config",
    "vggt_train_backbone": "backbone_trainable",
}
UNSUPPORTED_POLICY_CLASSES = (
    "VODPPlusDitFlowMatchingNoAdapter",
    "RTPolicyDitFlowMatchingNoAdapter",
)


def _rename_legacy_keys(value):
    if isinstance(value, dict):
        return {
            LEGACY_EMVIS_KEYS.get(key, key): _rename_legacy_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_legacy_keys(item) for item in value]
    return value


def patch_config(path: str, dry: bool) -> bool:
    with open(path, encoding="utf-8") as handle:
        raw = handle.read()
    if any(class_name in raw for class_name in UNSUPPORTED_POLICY_CLASSES):
        print(f"skip unsupported checkpoint policy removed from RTPolicy: {path}")
        return False
    config = json.loads(raw)
    updated = _rename_legacy_keys(normalize_checkpoint_metadata(config))
    if updated == config:
        return False
    if not dry:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(updated, handle, indent=2)
            handle.write("\n")
    print(("would patch " if dry else "patched ") + path)
    return True


WEIGHT_PREFIX_MAP = [
    ("vggt_encoder.", "backbone."),
]


def patch_weights(ckpt_dir: str, dry: bool) -> bool:
    """state_dict prefix remap (vggt_encoder. -> backbone.): rewrites model.safetensors in place."""
    import shutil
    from safetensors.torch import load_file, save_file
    wt = os.path.join(ckpt_dir, "model.safetensors")
    if not os.path.isfile(wt):
        return False
    sd = load_file(wt)
    sd2 = {}
    changed = False
    for k, v in sd.items():
        k2 = k
        for old, new in WEIGHT_PREFIX_MAP:
            k2 = k2.replace(old, new)
        if k2 != k:
            changed = True
        sd2[k2] = v
    if not changed:
        return False
    if not dry:
        shutil.copy(wt, wt + ".bak_vodpp_rename")
        save_file(sd2, wt)
    print(("would rewrite " if dry else "rewrote ") + wt)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="*", help="checkpoint dirs (must contain config.json)")
    ap.add_argument("--root", help="recursively find config.json under this dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rewrite-legacy-weights", action="store_true",
                    help="also migrate historical vggt_encoder tensor keys; writes a backup")
    args = ap.parse_args()

    targets = []
    for d in args.dirs:
        cfg = os.path.join(d, "config.json")
        if os.path.isfile(cfg):
            targets.append(cfg)
        else:
            print(f"skip (no config.json): {d}")
    if args.root:
        for cur, _, files in os.walk(args.root):
            if "config.json" in files:
                targets.append(os.path.join(cur, "config.json"))

    n = sum(patch_config(t, args.dry_run) for t in targets)
    print(f"\n{n}/{len(targets)} configs patched")
    if args.rewrite_legacy_weights:
        dirs = {os.path.dirname(t) for t in targets}
        m = sum(patch_weights(d, args.dry_run) for d in dirs)
        print(f"{m}/{len(dirs)} weight files rewritten")


if __name__ == "__main__":
    main()
