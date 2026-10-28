"""Link shared RoboTwin assets without modifying their contents."""

import argparse
from pathlib import Path


def link_assets(source, destination):
    source, destination = Path(source).resolve(), Path(destination)
    for name in ("background_texture", "embodiments", "objects"):
        if not (source / name).is_dir():
            raise FileNotFoundError(source / name)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("background_texture", "embodiments", "objects", "files"):
        target, link = source / name, destination / name
        if not target.exists():
            continue
        if link.is_symlink() and link.resolve() == target:
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"Refusing to replace existing asset path: {link}")
        link.symlink_to(target, target_is_directory=True)
    print(f"Assets linked: {destination} -> {source}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()
    link_assets(args.asset_root, Path(__file__).resolve().parents[1] / "assets")
