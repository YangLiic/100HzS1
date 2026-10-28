"""Download official RoboTwin archives to owned storage, then link them."""

import argparse
from pathlib import Path
import tempfile
import zipfile
from huggingface_hub import hf_hub_download
from link_assets import link_assets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()
    storage = args.asset_root.resolve()
    storage.mkdir(parents=True, exist_ok=True)
    for name in ("background_texture", "embodiments", "objects"):
        destination = storage / name
        if destination.is_dir():
            continue
        if destination.exists():
            raise FileExistsError(destination)
        archive = hf_hub_download(
            repo_id="TianxingChen/RoboTwin2.0",
            repo_type="dataset",
            filename=f"{name}.zip",
            local_dir=storage / ".downloads",
        )
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=storage) as temp:
            staging = Path(temp).resolve()
            with zipfile.ZipFile(archive) as bundle:
                for entry in bundle.infolist():
                    if not (staging / entry.filename).resolve().is_relative_to(staging):
                        raise ValueError(f"Unsafe archive member: {entry.filename}")
                bundle.extractall(staging)
            extracted = staging / name
            if not extracted.is_dir():
                raise RuntimeError(f"Unexpected archive layout: {name}.zip")
            extracted.rename(destination)
    link_assets(storage, Path(__file__).resolve().parents[1] / "assets")


if __name__ == "__main__":
    main()
