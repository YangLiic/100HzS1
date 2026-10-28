"""Fetch locally pinned MIT EfficientViT weights from its timm integration."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download


def download(model_name, root, revision=None):
    repo_id = "timm/" + model_name
    revision = HfApi(token=False).model_info(repo_id, revision=revision).sha
    directory = root / model_name
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ["config.json", "README.md", "model.safetensors"]:
        print(f"Downloading {repo_id}@{revision}: {filename}", flush=True)
        hf_hub_download(repo_id, filename, revision=revision, local_dir=directory, token=False)
    path = directory / "model.safetensors"
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    manifest = dict(model_name=model_name, repo_id=repo_id, revision=revision,
                    sha256=digest.hexdigest(), bytes=path.stat().st_size,
                    original_project="https://github.com/mit-han-lab/efficientvit",
                    pretraining="ImageNet-1K supervised classification; paper authors, timm port")
    (directory / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["efficientvit_b2.r256_in1k", "efficientvit_b3.r256_in1k"])
    parser.add_argument("--revision", help="Commit/tag for one explicitly selected model")
    args = parser.parse_args()
    if args.revision and len(args.models) != 1:
        parser.error("--revision requires exactly one --models entry")
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda model: download(model, args.root, args.revision), args.models))
