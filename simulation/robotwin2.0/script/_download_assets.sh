#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/assets"
python _download.py
for archive in background_texture embodiments objects; do
  unzip -o "$archive.zip"
done
cd "$ROOT"
echo "Configuring Path ..."
python script/update_embodiment_config_path.py
