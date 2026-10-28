#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ "$#" -eq 1 ] || { echo "Usage: $0 /absolute/path/to/owned/robotwin-assets"; exit 1; }
exec "${ROBOTWIN_ENV:-$ROOT/.venv}/bin/python" "$ROOT/script/download_assets.py" --asset-root "$1"
