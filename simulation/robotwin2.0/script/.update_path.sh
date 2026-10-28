#!/bin/bash
# Auto-refresh embodiment curobo yml paths from *_tmp.yml templates.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
python3 - <<'PY'
import os, glob
assets_path = os.getcwd()
pattern = os.path.join(assets_path, "assets", "embodiments", "**", "*_tmp.yml")
for tmp_file in glob.glob(pattern, recursive=True):
    target = tmp_file.replace("_tmp.yml", ".yml")
    with open(tmp_file) as f:
        content = f.read()
    new = content.replace("${ASSETS_PATH}", assets_path).replace("$ASSETS_PATH", assets_path)
    with open(target, "w") as f:
        f.write(new)
print(f"[.update_path.sh] refreshed embodiment paths under {assets_path}")
PY
