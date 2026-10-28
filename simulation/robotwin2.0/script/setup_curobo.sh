#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROBOTWIN_ENV:-$ROOT/.venv}/bin/python"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"
[ -n "$UV_BIN" ] || UV_BIN="$HOME/.local/bin/uv"
SOURCE="$ROOT/envs/curobo"
REVISION=d64c4b005459db10c5dd867d8b30a87d5bda9bdb
export GIT_LFS_SKIP_SMUDGE=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.cache/uv}"
export TMPDIR="${VFA_SETUP_TMPDIR:-$ROOT/.cache/tmp}"
mkdir -p "$TMPDIR"
if [ ! -d "$SOURCE/.git" ]; then
  [ ! -e "$SOURCE" ] || { echo "Refusing to replace $SOURCE"; exit 1; }
  git init "$SOURCE"
  git -C "$SOURCE" remote add origin https://github.com/NVlabs/curobo.git
  git -C "$SOURCE" fetch --depth 1 origin "$REVISION"
  git -C "$SOURCE" checkout --detach FETCH_HEAD
fi
[ "$(git -C "$SOURCE" rev-parse HEAD)" = "$REVISION" ] || { echo "Unexpected cuRobo revision"; exit 1; }
"$UV_BIN" pip install --python "$PYTHON" --no-deps --no-build-isolation -e "$SOURCE"
"$UV_BIN" pip check --python "$PYTHON"
"$PYTHON" -c 'import torch,curobo; from curobo.wrap.reacher.motion_gen import MotionGen; print("cuRobo ready",curobo.__file__,torch.__version__)'
