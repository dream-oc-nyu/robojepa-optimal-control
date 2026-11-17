#!/usr/bin/env bash
# Source this for isolated manual inspection/resume commands.
export SESSION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ "$SESSION_ROOT" == /workspace/robojepa-E001-* ]] || return 2
export HOME="$SESSION_ROOT/home" TMPDIR="$SESSION_ROOT/tmp"
export XDG_CACHE_HOME="$SESSION_ROOT/cache" HF_HOME="$SESSION_ROOT/cache/hf"
export TORCH_HOME="$SESSION_ROOT/cache/torch" UV_CACHE_DIR="$SESSION_ROOT/cache/uv"
export PIP_CACHE_DIR="$SESSION_ROOT/cache/pip" MAMBA_ROOT_PREFIX="$SESSION_ROOT/mamba"
export CONDA_PKGS_DIRS="$SESSION_ROOT/cache/conda" CUDA_CACHE_PATH="$SESSION_ROOT/cache/cuda"
export TRITON_CACHE_DIR="$SESSION_ROOT/cache/triton" TORCHINDUCTOR_CACHE_DIR="$SESSION_ROOT/cache/inductor"
export PYTHONPYCACHEPREFIX="$SESSION_ROOT/cache/pycache"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONHASHSEED=0
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PATH="$SESSION_ROOT/env/bin:$PATH"
export LD_LIBRARY_PATH="$SESSION_ROOT/env/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$SESSION_ROOT:$SESSION_ROOT/upstream/robo_jepa-f93106f5f4a451fd526d51778aa8ab7c852eb824"
export VJEPA2_HUB_REPO="$SESSION_ROOT/upstream/vjepa2-204698b45b3712590f06245fbfba32d3be539812"
cd "$SESSION_ROOT"
