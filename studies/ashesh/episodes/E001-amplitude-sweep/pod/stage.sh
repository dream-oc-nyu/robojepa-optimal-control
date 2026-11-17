#!/usr/bin/env bash
# Reentrant, session-contained E001 driver. Run via nohup; never source lab envs.
set -euo pipefail
export SESSION_ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ "$SESSION_ROOT" == /workspace/robojepa-E001-* ]] || exit 2
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
mkdir -p home tmp cache artifacts logs markers upstream weights results
exec 9>driver.lock
flock -n 9 || { printf 'Driver already running\n'; exit 3; }
printf '%s\n' "$$" > driver.pid
printf '%s driver %s\n' "$(date -u +%FT%TZ)" "$$" >> pids.log
run() { "$@" & local p=$!; printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$p" "$*" >> pids.log; wait "$p"; }
idle() { nvidia-smi > "logs/gpu-before-$1.txt"; [[ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')" == 0 ]] || { printf 'GPU not idle\n'; return 1; }; }
fetch() { local url=$1 target=$2; if [[ ! -f "$target" ]]; then run curl -fL --retry 4 --retry-delay 5 -C - "$url" -o "$target.part"; mv "$target.part" "$target"; fi; }
stage_env() {
 if [[ ! -x env/bin/python ]]; then run /workspace/bin/micromamba create -y -p "$SESSION_ROOT/env" -c conda-forge python=3.11 pip; fi
 run python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
 run python -m pip install numpy==2.2.5 scipy==1.15.3 einops==0.8.2 PyYAML==6.0.3 wandb==0.31.0 pytest==9.2.0 timm==1.0.30
 for spec in 'facebookresearch/robo_jepa f93106f5f4a451fd526d51778aa8ab7c852eb824 robo_jepa' 'robocasa/robocasa 456174f62b89b8fca99eaaf33949c29fec9cfc2a robocasa' 'ARISE-Initiative/robosuite 5ce6643f3092639d08f7b0f90ed1c6a84f50552c robosuite' 'facebookresearch/vjepa2 204698b45b3712590f06245fbfba32d3be539812 vjepa2'; do
  read -r repo rev name <<< "$spec"
  fetch "https://codeload.github.com/$repo/tar.gz/$rev" "upstream/$name.tar.gz"
  if [[ ! -d upstream/$name-$rev ]]; then tar -xzf "upstream/$name.tar.gz" -C upstream; fi
 done
 sha256sum upstream/*.tar.gz > artifacts/source-sha256.txt
 run python -m pip install --no-deps -e upstream/robosuite-5ce6643f3092639d08f7b0f90ed1c6a84f50552c -e upstream/robocasa-456174f62b89b8fca99eaaf33949c29fec9cfc2a
 run python -m pip install mujoco==3.3.1 numba==0.61.2 opencv-python-headless==5.0.0.93 termcolor==3.3.0 imageio==2.38.1 h5py==3.16.0 lxml==6.1.3 gymnasium==0.29.1 qpsolvers==4.13.0 quadprog==0.1.14
 run /workspace/bin/micromamba install -y -p "$SESSION_ROOT/env" -c conda-forge libegl libgl libglx mesalib
 python -m pip freeze > artifacts/pip-freeze.txt
 /workspace/bin/micromamba list -p "$SESSION_ROOT/env" --json > artifacts/conda-packages.json
 cp upstream/robo_jepa-f93106f5f4a451fd526d51778aa8ab7c852eb824/configs/released/wm-22m/config.json weights/config.json
}
stage_weights() {
 # dl.fbaipublicfiles.com TLS expired 2026-11-11; same official S3 bucket,
 # certificate-verified regional path endpoint; published hashes still required.
 fetch https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/robojepa/wm/wm-22m/cooldown_22m_150k.pth.tar weights/predictor.pth.tar
 fetch https://s3.us-west-2.amazonaws.com/dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt weights/encoder.pt
 printf '%s\n' 'bdd77b212d81e11438fd6bd85525bed72ff1f47a6b3211ac925ddfbc7c8351a6  weights/predictor.pth.tar' '7aae1a3c7a31d258af9c985388b5d2f20587469380f2e11b54d7876ac8cfe58a  weights/encoder.pt' | sha256sum -c - | tee artifacts/weight-verification.txt
 sha256sum weights/* > artifacts/weight-sha256.txt
}
stage_assets() { run python stage_scene_assets.py; idle assets; run python discover_scene_assets.py; }
stage_smoke() { idle smoke; run python collect.py --smoke; }
stage_collect() { idle collect; run python collect.py --collect --cap-seconds 6000; }
trap 'code=$?; printf "%s exit=%s stage=%s\n" "$(date -u +%FT%TZ)" "$code" "${stage:-startup}" >> logs/exit.log' EXIT
for stage in env weights assets smoke collect; do
 if [[ -f markers/$stage.done ]]; then continue; fi
 printf '%s\n' "$stage" > current-stage
 printf '%s START %s\n' "$(date -u +%FT%TZ)" "$stage" >> logs/timings.log
 stage_$stage > "logs/$stage.log" 2>&1
 date -u +%FT%TZ > "markers/$stage.done"
 printf '%s END %s\n' "$(date -u +%FT%TZ)" "$stage" >> logs/timings.log
done
printf 'complete\n' > current-stage
