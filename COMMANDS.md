# Commands: running on NYU HPC (Torch)

Steps 1–4 run once. Steps 5–6 run whenever you need them. `/home` has very little space, so everything goes
under `$SCRATCH` (`/scratch/<netid>`).

## 1. Clone both repositories

```bash
ssh <netid>@login.torch.hpc.nyu.edu      # or however you usually log in
cd $SCRATCH
git clone https://github.com/facebookresearch/robo_jepa.git
git clone <this repo's URL> robojepa-optimal-control
```

## 2. Environment and RoboCasa (official `setup_env.sh`)

This installs Miniforge and the `robo_jepa` conda env (Python 3.11, torch 2.7.1 / CUDA 12.6, RoboCasa,
robosuite, MuJoCo), all under `$SCRATCH`. `ROBOCASA_ASSETS=1` also downloads the kitchen assets (about 23 GB).

```bash
cd $SCRATCH/robo_jepa
ROBOCASA_ASSETS=1 bash scripts/setup_env.sh
```

It takes a while. If the login node kills it, run it in a CPU job instead (`srun --cpus-per-task=8 --mem=32G --time=02:00:00 --pty bash`).

Check it:

```bash
source $SCRATCH/miniforge3/bin/activate
conda activate $SCRATCH/miniforge3/envs/robo_jepa
python -c "import robocasa, robosuite, mujoco, torch; print(torch.__version__, mujoco.__version__)"
```

## 3. Checkpoints

```bash
export MODELS=$SCRATCH/checkpoints
mkdir -p $MODELS/wm-300m && cd $MODELS
wget https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt
wget -P wm-300m https://dl.fbaipublicfiles.com/robojepa/wm/wm-300m/cooldown_300m_300k.pth.tar
cp $SCRATCH/robo_jepa/configs/released/wm-300m/config.json wm-300m/
ln -s $MODELS/vjepa2_1_vitG_384.pt wm-300m/encoder.pth.tar
```

## 4. Cache the V-JEPA 2 hub code (compute nodes may have no internet)

The encoder's architecture loads through `torch.hub` (`facebookresearch/vjepa2`). `planners/run_client.py` resolves
it to `main` without the network, but the checkout must already be in the hub cache:

```bash
export TORCH_HOME=$SCRATCH/.cache/torch
mkdir -p $TORCH_HOME/hub
git clone https://github.com/facebookresearch/vjepa2.git $TORCH_HOME/hub/facebookresearch_vjepa2_main
```

## 5. Environment for every shell or job

```bash
source $SCRATCH/miniforge3/bin/activate
conda activate $SCRATCH/miniforge3/envs/robo_jepa
export REPO=$SCRATCH/robo_jepa MODELS=$SCRATCH/checkpoints OUT=$SCRATCH/outputs
export TORCH_HOME=$SCRATCH/.cache/torch XDG_CACHE_HOME=$SCRATCH/.cache
cd $SCRATCH/robojepa-optimal-control
```

## 6. Run

### Interactive (quick test on one setup)

```bash
srun --account=torch_pr_564_tandon_advanced --partition=h200_tandon,h200_public,h100_tandon \
     --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=01:00:00 --pty bash
# then the block from step 5, and:
nvidia-smi -L
TASK=reach PLANNER=cem SAMPLES=32 NUM_BATCHES=1 bash scripts/run_task.sh 300m 300k 0
TASK=reach PLANNER=adam bash scripts/run_task.sh 300m 300k 0     # the gradient planner (or PLANNER=lbfgs)
METRIC=final python planners/summarize.py closed $OUT/reach
```

### Batch job

Save this as `job.sh`. It runs one task with one planner on a list of setups. Settings go in as environment
variables and the arguments go to `run_task.sh` (`sbatch` passes your shell's environment to the job):

```bash
TASK=reach PLANNER=cem sbatch job.sh 300m 300k 0 1 2 3 4
```

```bash
#!/bin/bash
#SBATCH --job-name=robojepa
#SBATCH --account=torch_pr_564_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public,h100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.log
set -eo pipefail  # no -u: conda's activation scripts read unset variables
# conda's shell hook, not `source .../activate`: a sourced script gets the job's arguments and activate rejects them
eval "$($SCRATCH/miniforge3/bin/conda shell.bash hook)"
conda activate $SCRATCH/miniforge3/envs/robo_jepa
export REPO=$SCRATCH/robo_jepa MODELS=$SCRATCH/checkpoints OUT=$SCRATCH/outputs
export TORCH_HOME=$SCRATCH/.cache/torch XDG_CACHE_HOME=$SCRATCH/.cache
cd $SCRATCH/robojepa-optimal-control
nvidia-smi -L
bash scripts/run_task.sh "$@"
```

Rerunning a setup under the same tag appends to `plan_log.jsonl` and overwrites the episode file. `summarize.py`
counts only the last run of each setup, but you can also delete the folder first (`rm -rf $OUT/<task>/<tag>`).

With the default CEM (1024 samples) a setup takes about 30 min, so set `--time` to fit the number of setups, or split
them across several jobs. Runs of the same setup give identical episodes, so the outputs can be combined.
Then: `METRIC=final python planners/summarize.py closed $OUT/<task>`.

Useful commands: `squeue -u $USER`, `scancel <jobid>`, `tail -f robojepa_<jobid>.log`, and
`$OUT/<task>/<run>/env_server.log` when the env server dies.

## 7. Gradient planners (Adam, L-BFGS)

Both descend J by backpropagation through the predictor (`planners/planners.py`). Run folders are tagged
`<planner>_k<K>_it<iters>`, for example `adam_k8_it20` and `lbfgs_k8_it10`.

### First: one setup, interactively

The first run shows whether gradients survive fp16 and how much memory the planner needs. In a GPU session, after
the block from step 5:

```bash
TASK=reach PLANNER=adam bash scripts/run_task.sh 300m 300k 0
TASK=reach PLANNER=lbfgs bash scripts/run_task.sh 300m 300k 0

python - <<'EOF'
import json, os
for tag in ("adam_k8_it20", "lbfgs_k8_it10"):
    for line in open(os.path.expandvars(f"$OUT/reach/{tag}/plan_log.jsonl")):
        r = json.loads(line)
        print(tag, r["global_step"], f"J {r['J_warm']:.4f} -> {r['J_best']:.4f}", "nonfinite", r["nonfinite"],
              f"{r['seconds']:.1f}s", f"peak {r['gpu_mem_peak_gb']:.1f} GB")
EOF
```

What to check:

| You see | Do |
|---|---|
| `nonfinite` above 0 | gradients overflow or underflow in fp16: lower `PLAN_LOSS_SCALE` (default 4096) on overflow, or run in fp32 with `PLAN_FP32=1` |
| `J_best` equal to `J_warm` | no progress: raise `PLAN_LR` (Adam: 0.3), or check that `nonfinite` is 0 |
| CUDA out of memory | `PLAN_GRAD_CHUNK=2`, or `1` (candidates per forward+backward pass; default 4) |

### Batch jobs

The same `job.sh` as for CEM, with the planner's settings in front:

```bash
TASK=reach PLANNER=adam  sbatch job.sh 300m 300k 0 1 2 3 4
TASK=reach PLANNER=lbfgs sbatch job.sh 300m 300k 0 1 2 3 4

# other settings: more candidates or iterations, or another step size (each gets its own tag)
TASK=reach PLANNER=adam PLAN_K=16 PLAN_ITERS=40 sbatch job.sh 300m 300k 0 1 2 3 4
TASK=reach PLANNER=adam PLAN_LR=0.3 TAG=adam_k8_it20_lr0.3 sbatch job.sh 300m 300k 0 1 2 3 4
TASK=reach PLANNER=lbfgs PLAN_LOSS=l1 TAG=lbfgs_k8_it10_l1 sbatch job.sh 300m 300k 0 1 2 3 4

# the other tasks: horizon 5 instead of 1, so more memory; start with a smaller chunk
TASK=push PLANNER=adam PLAN_GRAD_CHUNK=2 sbatch job.sh 300m 300k 0 1 2 3 4
```

`PLAN_LR`, `PLAN_LOSS` and the other settings are not in the default tag, so give such runs their own `TAG`, or their results
land in the same folder as the default run's.

| Setting | Default | |
|---|---|---|
| `PLAN_K` | 8 | candidates: the warm start (the previous plan, shifted) and K-1 random perturbations of it |
| `PLAN_ITERS` | 20 Adam, 10 L-BFGS | iterations per planning call |
| `PLAN_LR` | 0.1 Adam, 1.0 L-BFGS | step size |
| `PLAN_INIT_STD` | the client's `robot_init_std` (0.3) | std of the perturbations |
| `PLAN_GRAD_CHUNK` | 4 | candidates per forward+backward pass |
| `PLAN_LOSS_SCALE` | 4096 | J is multiplied by this before the fp16 backward pass |
| `PLAN_FP32` | 0 | 1: gradient rollouts in fp32 |
| `PLAN_HISTORY` | 20 | L-BFGS history size |
| `PLAN_LOSS` | `l2` | the cost descended: `l2` (mean squared feature error) or `l1`. Plans are still chosen and reported by L1, as CEM's |

### Compare against CEM

```bash
METRIC=final python planners/summarize.py closed $OUT/reach cem_16x64_15-10 adam_k8_it20 lbfgs_k8_it10
```

This gives one row per planner: rollouts, backward passes (each costs about two forward ones), planning and episode
wall time, d_T and the reward. Compare only runs on the same setups.

## Notes

- Use full GPUs. MIG slices can't do the EGL rendering RoboCasa needs (`run_task.sh` sets `MUJOCO_GL=egl`).
- The account and partitions are the ones from your `flow_control` jobs. Check them against what your
  allocation has for this course.
