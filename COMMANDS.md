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
METRIC=final python planners/summarize.py closed $OUT/reach
```

### Batch job

Save this as `job.sh` and submit it with `sbatch job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=robojepa
#SBATCH --account=torch_pr_564_tandon_advanced
#SBATCH --partition=h200_tandon,h200_public,h100_tandon
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.log
set -euo pipefail
source $SCRATCH/miniforge3/bin/activate
conda activate $SCRATCH/miniforge3/envs/robo_jepa
export REPO=$SCRATCH/robo_jepa MODELS=$SCRATCH/checkpoints OUT=$SCRATCH/outputs
export TORCH_HOME=$SCRATCH/.cache/torch XDG_CACHE_HOME=$SCRATCH/.cache
cd $SCRATCH/robojepa-optimal-control
nvidia-smi -L

# the README's CEM 1024 vs 32 comparison (ObjectReach, paper settings, 20 setups)
OBJS=(banana can mug)
for s in 375 382 391 442 475 518 527 593 712 733 736 772 777 780 808 887 930 940 981 991; do
  O=${OBJS[$((s % 3))]}
  PAPER=1 TASK=objreach SCENE=$s OBJECT=$O TAG=cem1024 bash scripts/run_task.sh 300m 300k $s
  PAPER=1 TASK=objreach SCENE=$s OBJECT=$O TAG=cem32 SAMPLES=32 NUM_BATCHES=1 CUTOFF=4 \
    bash scripts/run_task.sh 300m 300k $s
done
python planners/summarize.py closed $OUT/objreach_paper cem1024 cem32
```

CEM-1024 takes about 30 min per setup, so all 20 setups need around 10–11 h. To get results sooner, split the
seeds across several jobs: runs of the same setup give identical episodes, so the outputs can be combined.

Useful commands: `squeue -u $USER`, `scancel <jobid>`, `tail -f robojepa_<jobid>.log`, and
`$OUT/<task>/<run>/env_server.log` when the env server dies.

## Notes

- Use full GPUs. MIG slices can't do the EGL rendering RoboCasa needs (`run_task.sh` sets `MUJOCO_GL=egl`).
- The account and partitions are the ones from your `flow_control` jobs. Check them against what your
  allocation has for this course.
