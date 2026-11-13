# Action Optimization in Planning with Learned World Models: A Study on RoboJEPA

ROB-GY 6323 group project, Fall 2026: Jetasri Gupta, Ashesh Kaji, Yazan Matarweh, Siddarth Nilol, Mingyuan Yan (alphabetical by last name).

RoboJEPA ([paper](https://arxiv.org/abs/2610.10515), [website](https://robojepa.github.io/)) plans by solving, at
every step,

$$
\min_{u_0, \dots, u_{N-1}} J(u) = \text{dist}(z_N,\ z_\text{goal}), \qquad z_{n+1} = P(z_{0:n},\ u_{0:n}), \qquad z_0 = E(o_t),
$$

with the encoder $E$ and the predictor $P$ fixed. The paper uses CEM. We keep the model fixed and change only the
optimizer, and compare how well each one plans and what it costs. This repository runs one optimizer in the paper's
four tasks (Reach, ObjectReach, ObstacleReach, ObjectPush) and logs the cost of every planning step (predictor
rollouts, wall time) next to the task's reward.

## Built on

The official code, [facebookresearch/robo_jepa](https://github.com/facebookresearch/robo_jepa): the world model, the
tasks and the planning loop. It is used unchanged and not copied here; this repository only swaps its CEM for another
optimizer.

## Layout

```
planners/
  planners.py      the planners
  run_client.py    runs the official client with the chosen planner
  summarize.py     tables from run folders
  draw_setups.py   random setup seeds
scripts/
  run_task.sh      one task, one planner, a list of setups
```

## Setup

1. Clone `facebookresearch/robo_jepa` and set up its environment with its `scripts/setup_env.sh`
2. Download the V-JEPA 2.1 encoder and a predictor (links in the official README), and arrange them as

```
$MODELS/wm-300m/cooldown_300m_300k.pth.tar
$MODELS/wm-300m/config.json        # from robo_jepa/configs/released/wm-300m/
$MODELS/wm-300m/encoder.pth.tar    # the encoder (a link is fine)
```

## Run

```bash
export REPO=~/robo_jepa MODELS=~/checkpoints OUT=~/outputs

TASK=push PLANNER=cem bash scripts/run_task.sh 300m 300k 0 1 2   # predictor size, checkpoint step, setups
METRIC=final python planners/summarize.py closed $OUT/push      # one row per run
```

| Setting | |
|---|---|
| `TASK` | `reach`, `objreach`, `obstacle` or `push`, with the settings of the official `reproduce_*.sh` |
| `PAPER=1` | the paper's settings (its Table 14) instead; the differences are listed at the top of `scripts/run_task.sh` |
| `SCENE` | the kitchen |
| `PLANNER` | the optimizer (see Planners) |
| `PLAN_SEED_OFFSET=k` | other random samples on the same setup |
| `DETERMINISTIC=0` | turns off the deterministic GPU kernels |
| `METRIC=final` | `summarize.py` scores by the final reward instead of the paper's NCR |

- Results: `$OUT/<task>/<run>/`, or `$OUT/<task>_paper/<run>/` with `PAPER=1`
- The same setup, run again, gives the same episode
- New setup seeds: `python planners/draw_setups.py <generator seed> <how many>`

## Example: CEM with 1024 samples against 32

The comparison in our preliminary report: ObjectReach, the paper's settings, the 300M predictor, 20 setups.

```bash
OBJS=(banana can mug)  # the paper's three objects, in turn
for s in 375 382 391 442 475 518 527 593 712 733 736 772 777 780 808 887 930 940 981 991; do
  O=${OBJS[$((s % 3))]}  # one process per setup; SCENE = the setup picks the kitchen
  PAPER=1 TASK=objreach SCENE=$s OBJECT=$O TAG=cem1024 bash scripts/run_task.sh 300m 300k $s
  PAPER=1 TASK=objreach SCENE=$s OBJECT=$O TAG=cem32 SAMPLES=32 NUM_BATCHES=1 CUTOFF=4 \
    bash scripts/run_task.sh 300m 300k $s
done
python planners/summarize.py closed $OUT/objreach_paper cem1024 cem32
```

Time per setup on one GPU: about 30 min with 1024 samples, a few minutes with 32.

## Planners

| `PLANNER` | |
|---|---|
| `cem` | the official CEM, the baseline |

A new planner is a function with the same inputs and outputs as the official `cem()`, added to `PLANNERS` in
`planners/planners.py` (the top of that file says how).

## Reference

```bibtex
@article{zholus2026robojepa,
  title  = {RoboJEPA: Scaling Robotic Latent World Models},
  author = {Zholus, Artem and Beltran-Velez, Nicolas and Yuan, Jianhao and Chandar, Sarath and
            Nagarajan, Tushar and Severo, Daniel and Sinha, Koustuv and Drozdzal, Michal and
            Romero Soriano, Adriana and Bohg, Jeannette and Ballas, Nicolas and Assran, Mahmoud},
  journal = {arXiv preprint arXiv:2610.10515},
  year   = {2026}
}
```

## License

CC BY-NC-SA 4.0, the same as the official RoboJEPA code (see `LICENSE`).
