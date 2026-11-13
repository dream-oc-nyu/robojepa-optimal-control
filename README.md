# Action Optimization in Planning with Learned World Models: A Study on RoboJEPA

ROB-GY 6323 group project, Fall 2026: Jetasri Gupta, Ashesh Kaji, Yazan Matarweh, Siddarth Nilol, Mingyuan Yan (alphabetical by last name).

RoboJEPA ([paper](https://arxiv.org/abs/2610.10515), [website](https://robojepa.github.io/)) plans by solving an
optimization problem at every step: find the actions whose predicted latent comes closest to the goal,

$$
\min_{u_0, \dots, u_{N-1}} J(u) = \text{dist}(z_N,\ z_\text{goal}), \qquad z_{n+1} = P(z_{0:n},\ u_{0:n}), \qquad z_0 = E(o_t),
$$

with the encoder $E$ and the predictor $P$ fixed. The paper solves it with CEM. We study inference only: how the
choice of optimizer shapes the decisions and what they cost, across tasks and predictor sizes. This repository runs
an optimizer of choice in the four RoboCasa tasks of the paper (Reach, ObjectReach, ObstacleReach, ObjectPush) and
logs every planning step's cost (predictor rollouts, wall time) next to the task's reward.

## Built on

The official RoboJEPA code, [facebookresearch/robo_jepa](https://github.com/facebookresearch/robo_jepa), provides
the world model, the simulated tasks and the planning loop. This repository swaps an optimizer in for its CEM when
it runs, so the official code stays unchanged and is not copied here.

## Layout

```
planners/   planners.py (the planners), run_client.py (plugs one into the client),
            summarize.py (tables from run folders), draw_setups.py (random setup seeds)
scripts/    run_task.sh (one task, one planner, some setups)
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

TASK=push PLANNER=cem bash scripts/run_task.sh 300m 300k 0 1 2
METRIC=final python planners/summarize.py closed $OUT/push
```

`TASK` is `reach`, `objreach`, `obstacle` or `push`, with the settings of the official `reproduce_*.sh` scripts.
The last arguments are setup seeds (start and goal); `SCENE` picks the kitchen. Results go to `$OUT/<task>/<run>/`.
`PAPER=1` uses the paper's settings (its Table 14) where they differ from the scripts, with results in
`$OUT/<task>_paper/<run>/`; the top of `scripts/run_task.sh` lists the differences.
A rerun of the same setup gives the same episode (deterministic kernels, each planning call seeded from its setup and
step); `PLAN_SEED_OFFSET=k` draws other samples on the same setup, `DETERMINISTIC=0` turns the kernels off.
`python planners/draw_setups.py <generator seed> <how many>` draws setup seeds that anyone can draw again.

## Planners

So far `PLANNER=cem`, the official CEM, the baseline. A new planner is a function with the same arguments and
return value as the official `cem()`, added to `PLANNERS` in `planners/planners.py`; the top of that file says how.

## Citation

If you use this code, please cite the RoboJEPA paper:

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
