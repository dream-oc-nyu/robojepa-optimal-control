"""Call pinned app.robo_jepa.mpc.mpc_utils.cem without altering its search.

Native settings: scripts/deployment/examples/reproduce_reach.sh:17,61-67.
The adapter supplies only the fixed contract's rollout and mean-L1 cost, as
one already-weighted view. Sampling/refitting/elite selection stay upstream.
"""
import numpy as np
import torch


class _ContractState:
    # CEM only reads the dimensions, repeats, and moves this placeholder;
    # the real history remains private inside ctx.rollout.
    def __init__(self):
        self.features = torch.empty(1, 1, 1, 1, 1, 1)
    def repeat(self, count):
        return self
    def to(self, device):
        return self


class CEMRelease:
    name = 'cem_release'

    def reset(self, seed):
        self.rng = np.random.RandomState(seed).get_state()

    def plan(self, ctx):
        from app.robo_jepa.mpc.mpc_utils import cem
        if ctx.horizon_max != 1:
            raise ValueError('reach-v0 release horizon is 1')
        def step(action, history, **kwargs):
            z = ctx.rollout(action[..., :3].to(ctx.goal.device))
            return z.reshape(len(action), 1, 1, 1, 1, -1), history
        def objective(unused_goal, features):
            return -ctx.cost(features.reshape(features.shape[0], 1, -1))[..., None]
        previous = None
        if ctx.warm_start is not None:
            previous = torch.nn.functional.pad(ctx.warm_start.detach().cpu(), (0, 4))
        outer_rng = np.random.get_state()
        np.random.set_state(self.rng)
        try:
            with torch.no_grad():
                _, means, _ = cem(None, _ContractState(), torch.zeros(1, 1, 1, 1, 1, 1),
                    rollout=1, cem_steps=15 if ctx.tick == 0 else 10,
                    action_dim=7, action_repeat=1, samples=16, num_batches=64, cutoff=20,
                    momentum_mean=0., momentum_std=.70,
                    momentum_mean_gripper=.15, momentum_std_gripper=.45,
                    robot_init_std=.3, gripper_init_std=.8, maxnorm=.3, maxgripper=.99,
                    translation_only=True, use_gripper=False, view_weights=[1.],
                    prev_actions=previous, objective=objective, step_func=step,
                    device=ctx.goal.device, use_cache=True, global_step=ctx.tick, total_steps=5,
                    world_size=1, rank=0, verbose=True)
            return means[-1][:, :3].to(ctx.goal.device)
        finally:
            self.rng = np.random.get_state()
            np.random.set_state(outer_rng)


Planner = CEMRelease
