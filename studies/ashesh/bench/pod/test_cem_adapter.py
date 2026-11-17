"""CPU boundary check: adapted CEM equals native two-view CEM at full settings."""
import contextlib
import gc
import io
import logging
import unittest
import numpy as np
import torch
from app.robo_jepa.mpc.mpc_utils import cem, l1
from bench.interface import PlanContext
from bench.harness import goal_cost
from planners.cem_release import CEMRelease


class State:
    features = torch.zeros(1, 1, 2, 1, 1, 3)
    def repeat(self, n): return self
    def to(self, device): return self


class NativeCEMCheck(unittest.TestCase):
    def test_full_native_search_matches_contract_adapter(self):
        logging.disable(logging.CRITICAL)
        # Test-only: exclude imported torch modules from native per-batch GC.
        # Neither search calls nor production timing use this acceleration.
        gc.freeze()
        self.addCleanup(gc.unfreeze)
        goal = torch.tensor([.03, -.01, .02, .01, .02, .01])
        calls = []
        def rollout(u):
            calls.append(len(u))
            return u.repeat(1, 1, 2)
        def step(u, history, **kwargs):
            return u[..., :3, None].reshape(len(u), 1, 1, 1, 1, 3).expand(-1, -1, 2, -1, -1, -1), history
        ctx = PlanContext(rollout, goal_cost(goal), goal, 1, .3, 1e9, 4)
        planner = CEMRelease(); planner.reset(2)
        with contextlib.redirect_stderr(io.StringIO()):
            adapted = planner.plan(ctx)
        np.random.seed(2)
        _, means, _ = cem(None, State(), goal.reshape(1, 1, 2, 1, 1, 3),
            rollout=1, cem_steps=10, samples=16, num_batches=64, cutoff=20,
            momentum_mean=0., momentum_std=.70, momentum_mean_gripper=.15,
            momentum_std_gripper=.45, robot_init_std=.3, gripper_init_std=.8,
            maxnorm=.3, maxgripper=.99, translation_only=True, use_gripper=False,
            view_weights=[.5, .5], objective=l1, step_func=step,
            device='cpu', use_cache=True, global_step=4, total_steps=5,
            world_size=1, rank=0, verbose=False)
        torch.testing.assert_close(adapted, means[-1][:, :3], rtol=1e-5, atol=1e-6)
        self.assertEqual(len(calls), 640)
        self.assertEqual(sum(calls), 10240)
        self.assertTrue(torch.isfinite(adapted).all())

if __name__ == '__main__': unittest.main()
