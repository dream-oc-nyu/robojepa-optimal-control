"""CPU tests for `adam` and `trgn` on the toy world. Run from the repo root:

    python -m unittest planners.test_planners -v
"""
from __future__ import annotations

import math
import time
import unittest

import torch

from bench.interface import Feedback, PlanContext
from planners.adam import AdamConfig, AdamPlanner
from planners.toy import ToyConfig, ToyWorld, simulate
from planners.trgn import (TRGNConfig, TRGNPlanner, fd_linearise, jvp_linearise, next_radius,
                           solve_trust_region)

BUDGET_S = 0.3
TICKS = 5
JVP_FIRST_CALL_S: list = []   # wall-clock of the untimed warm-up plan() calls in jvp mode


class RandomShootingPlanner:
    """Reference only: best of seeded uniform-in-ball samples (plus the null plan) under the model cost."""
    name = "random_shooting"

    def __init__(self, batch: int = 256):
        self.batch = batch

    def reset(self, seed: int) -> None:
        self._seed = seed

    def plan(self, ctx: PlanContext) -> torch.Tensor:
        t0 = time.perf_counter()
        H, g = ctx.horizon_max, ctx.goal
        gen = torch.Generator().manual_seed(self._seed * 7919 + ctx.seed * 31 + ctx.tick)
        best_u, best_c, dt = torch.zeros(H, 3, dtype=g.dtype), math.inf, 0.0
        first = True
        while time.perf_counter() + 1.5 * dt < t0 + 0.9 * ctx.budget_s:
            ts = time.perf_counter()
            dirs = torch.randn(self.batch, H, 3, generator=gen, dtype=g.dtype)
            radii = torch.rand(self.batch, H, 1, generator=gen, dtype=g.dtype) ** (1.0 / 3.0)
            u = dirs / dirs.norm(dim=-1, keepdim=True) * radii * ctx.step_bound_m
            if first:
                u[0], first = 0.0, False
            with torch.no_grad():
                c = ctx.cost(ctx.rollout(u))[:, -1]
            i = int(c.argmin())
            if float(c[i]) < best_c:
                best_u, best_c = u[i].clone(), float(c[i])
            dt = time.perf_counter() - ts
        return best_u


def make_ctx(world: ToyWorld, horizon: int = 1, budget_s: float = BUDGET_S, feedback=None, tick: int = 0):
    x = world.start_pos
    return PlanContext(rollout=lambda u: world.rollout_model(x, u), cost=world.cost, goal=world.goal,
                       horizon_max=horizon, step_bound_m=world.cfg.step_bound_m, budget_s=budget_s,
                       tick=tick, feedback=feedback, seed=0)


def fake_feedback(norm: float, rho: float | None) -> Feedback:
    """Feedback whose `rho` property equals the requested value (None -> zero predicted reduction)."""
    cmd = torch.tensor([norm, 0.0, 0.0])
    if rho is None:
        return Feedback(cmd, cmd, 1.0, 0.9, 1.0, 1.0)
    return Feedback(cmd, cmd, cost_before=1.0, cost_after=1.0 - 0.1 * rho, cost_predicted=0.9,
                    cost_model_before=1.0)


def fmt(xs, spec="{:.3f}"):
    return "[" + ", ".join("None" if x is None else spec.format(x) for x in xs) + "]"


class ContractTests(unittest.TestCase):
    """(i) bounded, finite, in-budget plans for every planner / horizon / model variant."""

    def check(self, planner, world, horizon):
        res = simulate(planner, world, ticks=3, horizon=horizon, budget_s=BUDGET_S, seed=1)
        for row in res["ticks"]:
            plan = row["plan"]
            self.assertEqual(tuple(plan.shape), (horizon, 3))
            self.assertFalse(plan.requires_grad)
            self.assertEqual(plan.dtype, world.goal.dtype)
            self.assertTrue(bool(torch.isfinite(plan).all()))
            self.assertLessEqual(float(plan.norm(dim=-1).max()), world.cfg.step_bound_m * (1 + 1e-5))
            self.assertLess(row["plan_s"], BUDGET_S)
            self.assertIn("predictor_calls", row["log"])
            self.assertGreaterEqual(row["rollouts"], 1)

    def test_all_variants(self):
        planners = {"adam-autograd": lambda: AdamPlanner(), "adam-fd": lambda: AdamPlanner(grad_mode="fd"),
                    "trgn-fd-central": lambda: TRGNPlanner(), "trgn-jvp": lambda: TRGNPlanner(jac_mode="jvp"),
                    "trgn-fd-forward": lambda: TRGNPlanner(config=TRGNConfig(fd_scheme="forward"))}
        for defect in (False, True):
            cfg = ToyConfig().with_defect() if defect else ToyConfig()
            for horizon in (1, 3):
                for label, make in planners.items():
                    with self.subTest(defect=defect, horizon=horizon, planner=label):
                        planner = make()
                        if label == "trgn-jvp":
                            # forward-mode AD has a one-off first-use cost per process (0.8-1.0 s here, far
                            # above this budget and not controllable by the planner): pay it untimed.
                            t0 = time.perf_counter()
                            planner.plan(make_ctx(ToyWorld(cfg), horizon=horizon))
                            JVP_FIRST_CALL_S.append(time.perf_counter() - t0)
                        self.check(planner, ToyWorld(cfg), horizon)
                        if label == "trgn-jvp":
                            self.assertEqual(planner.jac_mode_used, "jvp")
        print(f"\njvp warm-up plan() seconds, in call order: {fmt(JVP_FIRST_CALL_S)}")

    def test_float64_and_non_differentiable_rollout(self):
        world = ToyWorld(ToyConfig(), dtype=torch.float64)
        self.check(TRGNPlanner(), world, 1)
        ctx = make_ctx(world)
        ctx.rollout = lambda u: world.rollout_model(world.start_pos, u.detach())  # breaks autograd / jvp
        adam = AdamPlanner()
        plan = adam.plan(ctx)
        self.assertEqual(adam.grad_mode_used, "fd")
        self.assertTrue(bool(torch.isfinite(plan).all()))

    def test_deterministic_given_seed(self):
        world = ToyWorld(ToyConfig().with_defect())
        for make in (lambda: AdamPlanner(config=AdamConfig(max_iters=20)), lambda: TRGNPlanner()):
            plans = []
            for _ in range(2):
                p = make()
                p.reset(3)
                plans.append(p.plan(make_ctx(world, budget_s=5.0)))
            self.assertTrue(torch.equal(plans[0], plans[1]))

    def test_trgn_first_action_respects_radius(self):
        p = TRGNPlanner(radius_init_m=0.02)
        plan = p.plan(make_ctx(ToyWorld(ToyConfig())))
        self.assertLessEqual(float(plan[0].norm()), 0.02 * (1 + 1e-5))
        self.assertGreater(float(plan[0].norm()), 0.9 * 0.02)   # goal is far: the step sits on the radius


class MathsTests(unittest.TestCase):
    def test_trust_region_solver(self):
        gen = torch.Generator().manual_seed(0)
        M = torch.randn(40, 6, generator=gen, dtype=torch.float64)
        A, b = M.T @ M, torch.randn(6, generator=gen, dtype=torch.float64)
        free = -torch.linalg.solve(A, b)
        d, lam = solve_trust_region(A, b, radius=10.0 * float(free.norm()))
        self.assertTrue(torch.allclose(d, free, rtol=1e-5, atol=1e-9))
        radius = 0.3 * float(free.norm())
        d, lam = solve_trust_region(A, b, radius=radius)
        self.assertAlmostEqual(float(d.norm()), radius, delta=1e-6 * radius)
        self.assertGreater(lam, 0.0)
        kkt = (A + lam * torch.eye(6, dtype=torch.float64)) @ d + b
        self.assertLess(float(kkt.norm()), 1e-8 * float(b.norm()))
        d, _ = solve_trust_region(torch.zeros(3, 3, dtype=torch.float64), torch.zeros(3, dtype=torch.float64), 0.1)
        self.assertEqual(float(d.norm()), 0.0)

    def test_jacobians_agree(self):
        world = ToyWorld(ToyConfig(), dtype=torch.float64)
        ctx = make_ctx(world, horizon=2)
        u = torch.tensor([[0.02, -0.01, 0.03], [0.0, 0.04, -0.02]], dtype=torch.float64)
        z_fd, J_fd = fd_linearise(ctx, u, None, 1e-4, "central")
        z_jvp, J_jvp = jvp_linearise(ctx, u)
        self.assertEqual(tuple(J_fd.shape), (2, world.cfg.P, 6))
        self.assertTrue(torch.allclose(z_fd, z_jvp))
        self.assertLess(float((J_fd - J_jvp).abs().max()), 1e-6)
        self.assertEqual(float(J_fd[0, :, 3:].abs().max()), 0.0)     # causality: tick 0 ignores u_1

    def test_central_fd_at_zero_cancels_even_term(self):
        """At u = 0 the central-difference Jacobian of the defective model is G times the true one."""
        clean = ToyWorld(ToyConfig(), dtype=torch.float64)
        bad = ToyWorld(ToyConfig().with_defect(), dtype=torch.float64)
        u = torch.zeros(1, 3, dtype=torch.float64)
        _, J_true = fd_linearise(make_ctx(clean), u, None, 1e-3, "central")
        _, J_bad = fd_linearise(make_ctx(bad), u, None, 1e-3, "central")
        self.assertLess(float((J_bad - bad.cfg.gain * J_true).abs().max()), 1e-6)
        _, J_fwd = fd_linearise(make_ctx(bad), u, None, 1e-3, "forward")
        self.assertGreater(float((J_fwd - bad.cfg.gain * J_true).abs().max()), 1e-2)


class RadiusTests(unittest.TestCase):
    """(iii) Delta shrinks on poor rho, grows on good rho at the radius, and resets."""

    def test_rule(self):
        cfg = TRGNConfig()
        self.assertAlmostEqual(next_radius(0.05, 0.1, True, cfg, 0.3), 0.025)
        self.assertAlmostEqual(next_radius(0.05, None, True, cfg, 0.3), 0.025)
        self.assertAlmostEqual(next_radius(0.05, -2.0, True, cfg, 0.3), 0.025)
        self.assertAlmostEqual(next_radius(0.05, 1.5, True, cfg, 0.3), 0.10)
        self.assertAlmostEqual(next_radius(0.05, 1.5, False, cfg, 0.3), 0.05)
        self.assertAlmostEqual(next_radius(0.05, 0.5, True, cfg, 0.3), 0.05)
        self.assertAlmostEqual(next_radius(0.25, 1.5, True, cfg, 0.3), 0.3)       # capped at the bound
        self.assertAlmostEqual(next_radius(0.006, 0.0, True, cfg, 0.3), 0.005)    # floored

    def test_planner_state_follows_feedback(self):
        world = ToyWorld(ToyConfig())
        p = TRGNPlanner()
        p.reset(0)
        p.plan(make_ctx(world))                                   # tick 0: no feedback, radius kept
        self.assertAlmostEqual(p.radius, 0.05)
        self.assertIsNone(p.rho_used)
        p.plan(make_ctx(world, feedback=fake_feedback(0.05, 1.2), tick=1))
        self.assertAlmostEqual(p.radius, 0.10)                    # good rho, step was at the radius
        self.assertAlmostEqual(float(p.rho_used), 1.2)
        ctx = make_ctx(world, feedback=fake_feedback(0.10, 0.1), tick=2)
        p.plan(ctx)
        self.assertAlmostEqual(p.radius, 0.05)                    # poor rho
        p.plan(make_ctx(world, feedback=fake_feedback(0.05, None), tick=3))
        self.assertAlmostEqual(p.radius, 0.025)                   # undefined rho counts as poor
        log = ctx.extras["planner_log"]
        for key in ("radius", "rho_used", "inner_iters", "predictor_calls"):
            self.assertIn(key, log)
        self.assertAlmostEqual(log["radius"], 0.05)
        self.assertGreaterEqual(p.inner_iters, 1)
        self.assertGreaterEqual(p.predictor_calls, 1)
        p.reset(1)
        self.assertAlmostEqual(p.radius, 0.05)

    def test_closed_loop_traces(self):
        clean = simulate(TRGNPlanner(), ToyWorld(ToyConfig()), ticks=TICKS, budget_s=BUDGET_S)
        radii = [row["log"]["radius"] for row in clean["ticks"]]
        self.assertEqual(radii[:3], [0.05, 0.10, 0.20])           # model == world: rho ~ 1, radius doubles
        # a model that claims a reduction the world does not deliver: the radius must fall
        liar = ToyWorld(ToyConfig(exec_gain=0.0))
        res = simulate(TRGNPlanner(), liar, ticks=TICKS, budget_s=BUDGET_S)
        radii = [row["log"]["radius"] for row in res["ticks"]]
        self.assertEqual(radii, [0.05, 0.025, 0.0125, 0.00625, 0.005])


class ClosedLoopTests(unittest.TestCase):
    def test_defect_free_progress(self):
        """(ii) both planners exceed 90% progress in 5 ticks when the model equals the world."""
        for horizon in (1, 3):
            for make in (AdamPlanner, TRGNPlanner):
                res = simulate(make(), ToyWorld(ToyConfig()), ticks=TICKS, horizon=horizon, budget_s=BUDGET_S)
                with self.subTest(planner=make.name, horizon=horizon):
                    self.assertGreater(res["progress"], 0.90)

    def test_report_progress_table(self):
        """(iv) printed comparison, no winner asserted."""
        makers = [("adam", AdamPlanner), ("trgn", TRGNPlanner),
                  ("trgn-1step", lambda: TRGNPlanner(config=TRGNConfig(max_inner_iters=1))),
                  ("random_shooting", RandomShootingPlanner)]
        print(f"\n=== toy progress, {TICKS} ticks, H=1, budget {BUDGET_S}s, 3 toy seeds (mean [per seed]) ===")
        for label, cfg_fn in (("defect off", lambda s: ToyConfig(seed=s)),
                              ("defect on (G=0.25, beta=1.25/m)", lambda s: ToyConfig(seed=s).with_defect())):
            for name, make in makers:
                runs = [simulate(make(), ToyWorld(cfg_fn(s)), ticks=TICKS, budget_s=BUDGET_S, seed=s)
                        for s in range(3)]
                prog = [r["progress"] for r in runs]
                self.assertTrue(all(math.isfinite(p) for p in prog))
                print(f"{label:32s} {name:16s} progress {sum(prog) / len(prog):+.3f} {fmt(prog)}")
                first = runs[0]["ticks"]
                print(f"    seed 0: step_norm {fmt([t['step_norm'] for t in first])} "
                      f"dist {fmt([t['distance'] for t in first])} rho {fmt([t['rho'] for t in first], '{:.2f}')}")
                if name.startswith("trgn"):
                    print(f"    seed 0: radius {fmt([t['log']['radius'] for t in first], '{:.4f}')} "
                          f"inner {[t['log']['inner_iters'] for t in first]} "
                          f"calls {[t['log']['predictor_calls'] for t in first]}")
                if name == "adam":
                    print(f"    seed 0: iterations {[t['log']['iterations'] for t in first]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
