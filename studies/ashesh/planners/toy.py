"""Toy latent world with the planner contract's `rollout` / `cost` signature, plus a 5-tick closed loop.

World (ground truth). End-effector position x in R^3, features

    f(x) = z0 + A @ phi(x),   phi(x) = [tanh(x / s), tanh(x / s)^2 / 2] in R^6,   A in R^{P x 6}

Goal features g = f(x_goal), cost C(z) = mean |z - g|. A plan u in R^{H x 3} moves x by cumsum(u).

Model handed to planners (rolled out from the current position x):

    z_model_h(u) = f(x) + G * [ f(x + cumsum(u)_h) - f(x) ] + min(1, beta * m_h(u)) * [ g - f(x) ]
    m_h(u) = sum_{j <= h} ||u_j||                                  (smoothed at 0)

G = 1, beta = 0 is the defect-free model (model == world). The defect mimics E001: the directional
response is attenuated by G, and a direction-independent term pulls features toward the goal in
proportion to the commanded amplitude, so the model predicts a cost decrease for almost any motion.
Feedback is always produced by the true world.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch

from bench.interface import Feedback, PlanContext


@dataclass(frozen=True)
class ToyConfig:
    P: int = 2000
    goal_dist_m: float = 0.28
    step_bound_m: float = 0.30
    sat_scale_m: float = 0.5      # s in phi: tanh saturation length
    gain: float = 1.0             # G, directional gain of the model
    even_beta: float = 0.0        # beta (1/m), direction-independent pull toward the goal
    exec_gain: float = 1.0        # executed / commanded motion in the world
    seed: int = 0

    def with_defect(self, gain: float = 0.25, even_beta: float = 1.25) -> "ToyConfig":
        """E001-like numbers: toward the goal at 0.28 m the model predicts (G + beta*0.28) = 0.6 of the
        realized cost change; away from it and sideways it still predicts a decrease."""
        return ToyConfig(**{**self.__dict__, "gain": gain, "even_beta": even_beta})


class ToyWorld:
    def __init__(self, cfg: ToyConfig = ToyConfig(), dtype=torch.float32, device="cpu"):
        self.cfg = cfg
        gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
        kw = dict(dtype=dtype, device=device)
        self.A = (torch.randn(cfg.P, 6, generator=gen, dtype=torch.float64) / 6**0.5).to(**kw)
        self.z0 = torch.randn(cfg.P, generator=gen, dtype=torch.float64).to(**kw)
        direction = torch.randn(3, generator=gen, dtype=torch.float64)
        self.goal_pos = (direction / direction.norm() * cfg.goal_dist_m).to(**kw)
        self.start_pos = torch.zeros(3, **kw)
        self.goal = self.features(self.goal_pos)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """(..., 3) -> (..., P)."""
        t = torch.tanh(x / self.cfg.sat_scale_m)
        return self.z0 + torch.cat([t, 0.5 * t * t], dim=-1) @ self.A.T

    def cost(self, z: torch.Tensor) -> torch.Tensor:
        """(..., P) -> (...): mean |z - g|."""
        return (z - self.goal).abs().mean(-1)

    def rollout_true(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """(B, H, 3) -> (B, H, P), the world's features after each tick."""
        return self.features(x + torch.cumsum(u, dim=1))

    def rollout_model(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """(B, H, 3) -> (B, H, P), the (possibly defective) model's prediction. Differentiable in u."""
        z_now = self.features(x)
        z = z_now + self.cfg.gain * (self.rollout_true(x, u) - z_now)
        if self.cfg.even_beta != 0.0:
            eps = 1e-6
            amp = torch.cumsum(torch.sqrt((u * u).sum(-1) + eps * eps) - eps, dim=1)
            pull = (self.cfg.even_beta * amp).clamp(max=1.0)[..., None]
            z = z + pull * (self.goal - z_now)
        return z

    def distance(self, x: torch.Tensor) -> float:
        return float((x - self.goal_pos).norm())


def clip_norm(u: torch.Tensor, bound: float) -> torch.Tensor:
    n = float(u.norm())
    return u if n <= bound else u * (bound / n)


def simulate(planner, world: ToyWorld, ticks: int = 5, horizon: int = 1, budget_s: float = 0.3,
             seed: int = 0) -> dict:
    """Closed loop exactly as the contract describes: plan, clip row 0, execute it in the true world,
    build Feedback from realized and model costs, shift the plan into the next warm start, replan."""
    cfg = world.cfg
    planner.reset(seed)
    x = world.start_pos.clone()
    d0 = world.distance(x)
    feedback: Optional[Feedback] = None
    warm: Optional[torch.Tensor] = None
    rows = []
    for tick in range(ticks):
        calls = {"rollouts": 0, "rows": 0}

        def rollout(u, x=x, calls=calls):
            calls["rollouts"] += 1
            calls["rows"] += u.shape[0]
            return world.rollout_model(x, u)

        ctx = PlanContext(rollout=rollout, cost=world.cost, goal=world.goal, horizon_max=horizon,
                          step_bound_m=cfg.step_bound_m, budget_s=budget_s, tick=tick,
                          feedback=feedback, warm_start=warm, seed=seed)
        t0 = time.perf_counter()
        plan = planner.plan(ctx)
        plan_s = time.perf_counter() - t0

        commanded = clip_norm(plan[0].detach(), cfg.step_bound_m)
        executed = cfg.exec_gain * commanded
        with torch.no_grad():
            cost_before = float(world.cost(world.features(x)))
            cost_null = float(world.cost(world.rollout_model(x, torch.zeros_like(commanded)[None, None])[0, 0]))
            cost_pred = float(world.cost(world.rollout_model(x, commanded[None, None])[0, 0]))
            x = x + executed
            cost_after = float(world.cost(world.features(x)))
        feedback = Feedback(commanded=commanded, executed=executed, cost_before=cost_before,
                            cost_after=cost_after, cost_predicted=cost_pred, cost_model_before=cost_null)
        warm = None
        if plan.shape[0] > 1:
            warm = torch.cat([plan[1:], torch.zeros_like(plan[:1])], dim=0).detach()
        rows.append(dict(tick=tick, plan=plan, plan_s=plan_s, step_norm=float(commanded.norm()),
                         rho=feedback.rho, distance=world.distance(x), rollouts=calls["rollouts"],
                         rollout_rows=calls["rows"], log=dict(ctx.extras.get("planner_log", {}))))
    return dict(progress=1.0 - world.distance(x) / d0, d0=d0, ticks=rows)
