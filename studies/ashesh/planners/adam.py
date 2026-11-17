"""`adam` — projected Adam on the native goal cost. No feedback.

Objective over a plan u in R^{H x 3} (metres, one row per model tick):

    L(u) = sum_h  gamma^(H-1-h) * C(z_h(u)),      C(z) = mean |z - g|,   z = ctx.rollout(u)

so the final predicted step has weight 1 and earlier steps are discounted backwards (H = 1: L is the
native cost of the single predicted step). Update, for k = 1, 2, ...:

    g_k = dL/du (autograd, or batched central differences)
    m_k = b1 m_{k-1} + (1 - b1) g_k,   v_k = b2 v_{k-1} + (1 - b2) g_k^2
    u_{k+1} = Proj( u_k - lr * m_hat_k / (sqrt(v_hat_k) + eps) )

Proj clips every row to the L2 ball of radius `ctx.step_bound_m`. The loop runs until the wall-clock
budget is nearly used and returns the best-cost iterate that was actually evaluated.
"""
from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from bench.interface import PlanContext


@dataclass(frozen=True)
class AdamConfig:
    lr_m: float = 0.01               # Adam step size in metres (each coordinate moves ~lr per iteration)
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    lr_decay: float = 1.0            # multiplicative decay per iteration (1.0 = constant)
    discount: float = 0.99           # gamma in L(u)
    grad_mode: str = "autograd"      # "autograd" | "fd" (autograd falls back to fd if it fails)
    fd_step_m: float = 1e-3          # central-difference step for grad_mode == "fd"
    init_noise_m: float = 1e-3       # std of the seeded perturbation when there is no warm start
    horizon: Optional[int] = None    # None -> ctx.horizon_max
    max_iters: Optional[int] = None  # None -> budget-limited only (set it for exact reproducibility)
    budget_safety: float = 0.10      # fraction of budget_s kept in reserve
    time_factor: float = 1.5         # next iteration must fit time_factor * (recent iteration time)


def horizon_weights(H: int, discount: float, ref: torch.Tensor) -> torch.Tensor:
    """omega_h = discount^(H-1-h), shape (H,)."""
    exps = torch.arange(H - 1, -1, -1, device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(discount, device=ref.device, dtype=ref.dtype) ** exps


def project_rows(u: torch.Tensor, radius: float) -> torch.Tensor:
    """Row-wise projection onto {||u_h||_2 <= radius}."""
    norms = u.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return u * (radius / norms).clamp(max=1.0)


def objective(ctx: PlanContext, u: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """(B, H, 3) -> (B,):  L(u) = sum_h omega_h C(z_h(u))."""
    return (ctx.cost(ctx.rollout(u)) * omega).sum(-1)


def initial_plan(ctx: PlanContext, H: int, noise_m: float, seed: int) -> torch.Tensor:
    """Warm start (cut or zero-padded to H rows) if given, else zeros + seeded Gaussian noise."""
    g = ctx.goal
    u = torch.zeros(H, 3, device=g.device, dtype=g.dtype)
    if ctx.warm_start is not None:
        w = ctx.warm_start.detach().to(device=g.device, dtype=g.dtype)[:H]
        u[: w.shape[0]] = w
        return torch.nan_to_num(u)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(H, 3, generator=gen, dtype=torch.float64) * noise_m
    return u + noise.to(device=g.device, dtype=g.dtype)


def value_and_grad_autograd(ctx, u, omega) -> Tuple[torch.Tensor, torch.Tensor]:
    """L(u) and dL/du by reverse mode: one rollout, one backward."""
    with torch.enable_grad():
        x = u.detach().clone().requires_grad_(True)
        loss = objective(ctx, x[None], omega)[0]
        (grad,) = torch.autograd.grad(loss, x)
    return loss.detach(), grad.detach()


def value_and_grad_fd(ctx, u, omega, step: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """L(u) and central differences dL/du_i ~ [L(u + s e_i) - L(u - s e_i)] / 2s: one rollout of 6H+1 plans."""
    n = u.numel()
    eye = torch.eye(n, device=u.device, dtype=u.dtype).reshape(n, *u.shape) * step
    batch = torch.cat([u[None], u[None] + eye, u[None] - eye], dim=0)
    with torch.no_grad():
        vals = objective(ctx, batch, omega)
    grad = ((vals[1 : n + 1] - vals[n + 1 :]) / (2.0 * step)).reshape(u.shape)
    return vals[0], grad


class AdamPlanner:
    name = "adam"

    def __init__(self, grad_mode: Optional[str] = None, config: AdamConfig = AdamConfig()):
        self.cfg = config if grad_mode is None else dataclasses.replace(config, grad_mode=grad_mode)
        if self.cfg.grad_mode not in ("autograd", "fd"):
            raise ValueError(f"grad_mode must be 'autograd' or 'fd', got {self.cfg.grad_mode!r}")
        self.reset(0)

    def reset(self, seed: int) -> None:
        self._seed = int(seed)
        self.grad_mode_used = self.cfg.grad_mode
        self.iterations = 0
        self.predictor_calls = 0
        self.log: dict = {}

    # -- one gradient evaluation, with the fallback ------------------------------------------------
    def _value_and_grad(self, ctx, u, omega):
        if self.grad_mode_used == "autograd":
            try:
                return value_and_grad_autograd(ctx, u, omega)
            except RuntimeError as err:  # rollout not differentiable w.r.t. actions
                self.grad_mode_used = "fd"
                self.log["autograd_error"] = str(err)[:200]
        return value_and_grad_fd(ctx, u, omega, self.cfg.fd_step_m)

    def plan(self, ctx: PlanContext) -> torch.Tensor:
        t0 = time.perf_counter()
        cfg = self.cfg
        deadline = t0 + ctx.budget_s * (1.0 - cfg.budget_safety)
        H = max(1, min(cfg.horizon or ctx.horizon_max, ctx.horizon_max))
        omega = horizon_weights(H, cfg.discount, ctx.goal)
        seed = (self._seed * 1_000_003 + int(ctx.seed) * 9_176 + int(ctx.tick)) % (2**62)

        u = project_rows(initial_plan(ctx, H, cfg.init_noise_m, seed), ctx.step_bound_m)
        m, v = torch.zeros_like(u), torch.zeros_like(u)
        best_u, best_c, first_c = torch.zeros_like(u), math.inf, math.nan
        recent = [0.0, 0.0]
        k, lr = 0, cfg.lr_m
        self.log, self.predictor_calls = {}, 0

        while cfg.max_iters is None or k < cfg.max_iters:
            now = time.perf_counter()
            if k > 0 and now + cfg.time_factor * max(recent) > deadline:
                break
            loss, grad = self._value_and_grad(ctx, u, omega)
            self.predictor_calls += 1
            recent = [recent[1], time.perf_counter() - now]
            c = float(loss)
            if k == 0:
                first_c = c
            if math.isfinite(c) and c < best_c:
                best_u, best_c = u.clone(), c
            if not bool(torch.isfinite(grad).all()):
                break
            k += 1
            m = cfg.beta1 * m + (1.0 - cfg.beta1) * grad
            v = cfg.beta2 * v + (1.0 - cfg.beta2) * grad * grad
            m_hat, v_hat = m / (1.0 - cfg.beta1**k), v / (1.0 - cfg.beta2**k)
            u = project_rows(u - lr * m_hat / (v_hat.sqrt() + cfg.eps), ctx.step_bound_m)
            lr *= cfg.lr_decay

        self.iterations = k
        plan = project_rows(torch.nan_to_num(best_u), ctx.step_bound_m).detach()
        self.log.update(
            planner=self.name, iterations=k, predictor_calls=self.predictor_calls,
            grad_mode_used=self.grad_mode_used, cost_model_start=first_c, cost_model_end=best_c,
            plan_norm=float(plan[0].norm()), elapsed_s=time.perf_counter() - t0,
        )
        ctx.extras["planner_log"] = dict(self.log)
        return plan
