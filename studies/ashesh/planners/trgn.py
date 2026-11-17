"""`trgn` — trust-region Gauss-Newton (IRLS) on a pseudo-Huber-smoothed L1, radius set by realized feedback.

Plan u in R^{H x 3}, n = 3H unknowns. Residual r(u) = z(u) - g in R^{H x P}, Jacobian J = dz/du in
R^{(H P) x n} (only n columns, so it is obtained with n forward tangents or one batched finite-difference
rollout). Smoothed objective with horizon weights omega_h = gamma^(H-1-h):

    phi(u) = sum_{h,k} omega_h * ( sqrt(r_hk^2 + delta^2) - delta ),     delta = delta_frac * median |r|

IRLS weights w_hk = omega_h / sqrt(r_hk^2 + delta^2) make  sum w (r + J d)^2 / 2  a quadratic majoriser of
phi along the linearisation. Each inner iteration solves the n x n trust-region subproblem

    min_d  sum_k w_k (r_k + J_k d)^2    s.t. ||d|| <= Delta_in
    <=>   (J^T W J + lambda I) d = -J^T W r,   lambda >= 0 the smallest value with ||d(lambda)|| <= Delta_in

exactly, by eigen-decomposition of J^T W J and bisection on the scalar secular equation. The candidate
u + d is projected onto the feasible set {||u_0|| <= Delta, ||u_h|| <= step_bound_m} and accepted only if
the native model cost L(u) = sum_h omega_h mean|z_h - g| decreases; Delta_in follows the model-side ratio
(actual model decrease / decrease of the linearised L1).

Two radii, two sources of truth:
  * Delta_in (inner, per plan() call) guards the *linearisation* against the model itself.
  * Delta (outer, persists across ticks) caps the norm of the executed first action and guards the *model*
    against the world: after each executed tick, with rho = realized / predicted cost reduction,
        rho is None or rho < rho_low            -> Delta <- shrink * Delta
        rho > rho_high and last step at radius  -> Delta <- grow * Delta
        otherwise                               -> keep,
    clipped to [radius_min_m, ctx.step_bound_m].

With central differences taken at u = 0 the Jacobian keeps only the odd (direction-dependent) part of the
model's response; a direction-independent term that depends on ||u|| alone cancels in [z(+s e) - z(-s e)]/2s.
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
class TRGNConfig:
    radius_init_m: float = 0.05      # initial outer radius Delta (cap on ||first action||)
    radius_min_m: float = 0.005
    shrink: float = 0.5
    grow: float = 2.0
    rho_low: float = 0.25
    rho_high: float = 0.75
    at_radius_frac: float = 0.9      # "last step was at the radius" <=> ||u_0|| >= frac * Delta
    huber_delta_frac: float = 0.1    # delta = frac * median |r|
    jac_mode: str = "fd"             # "fd" | "jvp" (jvp falls back to fd if forward mode fails)
    fd_scheme: str = "central"       # "central" (6H rows) | "forward" (3H rows)
    fd_step_m: float = 1e-3
    discount: float = 0.99
    lambda_floor_rel: float = 1e-8   # lambda >= floor * largest eigenvalue of J^T W J
    inner_accept: float = 1e-4       # accept iff model decrease > inner_accept * linearised decrease
    max_inner_iters: int = 50        # linearisations per plan(); 1 = a single GN step from the start point
    step_tol_m: float = 1e-5
    use_warm_start: bool = True
    horizon: Optional[int] = None    # None -> ctx.horizon_max
    budget_safety: float = 0.10
    time_factor: float = 1.5


# ---------------------------------------------------------------- small pure functions

def horizon_weights(H: int, discount: float, ref: torch.Tensor) -> torch.Tensor:
    """omega_h = discount^(H-1-h), shape (H,)."""
    exps = torch.arange(H - 1, -1, -1, device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(discount, device=ref.device, dtype=ref.dtype) ** exps


def project_plan(u: torch.Tensor, first_radius: float, step_bound: float) -> torch.Tensor:
    """Row 0 onto the ball of radius min(first_radius, step_bound); other rows onto radius step_bound."""
    radii = torch.full((u.shape[0], 1), step_bound, device=u.device, dtype=u.dtype)
    radii[0, 0] = min(first_radius, step_bound)
    norms = u.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return u * (radii / norms).clamp(max=1.0)


def next_radius(radius: float, rho: Optional[float], at_radius: bool, cfg: TRGNConfig,
                step_bound: float) -> float:
    """Outer trust-radius rule from the realized ratio rho (see module docstring)."""
    if rho is None or not math.isfinite(rho) or rho < cfg.rho_low:
        radius *= cfg.shrink
    elif rho > cfg.rho_high and at_radius:
        radius *= cfg.grow
    return min(max(radius, cfg.radius_min_m), max(step_bound, cfg.radius_min_m))


def irls_normal_equations(r: torch.Tensor, J: torch.Tensor, omega: torch.Tensor,
                          delta_frac: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """A = J^T W J (n, n) and b = J^T W r (n,), float64 on CPU, both divided by sum(w).

    r: (H, P), J: (H, P, n), w_hk = omega_h / sqrt(r_hk^2 + delta^2).
    """
    n = J.shape[-1]
    delta = (delta_frac * r.abs().median()).clamp_min(torch.finfo(r.dtype).tiny ** 0.5)
    w = (omega[:, None] / torch.sqrt(r * r + delta * delta)).reshape(-1)
    Jf, rf = J.reshape(-1, n), r.reshape(-1)
    WJ = Jf * w[:, None]
    scale = w.sum().double().cpu().clamp_min(1e-300)
    return (Jf.T @ WJ).double().cpu() / scale, (WJ.T @ rf).double().cpu() / scale


def solve_trust_region(A: torch.Tensor, b: torch.Tensor, radius: float,
                       floor_rel: float = 1e-8) -> Tuple[torch.Tensor, float]:
    """argmin_d d^T A d / 2 + b^T d  s.t. ||d|| <= radius, for symmetric PSD A (n x n, tiny).

    With A = Q diag(e) Q^T:  d(lambda) = -Q [ (Q^T b) / (e + lambda) ],  ||d(lambda)|| decreasing in lambda.
    Returns (d, lambda) with lambda the smallest value >= floor for which ||d|| <= radius.
    """
    A = 0.5 * (A + A.T)
    if not (bool(torch.isfinite(A).all()) and bool(torch.isfinite(b).all())) or float(b.norm()) == 0.0:
        return torch.zeros_like(b), 0.0
    e, Q = torch.linalg.eigh(A)
    e = e.clamp_min(0.0)
    bq = Q.T @ b
    lam_lo = max(floor_rel * float(e.max()), 1e-300)

    def step_norm(lam: float) -> float:
        return float((bq / (e + lam)).norm())

    lam = lam_lo
    if step_norm(lam_lo) > radius:
        lam_hi = max(float(b.norm()) / radius, lam_lo)      # ||d|| <= ||b|| / lambda
        lo, hi = math.log(lam_lo), math.log(lam_hi)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if step_norm(math.exp(mid)) > radius else (lo, mid)
        lam = math.exp(hi)
    return -(Q @ (bq / (e + lam))), lam


def fd_linearise(ctx: PlanContext, u: torch.Tensor, z_u: Optional[torch.Tensor], step: float,
                 scheme: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """z(u) (H, P) and J (H, P, n) from ONE batched rollout of perturbed plans u +- step * e_i."""
    n = u.numel()
    eye = torch.eye(n, device=u.device, dtype=u.dtype).reshape(n, *u.shape) * step
    parts = [u[None] + eye] + ([u[None] - eye] if scheme == "central" else [])
    if z_u is None:                              # base point rides along in the same batch
        parts.append(u[None])
    with torch.no_grad():
        Z = ctx.rollout(torch.cat(parts, dim=0))
    if z_u is None:
        z_u = Z[-1]
    if scheme == "central":
        J = (Z[:n] - Z[n : 2 * n]) / (2.0 * step)
    else:
        J = (Z[:n] - z_u[None]) / step
    return z_u, J.permute(1, 2, 0)


def jvp_linearise(ctx: PlanContext, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """z(u) and J by forward mode: n = 3H tangent passes, column i = J e_i."""
    def f(v):
        return ctx.rollout(v[None])[0]

    cols, z = [], None
    for e in torch.eye(u.numel(), device=u.device, dtype=u.dtype).reshape(-1, *u.shape):
        with torch.enable_grad():
            z, jz = torch.func.jvp(f, (u,), (e,))
        cols.append(jz.detach())
    return z.detach(), torch.stack(cols, dim=-1)


# ---------------------------------------------------------------- planner

class TRGNPlanner:
    name = "trgn"

    def __init__(self, radius_init_m: Optional[float] = None, jac_mode: Optional[str] = None,
                 config: TRGNConfig = TRGNConfig()):
        over = {k: v for k, v in dict(radius_init_m=radius_init_m, jac_mode=jac_mode).items() if v is not None}
        self.cfg = dataclasses.replace(config, **over)
        if self.cfg.jac_mode not in ("fd", "jvp") or self.cfg.fd_scheme not in ("central", "forward"):
            raise ValueError("jac_mode in {'fd','jvp'}, fd_scheme in {'central','forward'}")
        self.reset(0)

    def reset(self, seed: int) -> None:
        self._seed = int(seed)                   # the planner is deterministic; kept for the protocol
        self.radius = float(self.cfg.radius_init_m)
        self.rho_used: Optional[float] = None
        self.inner_iters = 0
        self.predictor_calls = 0                 # ctx.rollout invocations in the last plan()
        self.rollout_rows = 0                    # action sequences evaluated in the last plan()
        self.jac_mode_used = self.cfg.jac_mode
        self._last_first_norm: Optional[float] = None
        self.log: dict = {}

    # -- model access (the only two places that touch ctx.rollout) ---------------------------------
    def _linearise(self, ctx, u, z_u):
        cfg = self.cfg
        if self.jac_mode_used == "jvp":
            try:
                out = jvp_linearise(ctx, u)
                self.predictor_calls += u.numel()
                self.rollout_rows += u.numel()
                return out
            except Exception as err:  # forward mode unsupported somewhere in the model
                self.jac_mode_used = "fd"
                self.log["jvp_error"] = f"{type(err).__name__}: {str(err)[:200]}"
        n = u.numel()
        self.predictor_calls += 1
        self.rollout_rows += n * (2 if cfg.fd_scheme == "central" else 1) + (z_u is None)
        return fd_linearise(ctx, u, z_u, cfg.fd_step_m, cfg.fd_scheme)

    def _evaluate(self, ctx, u):
        self.predictor_calls += 1
        self.rollout_rows += 1
        with torch.no_grad():
            return ctx.rollout(u[None])[0]

    def _update_radius(self, ctx) -> bool:
        """Apply the realized-feedback rule; returns whether the last step counted as 'at the radius'."""
        bound = float(ctx.step_bound_m)
        fb = ctx.feedback
        self.rho_used, at_radius = None, False
        if fb is not None:
            self.rho_used = fb.rho
            last = float(fb.commanded.norm()) if self._last_first_norm is None else self._last_first_norm
            at_radius = last >= self.cfg.at_radius_frac * min(self.radius, bound)
            self.radius = next_radius(self.radius, self.rho_used, at_radius, self.cfg, bound)
        self.radius = min(max(self.radius, self.cfg.radius_min_m), max(bound, self.cfg.radius_min_m))
        return at_radius

    def plan(self, ctx: PlanContext) -> torch.Tensor:
        t0 = time.perf_counter()
        cfg, g = self.cfg, ctx.goal
        deadline = t0 + ctx.budget_s * (1.0 - cfg.budget_safety)
        bound = float(ctx.step_bound_m)
        H = max(1, min(cfg.horizon or ctx.horizon_max, ctx.horizon_max))
        omega = horizon_weights(H, cfg.discount, g)
        self.log, self.inner_iters, self.predictor_calls, self.rollout_rows = {}, 0, 0, 0
        radius_prev = self.radius
        at_radius_last = self._update_radius(ctx)
        Delta = self.radius

        def model_cost(z):                       # native L(u) from features (H, P)
            return float((ctx.cost(z) * omega).sum())

        u = torch.zeros(H, 3, device=g.device, dtype=g.dtype)
        if cfg.use_warm_start and ctx.warm_start is not None:
            w = torch.nan_to_num(ctx.warm_start.detach().to(device=g.device, dtype=g.dtype))[:H]
            u[: w.shape[0]] = w
        u = project_plan(u, Delta, bound)

        z, c = None, math.nan
        cost_start, lam, accepted = math.nan, 0.0, 0
        delta_in = Delta * math.sqrt(H)
        delta_in_max = bound * math.sqrt(H)
        dt_lin, dt_eval = 0.0, 0.0
        A = b = J = r = None

        while self.inner_iters < cfg.max_inner_iters or A is not None:
            now = time.perf_counter()
            if A is None:                        # (re-)linearise at the current iterate
                if self.inner_iters >= cfg.max_inner_iters:
                    break
                if self.inner_iters > 0 and now + cfg.time_factor * (dt_lin + dt_eval) > deadline:
                    break
                z, J = self._linearise(ctx, u, z)
                self.inner_iters += 1
                dt_lin = time.perf_counter() - now
                if not (bool(torch.isfinite(z).all()) and bool(torch.isfinite(J).all())):
                    break
                r = z - g
                c = model_cost(z)
                if self.inner_iters == 1:
                    cost_start = c
                A, b = irls_normal_equations(r, J, omega, cfg.huber_delta_frac)
                now = time.perf_counter()

            d, lam = solve_trust_region(A, b, delta_in, cfg.lambda_floor_rel)
            u_new = project_plan(u + d.to(device=g.device, dtype=g.dtype).reshape(H, 3), Delta, bound)
            step = u_new - u
            if float(step.norm()) < cfg.step_tol_m:
                break
            if now + cfg.time_factor * (dt_eval if dt_eval > 0 else dt_lin) > deadline:
                break
            z_new = self._evaluate(ctx, u_new)
            dt_eval = time.perf_counter() - now
            c_new = model_cost(z_new)
            lin = r + (J @ step.reshape(-1))                 # linearised residual (H, P)
            pred = c - model_cost(lin + g)                   # decrease of the linearised native cost
            decrease = c - c_new
            ratio = decrease / pred if pred > 0 else (1.0 if decrease > 0 else -1.0)
            at_inner = float(d.norm()) >= 0.9 * delta_in
            if math.isfinite(c_new) and decrease > cfg.inner_accept * max(pred, 0.0) and decrease > 0:
                u, z, c, A = u_new, z_new, c_new, None       # accept, re-linearise next
                accepted += 1
                if ratio > 0.75 and at_inner:
                    delta_in = min(2.0 * delta_in, delta_in_max)
                elif ratio < 0.25:
                    delta_in *= 0.5
            else:                                            # reject: same linearisation, smaller radius
                delta_in = 0.5 * min(delta_in, float(d.norm()))
                if delta_in < cfg.step_tol_m:
                    break

        plan = project_plan(torch.nan_to_num(u), Delta, bound).detach()
        self._last_first_norm = float(plan[0].norm())
        self.log.update(
            planner=self.name, radius=Delta, radius_prev=radius_prev, rho_used=self.rho_used,
            at_radius_last=at_radius_last, inner_iters=self.inner_iters, accepted_steps=accepted,
            predictor_calls=self.predictor_calls, rollout_rows=self.rollout_rows,
            jac_mode_used=self.jac_mode_used, cost_model_start=cost_start, cost_model_end=c,
            lambda_last=lam, plan_norm=self._last_first_norm, elapsed_s=time.perf_counter() - t0,
        )
        ctx.extras["planner_log"] = dict(self.log)
        return plan
