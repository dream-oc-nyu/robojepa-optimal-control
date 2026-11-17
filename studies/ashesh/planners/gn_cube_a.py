"""Deterministic finite-difference IRLS/Gauss--Newton on Q = [-.05,.05]^3 m.

    r = z(a) - g,  delta = max(delta_frac median|r|, delta_floor)
    w_k = (r_k^2 + delta^2)^(-1/2),  A = J^T W J / P, b = J^T W r / P
    d_F(lambda) = -(A_FF + lambda I)^(-1) b_F, d_active = 0, ||d|| <= radius
    a_trial = projection_Q(a + d); accept iff native L(a_trial) < L(a)

F excludes face coordinates whose negative IRLS gradient points out of Q.
The smallest LM damping above a relative eigenvalue floor giving the trust
radius is found by bisection, in CPU float64. Rejection halves the radius
(and retries the SAME linearisation); acceptance grows it and re-linearises.

Finite differences keep their centimetre-scale width at faces: shift the
centre c_i = clip(a_i, -.05+s, .05-s), use endpoints c_i +/- s and divide
by their actual represented separation. Other coordinates remain at a.
Thus a face uses an inward one-sided secant of length 2s, never a clipped,
vanishing stencil. Six endpoints and the unknown base share one rollout.
The first central difference at zero cancels action-even feature responses;
native cost acceptance does not remove any goal-independent motion reward.

Every tick starts at zero, ignores feedback/warm starts, and counts every
rollout row. Only no-grad model access is used. The mandatory initial base
measurement establishes L(0), even for a time budget too small for search.
Time limits are checked between operations (an in-flight rollout cannot be
preempted). Any observed non-finite feature, cost or algebra aborts to zero;
an unavailable/non-finite logged cost is None, not a fabricated finite value.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch

from bench.interface import PlanContext


BOUND_M = 0.05


@dataclass(frozen=True)
class GNCubeConfig:
    fd_step_m: float = 0.02
    max_rows: int = 150
    delta_frac: float = 0.1
    delta_floor: float = 1e-8
    radius_init_m: float = 0.05
    radius_min_m: float = 0.0005
    radius_max_m: float = 0.10
    radius_shrink: float = 0.5
    radius_grow: float = 2.0
    step_tol_m: float = 0.0005
    max_linearisations: int = 20
    lambda_floor_rel: float = 1e-8
    damping_bisections: int = 60
    time_fraction: float = 0.5

    def __post_init__(self):
        numeric = (self.fd_step_m, self.delta_frac, self.delta_floor,
                   self.radius_init_m, self.radius_min_m, self.radius_max_m,
                   self.radius_shrink, self.radius_grow, self.step_tol_m,
                   self.lambda_floor_rel, self.time_fraction)
        if not all(math.isfinite(x) and x > 0 for x in numeric):
            raise ValueError('All continuous settings must be positive and finite')
        if not 0.01 <= self.fd_step_m <= BOUND_M:
            raise ValueError('fd_step_m must be centimetre-scale, in [0.01, 0.05]')
        if not self.radius_min_m <= self.radius_init_m <= self.radius_max_m:
            raise ValueError('Require radius_min <= radius_init <= radius_max')
        if not self.radius_shrink < 1 <= self.radius_grow:
            raise ValueError('Require shrink < 1 <= grow')
        if not self.time_fraction <= 0.5:
            raise ValueError('time_fraction must be <= 0.5')
        for value in (self.max_rows, self.max_linearisations, self.damping_bisections):
            if type(value) is not int or value < 1:
                raise ValueError('Row, linearisation and bisection limits must be positive integers')


def stencil(a: torch.Tensor, half_width: float):
    """Six in-box points, paired low/high per axis; retain full 2s width."""
    centre = a.clamp(-BOUND_M + half_width, BOUND_M - half_width)
    points = a.repeat(6, 1)
    for i in range(3):
        points[2 * i, i] = centre[i] - half_width
        points[2 * i + 1, i] = centre[i] + half_width
    points.clamp_(-BOUND_M, BOUND_M)
    widths = torch.stack([points[2 * i + 1, i] - points[2 * i, i] for i in range(3)])
    return points, widths


def normal_equations(residual: torch.Tensor, jacobian: torch.Tensor, cfg: GNCubeConfig):
    """Device-side reductions; one P-vector scratch, never a weighted P x 3 copy."""
    delta = (cfg.delta_frac * residual.abs().median()).clamp_min(cfg.delta_floor)
    weights = torch.rsqrt(residual.square() + delta.square())
    matrix = residual.new_empty(3, 3)
    gradient = residual.new_empty(3)
    for i in range(3):
        weighted_column = weights * jacobian[:, i]
        gradient[i] = torch.dot(weighted_column, residual) / residual.numel()
        for j in range(i, 3):
            matrix[i, j] = torch.dot(weighted_column, jacobian[:, j]) / residual.numel()
            matrix[j, i] = matrix[i, j]
    return matrix.to(device='cpu', dtype=torch.float64), gradient.to(device='cpu', dtype=torch.float64)


def damped_step(matrix: torch.Tensor, gradient: torch.Tensor, point: torch.Tensor,
                radius: float, cfg: GNCubeConfig):
    """Small CPU-float64 LM solve, freezing only outward-pointing face axes."""
    active = ((point >= BOUND_M) & (gradient < 0)) | ((point <= -BOUND_M) & (gradient > 0))
    free = (~active).nonzero().flatten()
    step = torch.zeros(3, dtype=torch.float64)
    if not free.numel():
        return step, 0.0
    reduced = matrix[free][:, free]
    eigenvalues, eigenvectors = torch.linalg.eigh((reduced + reduced.T) * 0.5)
    eigenvalues.clamp_(min=0)
    rotated = eigenvectors.T @ gradient[free]
    floor = max(cfg.lambda_floor_rel * float(eigenvalues.max()), 1e-30)

    def components(damping):
        return -rotated / (eigenvalues + damping)

    damping = floor
    if float(components(floor).norm()) > radius:
        low, high = floor, max(floor, float(rotated.norm()) / radius)
        for _ in range(cfg.damping_bisections):
            middle = (low + high) * 0.5
            if float(components(middle).norm()) > radius:
                low = middle
            else:
                high = middle
        damping = high
    step[free] = eigenvectors @ components(damping)
    return step, damping


class _NonFinite(Exception):
    """Internal fail-closed signal, not a catch-all for model/programming errors."""


class GNCubePlanner:
    name = 'gn_cube_a'

    def __init__(self, config: GNCubeConfig = GNCubeConfig()):
        self.config = config
        self.reset(0)

    def reset(self, seed: int) -> None:
        # No seed, radius, tensor, or history is carried between plan calls.
        self.log: dict = {}

    @torch.no_grad()
    def plan(self, ctx: PlanContext) -> torch.Tensor:
        started = time.perf_counter()
        cfg = self.config
        deadline = started + cfg.time_fraction * max(0.0, float(ctx.budget_s))
        zero = ctx.goal.new_zeros(3)
        point, features, cost, cost_zero = zero.clone(), None, None, None
        radius = cfg.radius_init_m
        log: dict = dict(rows_used=0, rollout_calls=0, linearisations=0,
                   accepted_steps=0, rejected_steps=0, cost_at_zero=None,
                   final_cost=None, final_point=zero.tolist(), iterations=[], stop_reason=None)
        self.log = log

        def finite(value):
            if not bool(torch.isfinite(value).all()):
                raise _NonFinite

        def evaluate(points):
            # All calls, including base-only fallback, go through this counter.
            if log['rows_used'] + len(points) > cfg.max_rows:
                raise AssertionError('Internal rollout row-budget violation')
            log['rows_used'] += len(points)
            log['rollout_calls'] += 1
            predictions = ctx.rollout(points[:, None, :])[:, 0, :]
            costs = ctx.cost(predictions).reshape(-1)
            return predictions, costs

        def available(rows):
            if time.perf_counter() >= deadline:
                log['stop_reason'] = 'time_budget'
                return False
            if log['rows_used'] + rows > cfg.max_rows:
                log['stop_reason'] = 'row_budget'
                return False
            return True

        try:
            finite(ctx.goal)
            # If a full stencil cannot fit, still establish the native null cost.
            if cfg.max_rows < 7 or time.perf_counter() >= deadline:
                predictions, costs = evaluate(zero[None])
                finite(predictions)
                finite(costs)
                cost_zero = cost = float(costs[0])
                log['stop_reason'] = 'row_budget' if cfg.max_rows < 7 else 'time_budget'
            else:
                for _ in range(cfg.max_linearisations):
                    needed = 7 if features is None else 6
                    if not available(needed):
                        break
                    points, widths = stencil(point, cfg.fd_step_m)
                    if features is None:
                        points = torch.cat((points, point[None]), dim=0)
                    predictions, costs = evaluate(points)
                    # Record a valid L(0) even if a different stencil row is NaN.
                    if features is None and bool(torch.isfinite(predictions[-1]).all()) and bool(torch.isfinite(costs[-1])):
                        cost_zero = cost = float(costs[-1])
                    finite(predictions)
                    finite(costs)
                    if features is None:
                        features = predictions[-1].clone()
                    if not available(0):
                        break
                    jacobian = ctx.goal.new_empty(ctx.goal.numel(), 3)
                    for i in range(3):
                        jacobian[:, i] = (predictions[2 * i + 1] - predictions[2 * i]) / widths[i]
                    del predictions, costs
                    finite(jacobian)
                    matrix, gradient = normal_equations(features - ctx.goal, jacobian, cfg)
                    del jacobian
                    finite(matrix)
                    finite(gradient)
                    log['linearisations'] += 1
                    accepted = False
                    while available(1):
                        step, damping = damped_step(matrix, gradient, point.double().cpu(), radius, cfg)
                        finite(step)
                        if not math.isfinite(damping):
                            raise _NonFinite
                        candidate = (point + step.to(point)).clamp(-BOUND_M, BOUND_M)
                        finite(candidate)
                        step_norm = float((candidate - point).norm())
                        if step_norm < cfg.step_tol_m:
                            log['stop_reason'] = 'step_tolerance'
                            break
                        # The tiny solve may itself consume the remaining time.
                        if not available(1):
                            break
                        trial_features, trial_costs = evaluate(candidate[None])
                        finite(trial_features)
                        finite(trial_costs)
                        trial_cost = float(trial_costs[0])
                        accepted = trial_cost < cost
                        log['iterations'].append(dict(point=candidate.tolist(), cost=trial_cost,
                                                      radius_m=radius, damping=damping, accepted=accepted))
                        if accepted:
                            point, features, cost = candidate, trial_features[0], trial_cost
                            log['accepted_steps'] += 1
                            radius = min(cfg.radius_max_m, radius * cfg.radius_grow)
                            break
                        log['rejected_steps'] += 1
                        if radius <= cfg.radius_min_m:
                            log['stop_reason'] = 'minimum_radius'
                            break
                        radius = max(cfg.radius_min_m, cfg.radius_shrink * min(radius, step_norm))
                    if not accepted:
                        break
                if log['stop_reason'] is None:
                    log['stop_reason'] = 'max_linearisations'
            if log['rows_used'] == 0:
                # Deadline may expire after the initial branch, before its
                # stencil reservation. L(0) remains a mandatory measurement.
                predictions, costs = evaluate(zero[None])
                finite(predictions)
                finite(costs)
                cost_zero = cost = float(costs[0])
        except _NonFinite:
            point, cost = zero, cost_zero
            log['stop_reason'] = 'nonfinite'
        except torch.linalg.LinAlgError:
            point, cost = zero, cost_zero
            log['stop_reason'] = 'linear_algebra_failure'

        if cost is None or cost_zero is None or cost > cost_zero:
            point, cost = zero, cost_zero
        result = point.clamp(-BOUND_M, BOUND_M).reshape(1, 3).detach()
        log.update(cost_at_zero=cost_zero, final_cost=cost, final_point=result[0].tolist(),
                   final_radius_m=radius, elapsed_s=time.perf_counter() - started)
        ctx.extras['planner_log'] = log
        return result


Planner = GNCubePlanner
