"""`gn_cube_b` - damped Gauss-Newton / IRLS on the model's action cube, finite-difference Jacobian.

Problem.  The model sees each commanded component as clamp(a_i / c, -1, 1), c = 0.05 m, so the
predicted native cost

    L(a) = ctx.cost(z(a)),   z(a) = ctx.rollout(a) in R^P,   L = mean |z - g| (per view, then .5/.5)

is a function on the cube Q = [-c, c]^3 and is constant along an axis beyond a face. The planner
searches  argmin_{a in Q} L(a)  from a = 0, deterministically, and never evaluates outside Q.

Linearisation at a (one batched rollout: 6 rows, 7 when z(a) is not yet known).
    r = z(a) - g                                   residual, (P,)
    J[:, i] = (z(a + (hi_i - a_i) e_i) - z(a + (lo_i - a_i) e_i)) / (hi_i - lo_i)
Stencil rule (per axis, h = min(s, c), s = fd_step_m):
    lo_i = min(max(a_i - h, -c), c - 2h),   hi_i = min(lo_i + 2h, c)
i.e. the central stencil [a_i - h, a_i + h] of full width 2h, translated inwards by the amount it
would stick out of Q. Both points are in Q, always 2h apart; in the interior this is the central
difference, at a face it is the secant over the 2h next to the face (centred h inside the face).
At a = 0 the stencil is symmetric, so a response that depends on |a| alone cancels in J.

IRLS for the L1 cost (plain mean over the P entries; the two views are equal-sized, weight .5):
    w_k = 1 / sqrt(r_k^2 + delta^2),   delta = delta_frac * median_k |r_k|
    A = J^T W J / P  (3 x 3),   b = J^T W r / P  (3,)        accumulated on the feature device
Sum_k w_k (r_k + J_k d)^2 / 2 is a quadratic majoriser of the smoothed L1 along the linearisation.

Step (3 x 3, float64, CPU).  For damping lambda the candidate is the exact box-constrained minimiser
    d(lambda) = argmin_{a + d in Q}  d^T (A + lambda I) d / 2 + b^T d
(27 active sets enumerated: every component is free, or fixed at its lower / upper face; with all
components free this is d = -(A + lambda I)^{-1} b). A component sitting on a face whose gradient
points outwards gets d_i = 0 and the remaining components are solved exactly given that, so a pinned
axis never blocks the free ones, and a + d is already the projection onto Q (it is clamped again).
Levenberg-Marquardt / trust radius: lambda(Delta) is the smallest lambda >= lambda_floor with
||d(lambda)|| <= Delta (||d(lambda)|| is non-increasing in lambda; bisection in log lambda).

Acceptance and radius.  For radii Delta, q Delta, q^2 Delta (candidates_per_try values, q =
candidate_ratio, duplicates and steps shorter than step_tol_m dropped) the candidates a + d are
evaluated in ONE rollout call; the candidate with the lowest native cost is accepted iff
    L(a + d) < L(a)                     (strict; native ctx.cost, not the IRLS model)
  accepted, largest candidate, ||d|| >= 0.9 Delta :  Delta <- min(grow * Delta, radius_max_m)
  accepted otherwise                              :  Delta <- max(radius of the accepted candidate, radius_min_m)
  none accepted                                   :  Delta <- shrink * (shortest evaluated ||d||), same linearisation
After an acceptance the model is re-linearised at the new point (z there is already known: 6 rows).

Face snap.  With a non-zero residual at the constrained minimum (goal outside Q) GN converges only
linearly towards a face and would stop a fraction of a millimetre short of it. So one extra candidate
rides in the same rollout call: the largest candidate (or a itself when every step is below
step_tol_m) with each component closer than face_snap_m to a face moved onto that face. It is
accepted under the same strict native-cost test, leaves Delta unchanged, and is tried once per point.

Stop: longest candidate step < step_tol_m (converged) | Delta < radius_min_m after a rejection |
rollout rows would exceed max_rows | max_linearisations | elapsed > time_frac * ctx.budget_s.
The iterate only moves on strict decrease from a = 0, so L(returned) <= L(0). Any non-finite
feature or cost returns exact zeros. ctx.feedback, ctx.warm_start and the seed are not read.

The acceptance test compares native costs, so a goal-independent term that lowers L with |a| is not
cancelled there (it is cancelled in J at a = 0 only).
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from bench.interface import PlanContext

Vec = List[float]
Mat = List[List[float]]


@dataclass(frozen=True)
class GNCubeBConfig:
    box_m: float = 0.05              # c: half-width of the cube Q (model-side clamp)
    fd_step_m: float = 0.02          # s: finite-difference half-width (centimetre scale)
    max_rows: int = 150              # hard cap on rollout rows per plan()
    delta_frac: float = 0.1          # delta = delta_frac * median |r|
    radius_init_m: float = 0.04      # initial trust radius Delta
    radius_max_m: float = 0.10
    radius_min_m: float = 0.001      # stop when a rejection pushes Delta below this
    radius_grow: float = 2.0
    radius_shrink: float = 0.5
    candidates_per_try: int = 3      # step lengths of one linearisation evaluated per rollout call
    candidate_ratio: float = 0.5     # q: successive candidate radii Delta * q^k
    lambda_floor_rel: float = 1e-9   # lambda >= floor * trace(A) / 3
    step_tol_m: float = 0.0005       # convergence: step below 0.5 mm
    face_snap_m: float = 0.002       # extra candidate: components this close to a face put on it (0 = off)
    max_linearisations: int = 20     # 1 = a single GN step from a = 0
    time_frac: float = 0.5           # stop when elapsed > time_frac * ctx.budget_s


# ---------------------------------------------------------------- small pure functions

def stencil_bounds(a: Sequence[float], step: float, box: float) -> List[Tuple[float, float]]:
    """Per-axis evaluation abscissae (lo_i, hi_i), both in [-box, box], hi_i - lo_i = 2 min(step, box)."""
    h = min(step, box)
    out = []
    for ai in a:
        lo = min(max(ai - h, -box), box - 2.0 * h)
        out.append((lo, min(lo + 2.0 * h, box)))
    return out


def stencil_points(a: Sequence[float], bounds: Sequence[Tuple[float, float]]) -> List[Vec]:
    """Rows [hi_0, hi_1, hi_2, lo_0, lo_1, lo_2]: a with one component replaced."""
    rows = []
    for side in (1, 0):
        for i, b in enumerate(bounds):
            p = list(a)
            p[i] = b[side]
            rows.append(p)
    return rows


def irls_normal_equations(r: torch.Tensor, J: torch.Tensor, delta_frac: float) -> Tuple[Mat, Vec, float]:
    """A = J^T W J / P and b = J^T W r / P as float64 Python lists; r (P,), J (3, P) (rows = columns of dz/da).

    Only (P,) temporaries are created; the sums are accumulated in float64 on the feature device.
    """
    n, P = J.shape
    delta = max(delta_frac * float(r.abs().median()), 1e-30)
    w = torch.rsqrt(r * r + delta * delta)
    A = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for i in range(n):
        wj = J[i] * w
        b[i] = float((wj * r).sum(dtype=torch.float64)) / P
        for j in range(i, n):
            A[i][j] = A[j][i] = float((wj * J[j]).sum(dtype=torch.float64)) / P
    return A, b, delta


def solve_small(M: Mat, y: Vec) -> Optional[Vec]:
    """Gaussian elimination with partial pivoting for an n x n system, n <= 3. None if singular."""
    n = len(y)
    T = [list(M[i]) + [y[i]] for i in range(n)]
    for k in range(n):
        p = max(range(k, n), key=lambda i: abs(T[i][k]))
        if abs(T[p][k]) < 1e-300:
            return None
        T[k], T[p] = T[p], T[k]
        for i in range(k + 1, n):
            f = T[i][k] / T[k][k]
            for j in range(k, n + 1):
                T[i][j] -= f * T[k][j]
    x = [0.0] * n
    for k in range(n - 1, -1, -1):
        x[k] = (T[k][n] - sum(T[k][j] * x[j] for j in range(k + 1, n))) / T[k][k]
    return x


def box_qp(A: Mat, b: Vec, lam: float, lo: Vec, hi: Vec) -> Vec:
    """argmin_{lo <= d <= hi} d^T (A + lam I) d / 2 + b^T d, exact, by enumerating the 3^n active sets."""
    n = len(b)
    H = [[A[i][j] + (lam if i == j else 0.0) for j in range(n)] for i in range(n)]
    best, best_q = [min(max(0.0, lo[i]), hi[i]) for i in range(n)], math.inf
    for state in itertools.product((0, -1, 1), repeat=n):       # 0 free, -1 at lo, +1 at hi
        d = [lo[i] if s < 0 else hi[i] if s > 0 else 0.0 for i, s in enumerate(state)]
        free = [i for i, s in enumerate(state) if s == 0]
        if free:
            rhs = [-(b[i] + sum(H[i][j] * d[j] for j in range(n) if state[j] != 0)) for i in free]
            x = solve_small([[H[i][j] for j in free] for i in free], rhs)
            if x is None or any(not (lo[i] <= xi <= hi[i]) for i, xi in zip(free, x)):
                continue
            for i, xi in zip(free, x):
                d[i] = xi
        q = sum(d[i] * (0.5 * sum(H[i][j] * d[j] for j in range(n)) + b[i]) for i in range(n))
        if q < best_q:
            best, best_q = d, q
    return best


def norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def trust_step(A: Mat, b: Vec, lo: Vec, hi: Vec, radius: float, floor_rel: float) -> Tuple[Vec, float]:
    """(d, lambda): box-constrained damped GN step with the smallest lambda >= floor giving ||d|| <= radius."""
    n = len(b)
    trace = sum(A[i][i] for i in range(n))
    if not all(math.isfinite(x) for x in b) or not math.isfinite(trace) or trace <= 0.0 or norm(b) == 0.0:
        return [0.0] * n, 0.0
    lam = floor_rel * trace / n
    d = box_qp(A, b, lam, lo, hi)
    if norm(d) <= radius:
        return d, lam
    lg_lo, lg_hi = math.log(lam), math.log(max(2.0 * norm(b) / radius, 2.0 * lam))  # ||d|| <= 2||b||/lambda
    for _ in range(60):
        mid = 0.5 * (lg_lo + lg_hi)
        if norm(box_qp(A, b, math.exp(mid), lo, hi)) > radius:
            lg_lo = mid
        else:
            lg_hi = mid
    lam = math.exp(lg_hi)
    return box_qp(A, b, lam, lo, hi), lam


def project_box(a: Sequence[float], box: float) -> Vec:
    return [min(max(x, -box), box) for x in a]


# ---------------------------------------------------------------- planner

class GNCubeB:
    name = 'gn_cube_b'

    def __init__(self, config: GNCubeBConfig = GNCubeBConfig()):
        self.cfg = config
        self.reset(0)

    def reset(self, seed: int) -> None:
        """Deterministic planner: the seed is ignored; all per-unit state is the last log."""
        self.log: dict = {}
        self._rows = 0
        self._calls = 0

    # -- the only place that touches ctx.rollout ---------------------------------------------------
    def _features(self, ctx: PlanContext, points: Sequence[Sequence[float]]) -> torch.Tensor:
        """(B, P) features for B points of Q; counts rows and calls."""
        acts = torch.tensor([list(p) for p in points], device=ctx.goal.device, dtype=ctx.goal.dtype)
        self._rows += acts.shape[0]
        self._calls += 1
        with torch.no_grad():
            return ctx.rollout(acts[:, None, :])[:, 0]

    def _finish(self, ctx, a, status, t0, L0, L, n_lin, n_acc, n_rej, trace):
        g = ctx.goal
        if status == 'nonfinite' or not all(math.isfinite(x) for x in a) or not (L <= L0):
            a, L = [0.0, 0.0, 0.0], L0
        out = torch.tensor([a], device=g.device, dtype=g.dtype).clamp(-self.cfg.box_m, self.cfg.box_m)
        self.log = dict(
            planner=self.name, status=status, rows=self._rows, rollout_calls=self._calls,
            linearisations=n_lin, accepted_steps=n_acc, rejected_steps=n_rej,
            cost_zero=L0 if math.isfinite(L0) else None, cost_final=L if math.isfinite(L) else None,
            point_final=out[0].tolist(), iterations=trace, elapsed_s=time.perf_counter() - t0,
            config=asdict(self.cfg),
        )
        ctx.extras['planner_log'] = dict(self.log)
        return out

    def plan(self, ctx: PlanContext) -> torch.Tensor:
        t0 = time.perf_counter()
        cfg, g, c = self.cfg, ctx.goal, self.cfg.box_m
        self._rows = self._calls = 0
        a, z = [0.0, 0.0, 0.0], None
        L0 = L = math.nan
        radius = min(cfg.radius_init_m, cfg.radius_max_m)
        n_lin = n_acc = n_rej = 0
        trace: list = []
        A: Mat = []
        b: Vec = []
        linearised = False
        snap_tried: set = set()

        def done(status):
            return self._finish(ctx, a, status, t0, L0, L, n_lin, n_acc, n_rej, trace)

        def out_of_time():
            return time.perf_counter() - t0 > cfg.time_frac * ctx.budget_s

        while True:
            if not linearised:                              # (re-)linearise at a
                if n_lin >= max(1, cfg.max_linearisations):
                    return done('max_linearisations')
                bounds = stencil_bounds(a, cfg.fd_step_m, c)
                pts = stencil_points(a, bounds) + ([a] if z is None else [])
                if self._rows + len(pts) > cfg.max_rows or (n_lin > 0 and out_of_time()):
                    return done('max_rows' if self._rows + len(pts) > cfg.max_rows else 'time')
                Z = self._features(ctx, pts)
                n_lin += 1
                if not bool(torch.isfinite(Z).all()):
                    return done('nonfinite')
                if z is None:
                    z = Z[6]
                    L0 = L = float(ctx.cost(z))
                    if not math.isfinite(L0):
                        return done('nonfinite')
                    trace.append(dict(point=list(a), cost=L, radius=None, lam=None, accepted=True, kind='start'))
                J = Z[:3] - Z[3:6]                          # (3, P)
                for i, (lo_i, hi_i) in enumerate(bounds):
                    J[i] /= (hi_i - lo_i)
                A, b, _ = irls_normal_equations(z - g, J, cfg.delta_frac)
                linearised = True
                del Z, J
                if not all(math.isfinite(x) for row in A for x in row) or not all(math.isfinite(x) for x in b):
                    return done('nonfinite')

            # candidates of this linearisation: radii Delta * q^k, largest first
            lo, hi = [-c - x for x in a], [c - x for x in a]
            cands = []                                       # (point, step norm, radius, lambda, kind)
            for k in range(max(1, cfg.candidates_per_try)):
                rad = radius * cfg.candidate_ratio ** k
                d, lam = trust_step(A, b, lo, hi, rad, cfg.lambda_floor_rel)
                p = project_box([x + di for x, di in zip(a, d)], c)
                sn = norm([pi - x for pi, x in zip(p, a)])
                if sn < cfg.step_tol_m or any(max(abs(pi - qi) for pi, qi in zip(p, q[0])) < 1e-7 for q in cands):
                    continue
                cands.append((p, sn, rad, lam, 'candidate'))
            base = cands[0][0] if cands else a
            snap = [math.copysign(c, x) if 0.0 < c - abs(x) < cfg.face_snap_m else x for x in base]
            if snap != base and tuple(snap) not in snap_tried and all(snap != q[0] for q in cands):
                cands.append((snap, norm([pi - x for pi, x in zip(snap, a)]), None, None, 'face_snap'))
            if not cands:
                return done('converged')
            cands = cands[: max(0, cfg.max_rows - self._rows)]
            if not cands or out_of_time():
                return done('max_rows' if not cands else 'time')

            Zc = self._features(ctx, [q[0] for q in cands])
            costs = [float(x) for x in ctx.cost(Zc)]
            if not bool(torch.isfinite(Zc).all()) or not all(math.isfinite(x) for x in costs):
                return done('nonfinite')
            best = min(range(len(cands)), key=lambda i: costs[i])   # ties -> largest radius
            ok = costs[best] < L
            for i, (p, sn, rad, lam, kind) in enumerate(cands):
                trace.append(dict(point=p, cost=costs[i], radius=rad, lam=lam,
                                  accepted=bool(ok and i == best), kind=kind))
            if ok:
                p, sn, rad, _, kind = cands[best]
                a, L, z = p, costs[best], Zc[best].clone()
                n_acc += 1
                n_rej += len(cands) - 1
                if kind == 'face_snap':
                    pass                                     # Delta unchanged
                elif best == 0 and sn >= 0.9 * radius:
                    radius = min(cfg.radius_grow * radius, cfg.radius_max_m)
                else:
                    radius = max(rad, cfg.radius_min_m)
                linearised = False
                snap_tried = set()
            else:
                n_rej += len(cands)
                snap_tried.update(tuple(q[0]) for q in cands if q[4] == 'face_snap')
                steps = [q[1] for q in cands if q[4] == 'candidate']
                if not steps:                                # only the face snap was left, and it lost
                    return done('converged')
                radius = cfg.radius_shrink * min(min(steps), radius)
                if radius < cfg.radius_min_m:
                    return done('min_radius')
            del Zc


Planner = GNCubeB
