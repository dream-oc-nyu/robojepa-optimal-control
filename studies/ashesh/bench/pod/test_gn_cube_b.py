"""CPU torch checks for `gn_cube_b` on toy models that reproduce the model-side clamp.

Toy model: z(a) = tanh(GAIN * W u + bias), u = clamp(a / 0.05 m, -1, 1), P = 4000 float32 features,
native-like cost = .5/.5 weighted per-half mean |z - g|. The toy rollout counts rows and refuses any
row outside Q. Reference minimisers come from a 21^3 grid over Q refined by a local 21^3 grid of
0.5 mm spacing. Nothing here says anything about the real model.
"""
import math
import unittest

import torch

from bench.interface import Feedback, PlanContext
from planners import gn_cube_b
from planners.gn_cube_b import GNCubeB, GNCubeBConfig, Planner

BOX = 0.05
P = 4000
GAIN = 0.6
TABLE = []          # rows printed at the end of the run
NOTES = []


class Toy:
    """z = f(clamp(a / BOX, -1, 1)); goal g = f(a_star / BOX) (a_star may lie outside Q) + goal_noise."""

    def __init__(self, a_star, seed=0, goal_noise=0.0, flat=False, nan=False):
        gen = torch.Generator().manual_seed(seed)
        self.W = torch.randn(P, 3, generator=gen)
        self.bias = 0.5 * torch.randn(P, generator=gen)
        self.flat, self.nan = flat, nan
        self.rows, self.calls = 0, 0
        u_star = torch.tensor([a_star], dtype=torch.float32) / BOX
        self.goal = (self._f(u_star)[0] + goal_noise * torch.randn(P, generator=gen)).float()

    def _f(self, u):
        return torch.tanh(GAIN * u @ self.W.T + self.bias)

    def features(self, a):
        """(B, 3) metres -> (B, P); no counting, no domain assertion (used by the grid reference)."""
        z = self._f((a.float() / BOX).clamp(-1, 1))
        if self.flat:
            z = self._f(torch.zeros_like(a.float()))
        if self.nan:
            z = z * float('nan')
        return z.float()

    def cost(self, z):
        per_view = (z - self.goal).abs().reshape(*z.shape[:-1], 2, -1).mean(-1)
        return (per_view * .5).sum(-1)

    def rollout(self, actions):
        assert actions.ndim == 3 and actions.shape[1:] == (1, 3), actions.shape
        assert not torch.is_grad_enabled()
        assert bool((actions.double().abs() <= BOX + 1e-9).all()), f'row outside Q: {actions.tolist()}'
        self.rows += actions.shape[0]
        self.calls += 1
        return self.features(actions[:, 0])[:, None, :]

    def ctx(self, tick=0, **kw):
        return PlanContext(self.rollout, self.cost, self.goal, 1, 0.30, 216.0, tick, **kw)

    def native(self, a):
        return float(self.cost(self.features(torch.as_tensor(a, dtype=torch.float32).reshape(1, 3)))[0])


class NoisyToy(Toy):
    """Adds (i) deterministic pseudo-noise of amplitude 1e-3 * RMS response at 2 cm and (ii) a
    direction-independent pull towards the goal, z <- g + (z - g)(1 - k(|a|)), k = kappa (1 - exp(-|a|/rho)),
    rho = 3 mm, kappa = |grad L(0)| rho / L(0): cost drop L k is ~0.63 of the directional drop at 3 mm
    and ~0.15 of it at 2 cm."""

    RHO = 0.003

    def __init__(self, a_star, seed=0, goal_noise=0.0):
        super().__init__(a_star, seed=seed, goal_noise=goal_noise)
        gen = torch.Generator().manual_seed(seed + 1000)
        self.dirs = torch.randn(P, 3, generator=gen)
        self.phase = 6.283 * torch.rand(P, generator=gen)
        eye = 0.02 * torch.eye(3)
        zp, zm, z0 = Toy.features(self, eye), Toy.features(self, -eye), Toy.features(self, torch.zeros(1, 3))
        self.amp = 1e-3 * float(((zp - z0) ** 2).mean().sqrt())
        grad = (self.cost(zp) - self.cost(zm)) / 0.04
        self.slope0, self.L0_clean = float(grad.norm()), float(self.cost(z0)[0])
        self.kappa = self.slope0 * self.RHO / self.L0_clean

    def features(self, a):
        ac = a.float().clamp(-BOX, BOX)
        z = Toy.features(self, a)
        k = self.kappa * (1 - torch.exp(-ac.norm(dim=-1, keepdim=True) / self.RHO))
        noise = self.amp * torch.sin(ac @ self.dirs.T * (6.283 / 0.0007) + self.phase)
        return (self.goal + (z - self.goal) * (1 - k) + noise).float()


def grid_min(toy, features=None):
    """(argmin, min) of the native cost over a 21^3 grid on Q, refined on a local 0.5 mm grid."""
    features = features or toy.features

    def search(axes):
        pts = torch.cartesian_prod(*axes)
        costs = torch.cat([toy.cost(features(chunk)) for chunk in pts.split(2048)])
        i = int(costs.argmin())
        return pts[i], float(costs[i])

    lin = torch.linspace(-BOX, BOX, 21, dtype=torch.float64)
    a0, c0 = search([lin, lin, lin])
    off = torch.linspace(-0.005, 0.005, 21, dtype=torch.float64)
    a1, c1 = search([(a0[i] + off).clamp(-BOX, BOX) for i in range(3)])
    return (a1, c1) if c1 < c0 else (a0, c0)


def run(toy, planner=None, **ctx_kw):
    planner = planner or Planner()
    planner.reset(0)
    ctx = toy.ctx(**ctx_kw)
    out = planner.plan(ctx)
    return out, planner, ctx


class GNCubeBTests(unittest.TestCase):
    def check_contract(self, out, toy, planner, ctx):
        self.assertEqual(tuple(out.shape), (1, 3))
        self.assertEqual(out.dtype, toy.goal.dtype)
        self.assertEqual(out.device, toy.goal.device)
        self.assertTrue(bool(torch.isfinite(out).all()))
        self.assertTrue(bool((out.abs() <= BOX).all()), out.tolist())
        log = planner.log
        self.assertLessEqual(log['rows'], 150)
        self.assertEqual(log['rows'], toy.rows)                 # own count == rows actually received
        self.assertEqual(log['rollout_calls'], toy.calls)
        self.assertEqual(ctx.extras['planner_log'], log)
        for key in ('linearisations', 'accepted_steps', 'rejected_steps', 'cost_zero', 'cost_final',
                    'point_final', 'iterations'):
            self.assertIn(key, log)
        self.assertEqual(log['point_final'], out[0].tolist())
        self.assertEqual(sum(1 for it in log['iterations'] if it['kind'] != 'start' and it['accepted']),
                         log['accepted_steps'])

    def solve(self, label, toy, min_frac, features_ref=None):
        """Plan on `toy`; record a table row; return (out, fraction of grid-best decrease, grid argmin)."""
        out, planner, ctx = run(toy)
        self.check_contract(out, toy, planner, ctx)
        a_ref, L_ref = grid_min(toy)
        L0, L = toy.native([0., 0., 0.]), toy.native(out[0])
        self.assertLessEqual(L, L0)
        if planner.log['accepted_steps']:
            # the log holds the cost from a batched row; a 1-row re-evaluation differs by float32 round-off
            self.assertAlmostEqual(L, planner.log['cost_final'], delta=1e-6 * L0)
        frac = (L0 - L) / (L0 - L_ref)
        dist = float((out[0].double() - a_ref).norm())
        log = planner.log
        TABLE.append((label, log['rows'], log['rollout_calls'], log['linearisations'], log['accepted_steps'],
                      log['rejected_steps'], frac, 1e3 * dist, log['status']))
        self.assertGreaterEqual(frac, min_frac, label)
        return out, frac, a_ref

    # 1 ------------------------------------------------------------------------------------------
    def test_1_interior_minimum(self):
        cases = [
            ((0.010, -0.020, 0.035), 0, 0.0),
            ((0.030, 0.030, -0.030), 1, 0.0),
            ((-0.012, 0.004, 0.027), 2, 0.0),
            ((0.040, -0.040, 0.010), 3, 0.0),
            ((0.045, 0.020, -0.045), 4, 0.0),
            ((-0.003, 0.002, 0.004), 5, 0.0),
            ((0.020, -0.035, -0.015), 6, 0.3),      # non-zero residual at the minimum
            ((-0.030, 0.015, 0.025), 7, 1.0),
        ]
        for a_star, seed, gnoise in cases:
            with self.subTest(a_star=a_star):
                toy = Toy(a_star, seed=seed, goal_noise=gnoise)
                label = 'int ' + ','.join(f'{1e2 * x:+.1f}' for x in a_star) + (f' gn{gnoise}' if gnoise else '')
                out, frac, a_ref = self.solve(label, toy, 0.95)
                self.assertTrue(bool((a_ref.abs() < BOX).all()), 'reference minimum must be interior')
                self.assertLess(float((out[0].double() - a_ref).norm()), 0.003)

    # 2 ------------------------------------------------------------------------------------------
    def test_2_minimum_outside_cube(self):
        cases = [
            ((0.150, 0.020, -0.030), 10),
            ((-0.250, 0.010, 0.000), 11),
            ((0.020, -0.100, 0.030), 12),
            ((0.100, -0.200, 0.015), 13),
            ((-0.120, 0.180, -0.030), 14),
            ((0.150, 0.200, 0.250), 15),
            ((-0.100, 0.250, -0.150), 16),      # coupling pulls the constrained minimum of axis 0 inside Q
            ((-0.200, 0.250, -0.150), 18),
            ((-0.250, -0.250, -0.250), 17),
        ]
        at_face_axes = off_face_exterior = 0
        for a_star, seed in cases:
            with self.subTest(a_star=a_star):
                toy = Toy(a_star, seed=seed)
                label = 'ext ' + ','.join(f'{1e2 * x:+.1f}' for x in a_star)
                out, frac, a_ref = self.solve(label, toy, 0.95)
                face = torch.tensor(BOX, dtype=toy.goal.dtype)
                for i, x in enumerate(a_star):
                    if abs(x) > BOX and float(a_ref[i]) == math.copysign(BOX, x):
                        at_face_axes += 1
                        self.assertEqual(float(out[0, i]), math.copysign(float(face), x), f'axis {i}')
                    else:                               # free axis (or reference inside Q) still solved
                        off_face_exterior += abs(x) > BOX
                        self.assertLess(abs(float(out[0, i]) - float(a_ref[i])), 0.003, f'axis {i}')
        # the reference itself sits on the face for every exterior axis except the one flagged above
        self.assertEqual((at_face_axes, off_face_exterior), (18, 1))

    # 3 ------------------------------------------------------------------------------------------
    def test_3_flat_and_nonfinite(self):
        for kw in (dict(flat=True), dict(nan=True)):
            with self.subTest(**kw):
                toy = Toy((0.02, 0.01, -0.03), seed=20, **kw)
                out, planner, ctx = run(toy)
                self.check_contract(out, toy, planner, ctx)
                self.assertTrue(bool((out == 0).all()), out.tolist())
                self.assertEqual(planner.log['accepted_steps'], 0)
        self.assertEqual(planner.log['status'], 'nonfinite')

        class NanAway(Toy):                             # finite at the stencil, NaN on one side
            def features(self, a):
                z = Toy.features(self, a)
                return torch.where(a.float()[:, :1] > 0.021, z * float('nan'), z)
        toy = NanAway((0.04, 0.0, 0.0), seed=21)
        out, planner, ctx = run(toy)
        self.check_contract(out, toy, planner, ctx)
        self.assertTrue(bool((out == 0).all()), out.tolist())

    # 4 ------------------------------------------------------------------------------------------
    def test_4_determinism_and_statelessness(self):
        a_star = (0.03, -0.12, 0.012)
        ref, planner, _ = run(Toy(a_star, seed=30))
        again, _, _ = run(Toy(a_star, seed=30))
        self.assertTrue(torch.equal(ref, again))

        fb = Feedback(torch.tensor([0.3, -0.1, 0.2]), torch.tensor([0.01, 0.0, -0.02]), 1.0, 5.0, 0.1, 0.2)
        warm = torch.tensor([[0.2, 0.2, -0.2]])
        for kw in (dict(feedback=fb, warm_start=warm, tick=3, seed=2), dict(warm_start=torch.zeros(0, 3), tick=4)):
            out, _, _ = run(Toy(a_star, seed=30), **kw)
            self.assertTrue(torch.equal(ref, out))

        # one planner object across ticks/units: no carried state, reset clears the log
        other = Toy((-0.02, 0.04, 0.2), seed=31)
        planner.plan(other.ctx(tick=1, feedback=fb, warm_start=warm))
        toy = Toy(a_star, seed=30)
        self.assertTrue(torch.equal(ref, planner.plan(toy.ctx(tick=2))))
        self.assertTrue(planner.log)
        planner.reset(7)
        self.assertEqual(planner.log, {})
        toy = Toy(a_star, seed=30)
        self.assertTrue(torch.equal(ref, planner.plan(toy.ctx())))

    # 5 ------------------------------------------------------------------------------------------
    def test_5_noise_and_magnitude_term(self):
        for a_star, seed, gnoise in [((0.022, -0.031, 0.012), 40, 0.3), ((-0.035, 0.010, 0.028), 41, 0.3),
                                     ((0.008, 0.015, -0.020), 42, 0.0), ((0.150, -0.020, 0.030), 43, 0.3)]:
            with self.subTest(a_star=a_star):
                toy = NoisyToy(a_star, seed=seed, goal_noise=gnoise)
                label = 'noisy ' + ','.join(f'{1e2 * x:+.1f}' for x in a_star) + (f' gn{gnoise}' if gnoise else '')
                out, frac, a_noisy = self.solve(label, toy, 0.80)
                a_clean, _ = grid_min(toy, features=lambda a: Toy.features(toy, a))
                k = lambda d: toy.kappa * (1 - math.exp(-d / toy.RHO)) * toy.L0_clean
                NOTES.append(
                    f'{label}: |a - clean-model argmin| = {1e3 * float((out[0].double() - a_clean).norm()):.2f} mm, '
                    f'|a - noisy-grid argmin| = {1e3 * float((out[0].double() - a_noisy).norm()):.2f} mm; '
                    f'magnitude term / directional drop = {k(0.003) / (toy.slope0 * 0.003):.2f} at 3 mm, '
                    f'{k(0.02) / (toy.slope0 * 0.02):.2f} at 2 cm; noise amp = {toy.amp:.2e}')
                self.assertLessEqual(toy.native(out[0]), toy.native([0., 0., 0.]))

    # 6 ------------------------------------------------------------------------------------------
    def test_6_row_accounting_and_budget(self):
        for max_rows in (150, 40, 16, 7, 6):
            with self.subTest(max_rows=max_rows):
                toy = Toy((0.2, -0.045, 0.049), seed=50)
                planner = GNCubeB(GNCubeBConfig(max_rows=max_rows))
                out, planner, ctx = run(toy, planner)
                self.check_contract(out, toy, planner, ctx)       # includes rows == toy.rows, all rows in Q
                self.assertLessEqual(toy.rows, max_rows)
                self.assertLessEqual(toy.native(out[0]), toy.native([0., 0., 0.]))
        self.assertEqual(toy.rows, 0)                             # 6 rows cannot hold the first linearisation
        self.assertTrue(bool((out == 0).all()))
        toy = Toy((0.2, -0.045, 0.049), seed=50)                  # time stop: budget 0 -> one linearisation only
        ctx = toy.ctx()
        ctx.budget_s = 0.0
        out = Planner().plan(ctx)
        self.assertEqual((toy.rows, bool((out == 0).all())), (7, True))

    def test_6b_stencil_stays_in_cube(self):
        s = 0.02
        for a in ([0., 0., 0.], [0.05, -0.05, 0.04], [0.031, -0.03, 0.0499], [-0.05, 0.05, 0.05]):
            for (lo, hi), ai in zip(gn_cube_b.stencil_bounds(a, s, BOX), a):
                self.assertTrue(-BOX <= lo < hi <= BOX)
                self.assertAlmostEqual(hi - lo, 2 * s, places=12)
                if abs(ai) <= BOX - s:                            # interior: central
                    self.assertAlmostEqual(lo, ai - s, places=12)
        self.assertEqual(gn_cube_b.stencil_bounds([0.05], s, BOX)[0], (0.05 - 2 * s, 0.05))

    def test_6c_box_qp_active_set(self):
        A = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.5, 1.0]]
        b = [-1.0, 0.3, -0.2]                                    # axis 0 wants +0.5 but sits on its upper face
        d = gn_cube_b.box_qp(A, b, 0.0, [-0.1, -0.1, -0.1], [0.0, 0.1, 0.1])
        self.assertEqual(d[0], 0.0)
        free = gn_cube_b.solve_small([[1.0, 0.5], [0.5, 1.0]], [-0.3, 0.2])
        self.assertAlmostEqual(d[1], max(free[0], -0.1), places=12)
        d, lam = gn_cube_b.trust_step(A, b, [-1.0] * 3, [1.0] * 3, 0.05, 1e-9)
        self.assertLessEqual(gn_cube_b.norm(d), 0.05 * (1 + 1e-9))
        self.assertGreater(gn_cube_b.norm(d), 0.049)

    # 7 ------------------------------------------------------------------------------------------
    def test_7_export(self):
        self.assertIs(Planner, GNCubeB)
        self.assertEqual(Planner.name, 'gn_cube_b')
        planner = Planner()
        self.assertEqual(planner.name, 'gn_cube_b')
        self.assertEqual(planner.cfg, GNCubeBConfig())
        self.assertEqual((planner.cfg.fd_step_m, planner.cfg.max_rows, planner.cfg.box_m), (0.02, 150, 0.05))


def tearDownModule():
    head = f"{'case':<28}{'rows':>5}{'calls':>6}{'lin':>4}{'acc':>4}{'rej':>4}{'frac':>8}{'d_ref mm':>9}  status"
    print('\n' + head + '\n' + '-' * len(head))
    for label, rows, calls, lin, acc, rej, frac, dist, status in sorted(TABLE):
        print(f'{label:<28}{rows:>5}{calls:>6}{lin:>4}{acc:>4}{rej:>4}{frac:>8.4f}{dist:>9.2f}  {status}')
    print('frac = (L(0) - L(planner)) / (L(0) - L(grid best on Q)); d_ref = distance to the refined-grid argmin')
    for note in NOTES:
        print(note)


if __name__ == '__main__':
    unittest.main()
