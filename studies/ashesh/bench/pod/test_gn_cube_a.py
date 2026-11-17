"""CPU-only contract tests; grid evaluation is independent of planner rows."""
from dataclasses import FrozenInstanceError, replace
import importlib
import json
import math
import time
import unittest
from unittest.mock import patch

import torch

from bench.interface import PlanContext
from planners.gn_cube_a import GNCubeConfig, Planner


class FeatureToy:
    """Predictor clamps actions, while a goal may lie beyond its action cube."""

    def __init__(self, target, separable=False):
        generator = torch.Generator().manual_seed(281)
        self.matrix = torch.randn(3072, 3, generator=generator) * 0.40
        if separable:
            self.matrix.zero_()
            axes = torch.arange(3072) % 3
            self.matrix[torch.arange(3072), axes] = torch.linspace(0.25, 0.60, 3072)
        self.bias = torch.linspace(-0.12, 0.12, 3072)
        self.target = torch.tensor(target, dtype=torch.float32)
        self.goal = self.features(self.target[None] / 0.05)[0]
        self.rows = 0
        self.calls = 0
        self.batches = []

    def features(self, normalized):
        return torch.tanh(normalized @ self.matrix.T + self.bias)

    def predict(self, actions):
        return self.features((actions / 0.05).clamp(-1, 1))

    def cost(self, z):
        views = (z - self.goal).abs().reshape(*z.shape[:-1], 2, -1).mean(-1)
        return (views * 0.5).sum(-1)

    def rollout(self, actions):
        assert not torch.is_grad_enabled(), 'rollout must use verified no-grad path'
        assert actions.ndim == 3 and actions.shape[1:] == (1, 3)
        assert actions.dtype == self.goal.dtype and actions.device == self.goal.device
        assert bool(torch.isfinite(actions).all())
        assert bool((actions.abs().double() <= 0.05 + 1e-9).all()), actions
        self.rows += len(actions)
        self.calls += 1
        self.batches.append(actions.clone())
        return self.predict(actions)

    def context(self, **kwargs):
        return PlanContext(self.rollout, self.cost, self.goal, 1, 0.30, 216.0, 0, **kwargs)

    def grid_best(self):
        axis = torch.linspace(-0.05, 0.05, 21)
        grid = torch.cartesian_prod(axis, axis, axis)
        costs = torch.cat([self.cost(self.predict(batch)) for batch in grid.split(256)])
        i = int(costs.argmin())
        return float(costs[i]), grid[i]


class NoisyToy(FeatureToy):
    """L1 kinks, 1e-3 deterministic relative noise and saturating motion reward.

    One third of the entries carry a goal-independent positive residual;
    motion lowers that residual by .06*(1-exp(-||a||/.003)). At 3 mm this
    contribution is comparable to direction gain, but saturates by 2 cm.
    The remaining entries carry identifiable directional features. There
    are equal numbers of both kinds in each camera view.
    """

    def __init__(self, target):
        super().__init__(target)
        self.mask = (torch.arange(3072) % 3 == 0)
        self.phase = torch.arange(3072, dtype=torch.float32) * 0.754877666
        self.goal[self.mask] = 0.0

    def predict(self, actions):
        normalized = (actions / 0.05).clamp(-1, 1)
        clean = self.features(normalized)
        response = clean - self.features(torch.zeros_like(normalized))
        argument = normalized @ normalized.new_tensor([113.0, -157.0, 211.0])
        noise = 1e-3 * response.abs() * torch.sin(argument[..., None] + self.phase)
        z = clean + noise
        magnitude = (normalized * 0.05).norm(dim=-1, keepdim=True)
        z[..., self.mask] = 0.20 - 0.06 * (-torch.expm1(-magnitude / 0.003))
        return z


class RejectionToy(FeatureToy):
    """Even curvature cancels at zero, making the first GN proposal overshoot."""

    def features(self, normalized):
        x = normalized[..., 0:1]
        return (x + 10 * x.square()).expand(*x.shape[:-1], 3072)


class GNCubeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.table = []

    @classmethod
    def tearDownClass(cls):
        print('\ncase                 rows  lin  accept  reject  grid-decrease')
        for name, log, fraction in cls.table:
            print(f'{name:20s} {log["rows_used"]:4d} {log["linearisations"]:4d}'
                  f' {log["accepted_steps"]:7d} {log["rejected_steps"]:7d} {fraction:14.6f}')
        torch.set_num_threads(cls.old_threads)

    def assert_contract(self, toy, planner, ctx, result, max_rows=150):
        self.assertEqual(tuple(result.shape), (1, 3))
        self.assertEqual(result.dtype, ctx.goal.dtype)
        self.assertEqual(result.device, ctx.goal.device)
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertTrue(bool((result.abs() <= 0.05).all()))
        self.assertLessEqual(toy.rows, max_rows)
        self.assertEqual(planner.log['rows_used'], toy.rows)
        self.assertEqual(planner.log['rollout_calls'], toy.calls)
        self.assertEqual(ctx.extras['planner_log'], planner.log)
        self.assertEqual(planner.log['final_point'], result[0].tolist())

    def test_interior(self):
        for name, target in [('interior-small', (0.01, -0.015, 0.02)),
                             ('interior-large', (0.04, 0.03, -0.035)),
                             ('interior-mixed', (-0.025, 0.04, 0.015))]:
            with self.subTest(name=name):
                toy, planner = FeatureToy(target), Planner()
                best, point = toy.grid_best()
                ctx = toy.context()
                result = planner.plan(ctx)
                self.assert_contract(toy, planner, ctx, result)
                c0 = float(toy.cost(toy.predict(torch.zeros(1, 3)))[0])
                final = float(toy.cost(toy.predict(result))[0])
                fraction = (c0 - final) / (c0 - best)
                self.table.append((name, planner.log, fraction))
                self.assertGreaterEqual(fraction, 0.95)
                self.assertLessEqual(float((result[0] - point).norm()), 0.003)
                self.assertEqual(planner.log['cost_at_zero'], c0)
                self.assertEqual(planner.log['final_cost'], final)

    def test_exterior_faces_and_free_axes(self):
        cases = [('exterior-one', (0.10, 0.025, -0.015)),
                 ('exterior-two', (-0.15, 0.25, 0.035)),
                 ('exterior-three', (0.25, -0.20, 0.10))]
        for name, target in cases:
            with self.subTest(name=name):
                toy, planner = FeatureToy(target, separable=True), Planner()
                best, grid_point = toy.grid_best()
                ctx = toy.context()
                result = planner.plan(ctx)
                self.assert_contract(toy, planner, ctx, result)
                expected = toy.target.clamp(-0.05, 0.05)
                saturated = toy.target.abs() > 0.05
                self.assertTrue(torch.equal(result[0, saturated], expected[saturated]))
                self.assertTrue(bool((result[0, ~saturated] - expected[~saturated]).abs().le(0.003).all()))
                self.assertLessEqual(float((result[0] - grid_point).norm()), 0.003)
                fraction = (planner.log['cost_at_zero'] - planner.log['final_cost']) / (planner.log['cost_at_zero'] - best)
                self.table.append((name, planner.log, fraction))
                self.assertGreaterEqual(fraction, 0.95)
                # Every linearisation uses a shifted full-width, in-box stencil,
                # including the axes already pinned at a face.
                for batch in toy.batches:
                    if len(batch) in (6, 7):
                        points = batch[:, 0]
                        for i in range(3):
                            width = points[2 * i + 1, i] - points[2 * i, i]
                            self.assertAlmostEqual(float(width), 0.04, places=7)

    def test_flat_and_zero_residual(self):
        for exact_goal in (False, True):
            with self.subTest(exact_goal=exact_goal):
                toy = FeatureToy((0.02, 0.01, -0.03))
                if exact_goal:
                    toy.goal = toy.features(torch.zeros(1, 3))[0]
                else:
                    toy.predict = lambda actions: toy.features(torch.zeros_like(actions))
                planner, ctx = Planner(), toy.context()
                result = planner.plan(ctx)
                self.assert_contract(toy, planner, ctx, result)
                self.assertTrue(torch.equal(result, torch.zeros_like(result)))
                self.assertEqual(toy.rows, 7)
                self.assertEqual(planner.log['accepted_steps'], 0)

    def test_nonfinite_fails_closed_even_after_progress(self):
        # Base, stencil, later accepted-point relinearisation, and candidate cost.
        for failure in ('all_nan', 'stencil_inf', 'late_nan', 'cost_inf', 'goal_nan'):
            with self.subTest(failure=failure):
                toy = FeatureToy((0.04, 0.03, -0.035))
                original = toy.predict

                def bad_predict(actions):
                    out = original(actions).clone()
                    if failure == 'all_nan':
                        out[:] = math.nan
                    elif failure == 'stencil_inf' and toy.calls == 1:
                        out[0, ..., 0] = math.inf
                    elif failure == 'late_nan' and toy.calls >= 3:
                        out[0, ..., 0] = math.nan
                    return out

                toy.predict = bad_predict
                if failure == 'goal_nan':
                    toy.goal[0] = math.nan
                ctx = toy.context()
                if failure == 'cost_inf':
                    native_cost = ctx.cost
                    ctx.cost = lambda z: native_cost(z) if toy.calls < 2 else torch.full(z.shape[:-1], math.inf)
                planner = Planner()
                result = planner.plan(ctx)
                self.assert_contract(toy, planner, ctx, result)
                self.assertTrue(torch.equal(result, torch.zeros_like(result)))
                self.assertEqual(planner.log['stop_reason'], 'nonfinite')
                json.dumps(planner.log, allow_nan=False)
                if failure == 'late_nan':
                    self.assertGreater(planner.log['accepted_steps'], 0)
                    self.assertEqual(planner.log['final_cost'], planner.log['cost_at_zero'])

    def test_determinism_reset_and_context_independence(self):
        first_toy, second_toy = FeatureToy((0.04, -0.03, 0.015)), FeatureToy((0.04, -0.03, 0.015))
        first, second = Planner(), Planner()
        first_ctx = first_toy.context()
        a = first.plan(first_ctx)
        self.assert_contract(first_toy, first, first_ctx, a)
        second_ctx = second_toy.context(feedback=object(), warm_start=torch.full((1, 3), math.nan), seed=999)
        second_ctx.tick = 4
        b = second.plan(second_ctx)
        self.assert_contract(second_toy, second, second_ctx, b)
        self.assertTrue(torch.equal(a, b))
        first.reset(542)
        self.assertEqual(first.log, {})
        for tick in range(5):
            toy = FeatureToy((0.04, -0.03, 0.015))
            ctx = toy.context(feedback=object(), warm_start=torch.ones(1, 3) * 100)
            ctx.tick = tick
            output = first.plan(ctx)
            self.assert_contract(toy, first, ctx, output)
            self.assertTrue(torch.equal(a, output))
            self.assertTrue(torch.equal(toy.batches[0][-1], torch.zeros(1, 3)))

    def test_noise_and_direction_independent_motion_reward(self):
        toy, planner = NoisyToy((0.03, -0.02, 0.015)), Planner()
        best, _ = toy.grid_best()
        ctx = toy.context()
        result = planner.plan(ctx)
        self.assert_contract(toy, planner, ctx, result)
        cost0 = float(toy.cost(toy.predict(torch.zeros(1, 3)))[0])
        final = float(toy.cost(toy.predict(result))[0])
        self.assertLessEqual(final, cost0)
        fraction = (cost0 - final) / (cost0 - best)
        self.assertGreaterEqual(fraction, 0.80)
        distance_mm = float((result[0] - toy.target).norm()) * 1000
        ratios = []
        direction = toy.target / toy.target.norm()
        for magnitude in (0.003, 0.02):
            at_zero = toy.features(torch.zeros(1, 3))
            at_step = toy.features((magnitude * direction / 0.05)[None])
            directional_gain = float(((at_zero - toy.goal).abs() - (at_step - toy.goal).abs())[..., ~toy.mask].sum() / toy.goal.numel())
            even_gain = (0.06 / 3) * (1 - math.exp(-magnitude / 0.003))
            ratios.append(even_gain / directional_gain)
        self.assertGreater(ratios[0], 0.5)
        self.assertLess(ratios[0], 2.0)
        self.assertLess(ratios[1], 0.3)
        print(f'\nnoise: distance from noise-free target = {distance_mm:.3f} mm; '
              f'grid-decrease = {fraction:.6f}; rows = {toy.rows}; '
              f'even/directional gain at 3 mm, 20 mm = {ratios[0]:.3f}, {ratios[1]:.3f}')

    def test_rejection_retries_and_native_monotonicity(self):
        toy, planner = RejectionToy((0.02, 0, 0)), Planner()
        ctx = toy.context()
        result = planner.plan(ctx)
        self.assert_contract(toy, planner, ctx, result)
        log = planner.log
        self.assertGreater(log['rejected_steps'], 0)
        self.assertGreater(log['accepted_steps'], 0)
        previous_cost = log['cost_at_zero']
        previous_row = None
        for row in log['iterations']:
            if row['accepted']:
                self.assertLess(row['cost'], previous_cost)
                previous_cost = row['cost']
            else:
                self.assertGreaterEqual(row['cost'], previous_cost)
            if previous_row is not None and not previous_row['accepted']:
                self.assertLess(row['radius_m'], previous_row['radius_m'])
            previous_row = row
        self.assertEqual(previous_cost, log['final_cost'])
        self.assertLess(log['final_cost'], 0.05 * log['cost_at_zero'])
        self.assertEqual(len(log['iterations']), log['accepted_steps'] + log['rejected_steps'])
        self.assertEqual(log['rows_used'], 1 + 6 * log['linearisations'] + len(log['iterations']))

    def test_row_limits_and_time_limit(self):
        for max_rows in (1, 6, 7, 8, 13, 14, 15, 27, 150):
            with self.subTest(max_rows=max_rows):
                toy = FeatureToy((0.04, 0.03, -0.035))
                ctx = toy.context()
                planner = Planner(replace(GNCubeConfig(), max_rows=max_rows))
                result = planner.plan(ctx)
                self.assert_contract(toy, planner, ctx, result, max_rows=max_rows)
                self.assertIsNotNone(planner.log['cost_at_zero'])
                self.assertLessEqual(planner.log['final_cost'], planner.log['cost_at_zero'])
        toy = FeatureToy((0.04, 0.03, -0.035))
        original_rollout = toy.rollout

        def slow_rollout(actions):
            result = original_rollout(actions)
            time.sleep(0.02)
            return result

        ctx = toy.context()
        ctx.rollout = slow_rollout
        ctx.budget_s = 0.01
        planner = Planner()
        result = planner.plan(ctx)
        self.assert_contract(toy, planner, ctx, result)
        self.assertEqual(toy.calls, 1)  # no new rollout after an in-flight call overruns
        self.assertEqual(planner.log['stop_reason'], 'time_budget')
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_initial_measurement_survives_deadline_race(self):
        toy, planner = FeatureToy((0.03, -0.02, 0.015)), Planner()
        ctx = toy.context()
        ctx.budget_s = 0.01
        # Time expires between choosing a full initial stencil and reserving it.
        with patch('planners.gn_cube_a.time.perf_counter', side_effect=[0.0, 0.0, 0.02, 0.03]):
            result = planner.plan(ctx)
        self.assert_contract(toy, planner, ctx, result)
        self.assertEqual(toy.rows, 1)
        self.assertEqual(planner.log['cost_at_zero'], float(toy.cost(toy.predict(result))[0]))
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))
        self.assertEqual(planner.log['stop_reason'], 'time_budget')

    def test_export_and_frozen_config(self):
        module = importlib.import_module('planners.gn_cube_a')
        planner = module.Planner()
        self.assertEqual(planner.name, 'gn_cube_a')
        with self.assertRaises(FrozenInstanceError):
            planner.config.max_rows = 100
        for override in ({'fd_step_m': 0.003}, {'max_rows': 0}, {'time_fraction': 0.6}):
            with self.assertRaises(ValueError):
                replace(GNCubeConfig(), **override)
        toy = FeatureToy((0.02, 0.01, -0.03))
        ctx = toy.context()
        result = planner.plan(ctx)
        self.assert_contract(toy, planner, ctx, result)


if __name__ == '__main__':
    unittest.main(verbosity=2)
