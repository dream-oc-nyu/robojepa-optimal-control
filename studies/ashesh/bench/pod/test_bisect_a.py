"""CPU-only length-rule checks; no native CEM or learned model is run."""
import json
import unittest
from dataclasses import FrozenInstanceError
from fractions import Fraction

import numpy as np
import torch

from bench.interface import Feedback, PlanContext
from planners import bisect_a
from planners.bisect_a import LengthRule, LengthRuleConfig
from planners.cem_release import CEMRelease
from planners.gn_cube_a import Planner as GNA
from planners.gn_cube_b import Planner as GNB


class ScriptedPlanner:
    name = 'scripted'

    def __init__(self, commands):
        self.commands = commands
        self.reset(0)

    def reset(self, seed):
        self.seed = seed
        self.index = 0

    def plan(self, ctx):
        ctx.extras['planner_log'] = {'base_key': 'retained', 'base_tick': ctx.tick}
        command = self.commands[self.index]
        self.index += 1
        return command


class SignOracle:
    name = 'sign_oracle'

    def __init__(self, start_cm, always_saturated=False):
        self.start = [Fraction(str(x)) / 100 for x in start_cm]
        self.always_saturated = always_saturated

    def reset(self, seed):
        self.remaining = self.start[:]

    def plan(self, ctx):
        # Exact plant state keeps the mathematical 5 cm boundary unambiguous.
        command = []
        for r in self.remaining:
            sign = (r > 0) - (r < 0)
            a = sign * Fraction(1, 10) if self.always_saturated or abs(r) >= Fraction(1, 20) else r
            command.append(float(a))
        return torch.tensor([command], dtype=torch.float64)

    def apply(self, out):
        self.remaining = [r - Fraction(str(float(e))) for r, e in zip(self.remaining, out)]
        return torch.tensor([float(r * 100) for r in self.remaining], dtype=torch.float64)


def context(tick=0, executed=None, commanded=None):
    def forbidden_model(*args):
        raise AssertionError('The wrapper must not call a model')

    feedback = None
    if executed is not None:
        executed = torch.as_tensor(executed, dtype=torch.float64)
        commanded = executed if commanded is None else torch.as_tensor(commanded, dtype=torch.float64)
        feedback = Feedback(commanded, executed, 1.0, 0.5, 0.5, 1.0)
    return PlanContext(forbidden_model, forbidden_model, torch.zeros(1), 1,
                       0.30, 1.0, tick, feedback=feedback)


def scripted(commands, bisect=True, dtype=torch.float64, shrink='flip'):
    base = ScriptedPlanner([torch.tensor([row], dtype=dtype) for row in commands])
    rule = LengthRule(base, LengthRuleConfig(bisect=bisect, shrink=shrink))
    rule.reset(0)
    return rule


class LengthRuleTests(unittest.TestCase):
    def test_unsaturated_passes_through_bit_exactly(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                source = torch.tensor([[0.0123456789, -0.04998, -0.0]], dtype=dtype)
                rule = LengthRule(ScriptedPlanner([source]))
                rule.reset(0)
                result = rule.plan(context())
                self.assertEqual(result.numpy().tobytes(), source.numpy().tobytes())

    def test_saturated_lengths_are_replaced_not_capped(self):
        for dtype in (torch.float32, torch.float64):
            for sign in (-1, 1):
                with self.subTest(dtype=dtype, sign=sign):
                    rule = scripted([[sign * x for x in (0.05, 0.07, 0.17)]], dtype=dtype)
                    actual = rule.plan(context())
                    expected = torch.full((1, 3), sign * 0.10, dtype=dtype)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_saturation_tolerance_includes_boundary(self):
        threshold = 0.05 - 1e-5
        rule = scripted([[threshold, -threshold, threshold - 1e-9]])
        actual = rule.plan(context())
        torch.testing.assert_close(actual, torch.tensor([[0.10, -0.10, threshold - 1e-9]],
                                                       dtype=torch.float64), rtol=0, atol=0)

    def test_alternating_saturated_signs_halve_length_to_floor(self):
        signs = [1, -1, 1, -1, 1, -1]
        rule = scripted([[s * 0.05, 0.0, 0.0] for s in signs])
        lengths = [0.10, 0.05, 0.025, 0.0125, 0.01, 0.01]
        for tick, (sign, length) in enumerate(zip(signs, lengths)):
            ctx = context(tick, None if tick == 0 else [0.22, 0.0, 0.0])
            out = rule.plan(ctx)
            self.assertEqual(float(out[0, 0]), sign * length)
            log = ctx.extras['planner_log']['length_rule']
            self.assertEqual(log['length_before'],
                             [lengths[max(0, tick - 1)], 0.10, 0.10])
            self.assertEqual(log['length_after'], [length, 0.10, 0.10])
            self.assertEqual(log['flip'], [tick > 0, False, False])
            self.assertEqual(log['prev_sat_sign'],
                             [0.0 if tick == 0 else float(signs[tick - 1]), 0.0, 0.0])

    def test_both_schedules_on_reference_sign_sequences(self):
        cases = (
            ([1, -1, -1, -1, -1, -1],
             [0.10, 0.05, 0.05, 0.05, 0.05, 0.05],
             [0.10, 0.05, 0.025, 0.0125, 0.01, 0.01]),
            ([1, 1, -1, -1, 1],
             [0.10, 0.10, 0.05, 0.05, 0.025],
             [0.10, 0.10, 0.05, 0.025, 0.0125]),
        )
        for signs, flip_lengths, saturated_lengths in cases:
            for shrink, lengths in (('flip', flip_lengths), ('saturated', saturated_lengths)):
                with self.subTest(signs=signs, shrink=shrink):
                    rule = scripted([[s * 0.10, 0.0, 0.0] for s in signs], shrink=shrink)
                    for tick, (sign, length) in enumerate(zip(signs, lengths)):
                        out = rule.plan(context(tick, None if tick == 0 else [0.10, 0.0, 0.0]))
                        self.assertEqual(float(out[0, 0]), sign * length)

    def test_saturated_schedule_requires_movement_after_flip(self):
        for stalled in (None, [0.0, 0.0, 0.0], [0.000999, 0.0, 0.0]):
            with self.subTest(stalled=stalled):
                rule = scripted([[s * 0.10, 0.0, 0.0] for s in (1, -1, -1, -1)],
                                shrink='saturated')
                for tick, feedback, expected in (
                        (0, None, 0.10), (1, [0.10, 0.0, 0.0], -0.05),
                        (2, stalled, -0.05), (3, [-0.05, 0.0, 0.0], -0.025)):
                    ctx = context(tick, feedback)
                    self.assertEqual(float(rule.plan(ctx)[0, 0]), expected)
                    log = ctx.extras['planner_log']['length_rule']
                    self.assertEqual(log['shrink'], 'saturated')
                    self.assertEqual(log['rule'], 'halve_on_saturated_after_flip')
                    self.assertEqual(log['flipped_before'], [tick > 1, False, False])
                    self.assertEqual(log['flipped_after'], [tick > 0, False, False])

    def test_saturated_schedule_remembers_flip_across_unsaturated_tick(self):
        rule = scripted([[0.10, 0.0, 0.0], [-0.10, 0.0, 0.0],
                         [0.02, 0.0, 0.0], [-0.10, 0.0, 0.0]], shrink='saturated')
        for tick, expected, length in ((0, 0.10, 0.10), (1, -0.05, 0.05),
                                       (2, 0.02, 0.05), (3, -0.025, 0.025)):
            ctx = context(tick, None if tick == 0 else [0.02, 0.0, 0.0])
            self.assertEqual(float(rule.plan(ctx)[0, 0]), expected)
            log = ctx.extras['planner_log']['length_rule']
            self.assertEqual(log['length_after'], [length, 0.10, 0.10])
            self.assertEqual(log['flipped_after'], [tick > 0, False, False])
            self.assertEqual(log['flip'], [tick == 1, False, False])
            if tick == 3:
                self.assertEqual(log['prev_sat_sign'], [0.0, 0.0, 0.0])

    def test_saturated_schedule_axes_and_reset_are_independent(self):
        rule = scripted([[0.10, 0.10, 0.0], [-0.10, 0.10, 0.0],
                         [-0.10, 0.10, 0.0]], shrink='saturated')
        for tick, x in enumerate((0.10, -0.05, -0.025)):
            ctx = context(tick, None if tick == 0 else [0.10, 0.0, 0.0])
            torch.testing.assert_close(rule.plan(ctx),
                                       torch.tensor([[x, 0.10, 0.0]], dtype=torch.float64),
                                       rtol=0, atol=0)
            self.assertEqual(ctx.extras['planner_log']['length_rule']['flipped_after'],
                             [tick > 0, False, False])
        rule.reset(29)
        ctx = context(0, [-0.025, 0.10, 0.0])
        torch.testing.assert_close(rule.plan(ctx),
                                   torch.tensor([[0.10, 0.10, 0.0]], dtype=torch.float64),
                                   rtol=0, atol=0)
        log = ctx.extras['planner_log']['length_rule']
        self.assertEqual(log['flipped_before'], [False, False, False])
        self.assertEqual(log['flipped_after'], [False, False, False])

    def test_saturated_schedule_disabled_never_flips_or_halves(self):
        rule = scripted([[s * 0.10, -s * 0.10, 0.0] for s in (1, -1, -1, 1)],
                        bisect=False, shrink='saturated')
        for tick, sign in enumerate((1, -1, -1, 1)):
            ctx = context(tick, None if tick == 0 else [0.10, 0.0, 0.0])
            torch.testing.assert_close(rule.plan(ctx),
                                       torch.tensor([[sign * 0.10, -sign * 0.10, 0.0]],
                                                    dtype=torch.float64), rtol=0, atol=0)
            log = ctx.extras['planner_log']['length_rule']
            self.assertEqual(log['length_after'], [0.10, 0.10, 0.10])
            self.assertEqual(log['flipped_before'], [False, False, False])
            self.assertEqual(log['flipped_after'], [False, False, False])
            self.assertEqual(log['flip'], [False, False, False])

    def test_same_sign_saturated_tick_does_not_shorten_length(self):
        rule = scripted([[sign * 0.05, 0.0, 0.0] for sign in (1, -1, -1, 1)])
        for tick, expected in enumerate((0.10, -0.05, -0.05, 0.025)):
            out = rule.plan(context(tick, None if tick == 0 else [0.10, 0.0, 0.0]))
            self.assertEqual(float(out[0, 0]), expected)

    def test_reference_arm_never_changes_length(self):
        rule = scripted([[0.05, 0.0, 0.0], [-0.17, 0.0, 0.0], [0.07, 0.0, 0.0]],
                        bisect=False)
        for tick, expected in enumerate((0.10, -0.10, 0.10)):
            ctx = context(tick, None if tick == 0 else [0.0993, 0.0, 0.0])
            self.assertEqual(float(rule.plan(ctx)[0, 0]), expected)
            log = ctx.extras['planner_log']['length_rule']
            self.assertEqual(log['length_before'], [0.10, 0.10, 0.10])
            self.assertEqual(log['length_after'], [0.10, 0.10, 0.10])
            self.assertEqual(log['flip'], [False, False, False])
            self.assertFalse(log['bisect'])

    def test_axes_are_independent(self):
        rule = scripted([[0.05, 0.05, -0.02], [-0.05, 0.05, 0.02],
                         [0.05, -0.05, 0.05], [0.05, -0.05, -0.05]])
        rule.plan(context())
        for tick, expected, lengths, flips in (
                (1, [-0.05, 0.10, 0.02], [0.05, 0.10, 0.10], [True, False, False]),
                (2, [0.025, -0.05, 0.10], [0.025, 0.05, 0.10], [True, True, False]),
                (3, [0.025, -0.05, -0.05], [0.025, 0.05, 0.05], [False, False, True])):
            ctx = context(tick, [0.08, 0.06, -0.02])
            torch.testing.assert_close(rule.plan(ctx), torch.tensor([expected], dtype=torch.float64),
                                       rtol=0, atol=0)
            log = ctx.extras['planner_log']['length_rule']
            self.assertEqual(log['flip'], flips)
            self.assertEqual(log['length_after'], lengths)

    def test_reset_restores_lengths_and_clears_previous_signs(self):
        rule = scripted([[0.05, 0.05, 0.0], [-0.05, 0.05, 0.0]])
        rule.plan(context())
        rule.plan(context(1, [0.08, 0.06, 0.0]))
        rule.reset(37)
        self.assertEqual(rule.base.seed, 37)
        ctx = context(0, [-0.08, -0.06, 0.0])
        torch.testing.assert_close(rule.plan(ctx), torch.tensor([[0.10, 0.10, 0.0]],
                                                               dtype=torch.float64), rtol=0, atol=0)
        log = ctx.extras['planner_log']['length_rule']
        self.assertEqual(log['prev_sat_sign'], [0.0, 0.0, 0.0])
        self.assertEqual(log['length_before'], [0.10, 0.10, 0.10])
        self.assertEqual(log['length_after'], [0.10, 0.10, 0.10])

    def test_missing_or_subthreshold_feedback_cannot_halve_length(self):
        for executed in (None, [0.0, 0.0, 0.0], [0.000999, 0.0, 0.0],
                         [0.0005, 0.0005, 0.0005]):
            with self.subTest(executed=executed):
                rule = scripted([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0],
                                 [0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]])
                for tick, expected, feedback in (
                        (0, 0.10, None), (1, -0.10, executed),
                        (2, 0.05, [0.10, 0.0, 0.0]), (3, -0.05, executed)):
                    ctx = context(tick, feedback)
                    self.assertEqual(float(rule.plan(ctx)[0, 0]), expected)
                    self.assertEqual(ctx.extras['planner_log']['length_rule']['flip'],
                                     [tick == 2, False, False])

    def test_movement_gate_uses_full_norm_and_inclusive_threshold(self):
        for executed in ([0.001, 0.0, 0.0], [0.0, 0.001, 0.0],
                         [0.0006, 0.0008, 0.0], [0.0007, 0.0007, 0.0007]):
            with self.subTest(executed=executed):
                rule = scripted([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]])
                rule.plan(context())
                ctx = context(1, executed)
                self.assertEqual(float(rule.plan(ctx)[0, 0]), -0.05)
                log = ctx.extras['planner_log']['length_rule']
                self.assertEqual(log['flip'], [True, False, False])
                self.assertEqual(log['length_after'], [0.05, 0.10, 0.10])

    def test_length_is_independent_of_previous_executed_component_size(self):
        for size in (0.22, 0.03):
            with self.subTest(size=size):
                rule = scripted([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0]])
                rule.plan(context())
                self.assertEqual(float(rule.plan(context(1, [size, 0.0, 0.0]))[0, 0]), -0.05)

    def test_flip_compares_base_sign_not_feedback_signs(self):
        for next_sign, expected in ((-1, -0.05), (1, 0.10)):
            with self.subTest(next_sign=next_sign):
                rule = scripted([[0.05, 0.0, 0.0], [next_sign * 0.05, 0.0, 0.0]])
                rule.plan(context())
                ctx = context(1, [-0.08, 0.0, 0.0], commanded=[-0.17, 0.0, 0.0])
                self.assertEqual(float(rule.plan(ctx)[0, 0]), expected)
                self.assertEqual(ctx.extras['planner_log']['length_rule']['flip'][0],
                                 next_sign == -1)

    def test_unsaturated_tick_breaks_flip_history(self):
        rule = scripted([[0.05, 0.0, 0.0], [0.02, 0.0, 0.0], [-0.05, 0.0, 0.0]])
        for tick, expected in enumerate((0.10, 0.02, -0.10)):
            ctx = context(tick, None if tick == 0 else [0.10, 0.0, 0.0])
            self.assertEqual(float(rule.plan(ctx)[0, 0]), expected)
            if tick == 2:
                log = ctx.extras['planner_log']['length_rule']
                self.assertEqual(log['prev_sat_sign'], [0.0, 0.0, 0.0])
                self.assertEqual(log['length_after'], [0.10, 0.10, 0.10])

    def test_shortened_length_persists_across_unsaturated_tick(self):
        for feedback in (None, [-0.04, 0.0, 0.0]):
            with self.subTest(feedback=feedback):
                rule = scripted([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0],
                                 [0.02, 0.0, 0.0], [0.05, 0.0, 0.0]])
                rule.plan(context())
                rule.plan(context(1, [0.08, 0.0, 0.0]))
                for tick, expected in ((2, 0.02), (3, 0.05)):
                    ctx = context(tick, feedback)
                    self.assertEqual(float(rule.plan(ctx)[0, 0]), expected)
                    log = ctx.extras['planner_log']['length_rule']
                    self.assertEqual(log['length_before'], [0.05, 0.10, 0.10])
                    self.assertEqual(log['length_after'], [0.05, 0.10, 0.10])
                    self.assertEqual(log['flip'], [False, False, False])
                    if tick == 3:
                        self.assertEqual(log['prev_sat_sign'], [0.0, 0.0, 0.0])

    def test_tensor_contract_and_future_rows_preserved_without_mutating_base(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                # A noncontiguous base output also remains a valid input.
                source = torch.tensor([[0.05, 0.17], [-0.07, -0.30], [-0.0, 0.02]],
                                      dtype=dtype).T
                original = source.clone()
                rule = LengthRule(ScriptedPlanner([source]))
                rule.reset(0)
                ctx = context()
                ctx.horizon_max = 2
                out = rule.plan(ctx)
                self.assertEqual(out.shape, source.shape)
                self.assertEqual(out.dtype, source.dtype)
                self.assertEqual(out.device, source.device)
                torch.testing.assert_close(out[0], torch.tensor([0.10, -0.10, -0.0], dtype=dtype),
                                           rtol=0, atol=0)
                self.assertEqual(out[1:].numpy().tobytes(), source[1:].numpy().tobytes())
                self.assertEqual(source.numpy().tobytes(), original.numpy().tobytes())

    def test_log_contains_primitives_preserves_base_keys_and_keeps_snapshots(self):
        rule = scripted([[0.05, -0.02, 0.0], [-0.05, -0.02, 0.0], [-0.05, 0.0, 0.0]])
        snapshots = []
        for tick in range(3):
            ctx = context(tick, None if tick == 0 else [0.08, 0.0, 0.0])
            out = rule.plan(ctx)
            log = ctx.extras['planner_log']
            self.assertEqual(log['base_key'], 'retained')
            self.assertEqual(log['base_tick'], tick)
            details = log['length_rule']
            self.assertEqual(set(details), {'base_command', 'saturated', 'prev_sat_sign',
                                           'flip', 'length_before', 'length_after',
                                           'executed_prev', 'out', 'l0_m', 'min_step_m',
                                           'bisect', 'rule', 'shrink',
                                           'flipped_before', 'flipped_after'})
            self.assertEqual(details['base_command'], rule.base.commands[tick][0].tolist())
            self.assertEqual(details['saturated'], [True, False, False])
            self.assertEqual(details['executed_prev'], None if tick == 0 else [0.08, 0.0, 0.0])
            self.assertEqual(details['out'], out[0].tolist())
            self.assertEqual(details['l0_m'], 0.10)
            self.assertEqual(details['min_step_m'], 0.01)
            self.assertEqual(details['rule'], 'halve_on_flip')
            self.assertEqual(details['shrink'], 'flip')
            self.assertEqual(details['flipped_before'], [tick > 1, False, False])
            self.assertEqual(details['flipped_after'], [tick > 0, False, False])
            self.assertIs(details['bisect'], True)
            for value in details.values():
                for item in value if isinstance(value, list) else [value]:
                    self.assertIn(type(item), (float, bool, str, type(None)))
            snapshots.append((log, json.dumps(log, allow_nan=False)))
        for log, snapshot in snapshots:
            self.assertEqual(json.dumps(log, allow_nan=False), snapshot)

    def test_missing_or_non_dict_base_log_is_replaced(self):
        class OtherLogPlanner(ScriptedPlanner):
            def plan(self, ctx):
                out = super().plan(ctx)
                ctx.extras.pop('planner_log')
                if payload is not None:
                    ctx.extras['planner_log'] = payload
                return out

        for payload in (None, 'not a dict', []):
            with self.subTest(payload=payload):
                rule = LengthRule(OtherLogPlanner([torch.tensor([[0.05, 0.0, 0.0]])]))
                rule.reset(0)
                ctx = context()
                rule.plan(ctx)
                self.assertEqual(set(ctx.extras['planner_log']), {'length_rule'})
                json.dumps(ctx.extras['planner_log'], allow_nan=False)

    def test_shrink_config_defaults_and_validation(self):
        self.assertEqual(getattr(LengthRuleConfig(), 'shrink', None), 'flip')
        for shrink in ('flip', 'saturated'):
            self.assertEqual(LengthRuleConfig(shrink=shrink).shrink, shrink)
        for shrink in ('', 'FLIP', 'other', None, 1, []):
            with self.subTest(shrink=shrink), self.assertRaises(ValueError):
                LengthRuleConfig(shrink=shrink)

    def test_config_is_frozen_and_custom_lengths_are_used(self):
        cfg = LengthRuleConfig(l0_m=0.12, sat_m=0.04, sat_tol_m=0.0,
                               min_step_m=0.02, moved_m=0.002, bisect=True)
        with self.assertRaises(FrozenInstanceError):
            setattr(cfg, 'l0_m', 0.20)
        rule = LengthRule(ScriptedPlanner([torch.tensor([[0.04, 0.0, 0.0]], dtype=torch.float64),
                                          torch.tensor([[-0.04, 0.0, 0.0]], dtype=torch.float64)]), cfg)
        rule.reset(0)
        self.assertEqual(float(rule.plan(context())[0, 0]), 0.12)
        self.assertEqual(float(rule.plan(context(1, [0.03, 0.0, 0.0]))[0, 0]), -0.06)

    def test_registry_factories_and_cem_reset_rng_parity(self):
        expected = ['sat10_cem_a', 'bisect_cem_a', 'halvesat_cem_a',
                    'sat10_gna_a', 'bisect_gna_a', 'halvesat_gna_a',
                    'sat10_gnb_a', 'bisect_gnb_a', 'halvesat_gnb_a']
        self.assertEqual(list(bisect_a.PLANNERS), expected)
        for name, factory in bisect_a.PLANNERS.items():
            with self.subTest(name=name):
                planner = factory()
                self.assertEqual(planner.name, name)
                self.assertRegex(name, r'^[a-z0-9_]+$')
                self.assertEqual(planner.config.bisect, not name.startswith('sat10_'))
                self.assertEqual(planner.config.shrink,
                                 'saturated' if name.startswith('halvesat_') else 'flip')
                kind = name.split('_')[1]
                self.assertIsInstance(planner.base, {'cem': CEMRelease, 'gna': GNA, 'gnb': GNB}[kind])
                self.assertIsNot(planner.base, factory().base)
                for seed in (0, 1, 2):
                    planner.reset(seed)
                    if kind == 'cem':
                        baseline = CEMRelease()
                        baseline.reset(seed)
                        self.assertEqual(planner.base.rng[0], baseline.rng[0])
                        np.testing.assert_array_equal(planner.base.rng[1], baseline.rng[1])
                        self.assertEqual(planner.base.rng[2:], baseline.rng[2:])

    def test_sign_oracle_closed_loop_near_and_far_starts(self):
        for start, expected_cm in (
                ((3.0, -2.8, 5.8), [[0, 0, -4.2], [0, 0, 0], [0, 0, 0],
                                    [0, 0, 0], [0, 0, 0]]),
                ((25, -18, 9), [[15, -8, -1], [5, 2, 0], [-5, 0, 0],
                                [0, 0, 0], [0, 0, 0]])):
            with self.subTest(start=start):
                oracle = SignOracle(start)
                rule = LengthRule(oracle)
                rule.reset(0)
                executed = None
                for tick, expected in enumerate(expected_cm):
                    executed = rule.plan(context(tick, executed))[0]
                    remaining_cm = oracle.apply(executed)
                    torch.testing.assert_close(remaining_cm,
                                               torch.tensor(expected, dtype=torch.float64),
                                               rtol=0, atol=1e-12)

    def test_always_saturated_sign_oracle_both_schedules_five_ticks(self):
        for shrink, expected_cm in (('flip', ('2.0', '2.2', '0.8')),
                                    ('saturated', ('0.25', '0.05', '0.55'))):
            with self.subTest(shrink=shrink):
                oracle = SignOracle((3.0, -2.8, 5.8), always_saturated=True)
                rule = LengthRule(oracle, LengthRuleConfig(shrink=shrink))
                rule.reset(0)
                executed = None
                for tick in range(5):
                    executed = rule.plan(context(tick, executed))[0]
                    oracle.apply(executed)
                for remaining, expected in zip(oracle.remaining, expected_cm):
                    self.assertLessEqual(abs(abs(remaining) - Fraction(expected) / 100),
                                         Fraction(1, 10**9))

    def test_always_saturated_sign_oracle_worst_case(self):
        for bisect, expected_cm in ((True, [7.0, 2.0, 3.0, 0.5, 2.0]),
                                   (False, [7.0, 3.0, 7.0, 3.0, 7.0])):
            with self.subTest(bisect=bisect):
                oracle = SignOracle((3.0, 0.0, 0.0), always_saturated=True)
                rule = LengthRule(oracle, LengthRuleConfig(bisect=bisect))
                rule.reset(0)
                executed = None
                for tick, expected in enumerate(expected_cm):
                    executed = rule.plan(context(tick, executed))[0]
                    remaining_cm = oracle.apply(executed)
                    self.assertAlmostEqual(abs(float(remaining_cm[0])), expected, places=12)


if __name__ == '__main__':
    unittest.main()
