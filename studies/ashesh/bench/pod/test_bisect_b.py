"""CPU checks for the per-axis length rule; the base planner is a scripted fake (no CEM, no model)."""
import json
import re
import unittest

import numpy as np
import torch

from bench.interface import Feedback, PlanContext
from planners.bisect_b import PLANNERS, LengthRule, LengthRuleConfig
from planners.cem_release import CEMRelease
from planners.gn_cube_a import Planner as GNCubeA
from planners.gn_cube_b import Planner as GNCubeB

SAT = 0.05000000074505806  # float32 face value returned by the GN planners


class Scripted:
    """Returns prescribed (H, 3) commands in order; optionally leaves its own planner_log."""
    name = 'scripted'

    def __init__(self, commands, dtype=torch.float32, log=None):
        self.commands = [torch.tensor(c, dtype=dtype).reshape(-1, 3) for c in commands]
        self.log = log
        self.seeds = []
        self.k = 0

    def reset(self, seed):
        self.seeds.append(seed)

    def plan(self, ctx):
        if self.log is not None:
            ctx.extras['planner_log'] = dict(self.log)
        out = self.commands[self.k]
        self.k += 1
        return out


def make_ctx(tick, commanded=None, executed=None, extras=None, dtype=torch.float32):
    feedback = None
    if executed is not None:
        feedback = Feedback(commanded=torch.as_tensor(commanded, dtype=dtype),
                            executed=torch.as_tensor(executed, dtype=dtype),
                            cost_before=0.0, cost_after=0.0, cost_predicted=0.0, cost_model_before=0.0)
    return PlanContext(rollout=None, cost=None, goal=torch.zeros(1), horizon_max=1, step_bound_m=0.3,
                       budget_s=1.0, tick=tick, feedback=feedback,
                       extras={} if extras is None else extras)


def run(rule, executed_per_tick, n):
    """Plan n ticks; executed_per_tick[t] is the executed (3,) of tick t (None: executed = commanded)."""
    outs, logs, commanded, executed = [], [], None, None
    for t in range(n):
        ctx = make_ctx(t, commanded, executed)
        out = rule.plan(ctx)
        outs.append(out[0].tolist())
        logs.append(ctx.extras['planner_log']['length_rule'])
        commanded = out[0].clone()
        e = executed_per_tick[t] if t < len(executed_per_tick) else None
        executed = commanded.clone() if e is None else torch.tensor(e, dtype=torch.float32)
    return outs, logs


def rule_for(commands, bisect=True, shrink='flip', **kw):
    rule = LengthRule(Scripted(commands, **kw), LengthRuleConfig(bisect=bisect, shrink=shrink))
    rule.reset(0)
    return rule


def x_cmds(signs):
    """Saturated commands on x with the given signs; 0 stands for an unsaturated 3 cm command."""
    return [[s * SAT if s else 0.03, 0.0, 0.0] for s in signs]


ALTERNATING = x_cmds([1, -1, 1, -1, 1, -1])
L0 = 0.10


class LengthRuleTests(unittest.TestCase):
    def test_01_unsaturated_pass_through_bit_exact(self):
        for dtype in (torch.float32, torch.float64):
            for bisect in (False, True):
                cmd = [[0.0293, -0.0307, 0.04998], [0.0, -0.0499, 1e-9]]
                rule = rule_for([cmd[0], cmd[1]], bisect=bisect, dtype=dtype)
                for t, c in enumerate(cmd):
                    native = torch.tensor([c], dtype=dtype)
                    out = rule.plan(make_ctx(t))
                    self.assertTrue(torch.equal(out, native))
                    self.assertEqual(out.dtype, dtype)
                self.assertEqual(rule.length, [L0] * 3)

    def test_02_saturated_without_history_is_l0(self):
        for bisect in (False, True):
            for raw in (0.05, SAT, 0.07, 0.17):
                with self.subTest(bisect=bisect, raw=raw):
                    rule = rule_for([[raw, -raw, 0.01]], bisect=bisect, dtype=torch.float64)
                    out = rule.plan(make_ctx(0))
                    self.assertEqual(out[0].tolist(), [0.10, -0.10, 0.01])
        # float32: exactly the float32 nearest to 0.10
        out = rule_for([[0.17, -SAT, 0.0]]).plan(make_ctx(0))
        self.assertTrue(torch.equal(out, torch.tensor([[0.10, -0.10, 0.0]], dtype=torch.float32)))

    def test_03_alternating_signs_halve_down_to_the_floor(self):
        rule = rule_for(ALTERNATING, dtype=torch.float64)
        outs, logs = run(rule, [], 6)
        self.assertEqual([o[0] for o in outs], [0.10, -0.05, 0.025, -0.0125, 0.01, -0.01])
        for o in outs:
            self.assertEqual(o[1:], [0.0, 0.0])
        self.assertEqual([l['length_before'][0] for l in logs], [0.10, 0.10, 0.05, 0.025, 0.0125, 0.01])
        self.assertEqual([l['length_after'][0] for l in logs], [0.10, 0.05, 0.025, 0.0125, 0.01, 0.01])
        self.assertEqual([l['flip'][0] for l in logs], [False, True, True, True, True, True])
        self.assertEqual([l['prev_sat_sign'][0] for l in logs], [0, 1, -1, 1, -1, 1])
        self.assertEqual(rule.length, [0.01, L0, L0])
        self.assertEqual(rule.prev_sat_sign, [-1, 0, 0])

    def test_03b_same_sign_does_not_shorten(self):
        rule = rule_for(x_cmds([1, -1, -1, 1]), dtype=torch.float64)
        outs, logs = run(rule, [], 4)
        self.assertEqual([o[0] for o in outs], [0.10, -0.05, -0.05, 0.025])
        self.assertEqual([l['flip'][0] for l in logs], [False, True, False, True])
        self.assertEqual(logs[2]['length_before'], logs[2]['length_after'])
        # a long run of one sign never changes the length
        rule = rule_for(x_cmds([-1] * 6), dtype=torch.float64)
        outs, _ = run(rule, [], 6)
        self.assertEqual([o[0] for o in outs], [-0.10] * 6)
        self.assertEqual(rule.length, [L0] * 3)

    def test_03c_length_ignores_size_and_sign_of_executed_component(self):
        want = [0.10, -0.05, 0.025]
        for ex in (0.22, 0.10, 0.03, 0.0, -0.22, -0.03):
            with self.subTest(executed_x=ex):
                rule = rule_for(x_cmds([1, -1, 1]), dtype=torch.float64)
                outs, logs = run(rule, [[ex, 0.004, 0.0], [ex, 0.004, 0.0]], 3)
                self.assertEqual([o[0] for o in outs], want)
                self.assertEqual(logs[2]['length_after'], [0.025, L0, L0])

    def test_04_bisect_off_never_changes_length(self):
        rule = rule_for(ALTERNATING, bisect=False, dtype=torch.float64)
        outs, logs = run(rule, [], 6)
        self.assertEqual([o[0] for o in outs], [0.10, -0.10, 0.10, -0.10, 0.10, -0.10])
        for l in logs:
            self.assertEqual(l['flip'], [False] * 3)
            self.assertEqual(l['length_before'], [L0] * 3)
            self.assertEqual(l['length_after'], [L0] * 3)
            self.assertIs(l['bisect'], False)
        self.assertEqual([l['prev_sat_sign'][0] for l in logs], [0, 1, -1, 1, -1, 1])
        self.assertEqual(rule.length, [L0] * 3)

    def test_05_axes_are_independent(self):
        # x flips once then keeps its sign; y stays saturated with one sign; z is unsaturated
        cmds = [[SAT, SAT, 0.02], [-SAT, SAT, -0.03], [-SAT, SAT, 0.01]]
        rule = rule_for(cmds, dtype=torch.float64)
        outs, logs = run(rule, [], 3)
        self.assertEqual([o[0] for o in outs], [0.10, -0.05, -0.05])
        self.assertEqual([o[1] for o in outs], [0.10, 0.10, 0.10])
        self.assertEqual([o[2] for o in outs], [0.02, -0.03, 0.01])
        self.assertEqual(logs[1]['flip'], [True, False, False])
        self.assertEqual(rule.length, [0.05, L0, L0])
        # x flips every tick, y flips once at tick 2, z never
        cmds = [[SAT, -SAT, SAT], [-SAT, -SAT, SAT], [SAT, SAT, SAT]]
        rule = rule_for(cmds, dtype=torch.float64)
        outs, logs = run(rule, [], 3)
        self.assertEqual(outs, [[0.10, -0.10, 0.10], [-0.05, -0.10, 0.10], [0.025, 0.05, 0.10]])
        self.assertEqual([l['flip'] for l in logs],
                         [[False] * 3, [True, False, False], [True, True, False]])
        self.assertEqual(rule.length, [0.025, 0.05, L0])
        self.assertEqual(rule.prev_sat_sign, [1, 1, 1])

    def test_06_reset_restores_l0_and_resets_base(self):
        rule = rule_for(x_cmds([1, -1, -1, 1]), dtype=torch.float64)
        run(rule, [], 2)
        self.assertEqual(rule.length, [0.05, L0, L0])
        self.assertEqual(rule.prev_sat_sign, [-1, 0, 0])
        rule.reset(7)
        self.assertEqual(rule.base.seeds, [0, 7])
        self.assertEqual(rule.length, [L0, L0, L0])
        self.assertEqual(rule.prev_sat_sign, [0, 0, 0])
        # after reset: full length again, and p cleared (-sat before, +sat two ticks later would flip)
        ctx = make_ctx(0)
        self.assertEqual(rule.plan(ctx)[0].tolist(), [-0.10, 0.0, 0.0])
        log = ctx.extras['planner_log']['length_rule']
        self.assertEqual(log['prev_sat_sign'], [0, 0, 0])
        self.assertEqual(log['length_before'], [L0] * 3)
        # previous sign was cleared too: a +sat before reset, -sat after it, with moving feedback
        rule = rule_for(x_cmds([1, -1]), dtype=torch.float64)
        rule.plan(make_ctx(0))
        rule.reset(1)
        out = rule.plan(make_ctx(1, [0.10, 0, 0], [0.10, 0, 0]))
        self.assertEqual(out[0, 0].item(), -0.10)
        self.assertEqual(rule.length, [L0] * 3)

    def test_07_no_halving_without_motion_or_feedback(self):
        # (executed of the previous tick, float64) -> is the opposite-sign tick a flip?
        cases = {
            'none': (None, False),
            'zero': ([0.0, 0.0, 0.0], False),
            'still': ([0.0005, 0.0005, 0.0005], False),          # norm 0.87 mm
            'just_below': ([0.000999, 0.0, 0.0], False),
            'at_threshold_on_axis': ([0.001, 0.0, 0.0], True),   # inclusive: sqrt(x*x) == x exactly
            'at_threshold_off_axis': ([0.0, -0.001, 0.0], True), # the axis itself did not move
            'norm_only': ([0.0006, 0.0006, 0.0006], True),       # each component < 1 mm, norm 1.04 mm
            'large': ([0.10, 0.0, 0.0], True),
        }
        for label, (executed, flip) in cases.items():
            with self.subTest(label):
                rule = rule_for(x_cmds([1, -1]), dtype=torch.float64)
                rule.plan(make_ctx(0))
                ctx = make_ctx(1, [0.10, 0, 0], executed, dtype=torch.float64)
                out = rule.plan(ctx)
                log = ctx.extras['planner_log']['length_rule']
                self.assertEqual(log['flip'], [flip, False, False])
                self.assertEqual(out[0].tolist(), [-0.05 if flip else -0.10, 0.0, 0.0])
                self.assertEqual(rule.length, [0.05 if flip else L0, L0, L0])
                self.assertEqual(log['executed_prev'], executed)
        # a refused flip is not made up later: p is now -1, the next -sat tick is no flip either
        rule = rule_for(x_cmds([1, -1, -1]), dtype=torch.float64)
        outs, logs = run(rule, [[0.0, 0.0, 0.0]], 3)
        self.assertEqual([o[0] for o in outs], [0.10, -0.10, -0.10])
        self.assertEqual(rule.length, [L0] * 3)

    def test_08_unsaturated_tick_breaks_the_flip_and_keeps_the_length(self):
        # +, unsaturated, -: no flip
        rule = rule_for(x_cmds([1, 0, -1]), dtype=torch.float64)
        outs, logs = run(rule, [], 3)
        self.assertEqual([o[0] for o in outs], [0.10, 0.03, -0.10])
        self.assertEqual([l['prev_sat_sign'][0] for l in logs], [0, 1, 0])
        self.assertEqual(logs[2]['flip'], [False] * 3)
        self.assertEqual(rule.length, [L0] * 3)
        # +, - (flip, 0.05), unsaturated, + (no flip, still 0.05), - (flip, 0.025)
        rule = rule_for(x_cmds([1, -1, 0, 1, -1]), dtype=torch.float64)
        outs, logs = run(rule, [], 5)
        self.assertEqual([o[0] for o in outs], [0.10, -0.05, 0.03, 0.05, -0.025])
        self.assertEqual([l['flip'][0] for l in logs], [False, True, False, False, True])
        self.assertEqual([l['length_after'][0] for l in logs], [0.10, 0.05, 0.05, 0.05, 0.025])
        self.assertEqual([l['prev_sat_sign'][0] for l in logs], [0, 1, -1, 0, 1])

    def test_09_shape_dtype_device_and_other_rows(self):
        rows = [[SAT, -0.17, 0.01], [0.3, -0.2, 0.06], [-0.05, 0.05, 0.0]]
        for dtype in (torch.float32, torch.float64):
            for h in (1, 3):
                with self.subTest(dtype=dtype, h=h):
                    native = torch.tensor(rows[:h], dtype=dtype)
                    rule = rule_for([rows[:h]], dtype=dtype)
                    out = rule.plan(make_ctx(0))
                    self.assertEqual(tuple(out.shape), (h, 3))
                    self.assertEqual(out.dtype, dtype)
                    self.assertEqual(out.device, native.device)
                    self.assertTrue(torch.equal(out[0], torch.tensor([0.10, -0.10, 0.01], dtype=dtype)))
                    self.assertTrue(torch.equal(out[1:], native[1:]))
                    # the base's tensor is not modified in place
                    self.assertTrue(torch.equal(rule.base.commands[0], native))

    def test_10_planner_log_is_json_and_keeps_base_keys(self):
        base_log = {'planner': 'scripted', 'rows': 7, 'trace': [1.0, None]}
        rule = rule_for(ALTERNATING, log=base_log)
        commanded = executed = None
        keys = {'rule', 'base_command', 'saturated', 'prev_sat_sign', 'flip', 'length_before',
                'length_after', 'executed_prev', 'out', 'l0_m', 'min_step_m', 'bisect',
                'shrink', 'flipped_before', 'flipped_after'}
        extras = {}  # one dict reused over ticks, as a harness might
        lengths = [0.10, 0.05, 0.025, 0.0125, 0.01, 0.01]
        for t in range(6):
            ctx = make_ctx(t, commanded, executed, extras)
            out = rule.plan(ctx)
            log = ctx.extras['planner_log']
            for k, v in base_log.items():
                self.assertEqual(log[k], v)
            entry = log['length_rule']
            self.assertEqual(set(entry), keys)
            self.assertEqual(json.loads(json.dumps(log, allow_nan=False)), log)
            for k in ('base_command', 'saturated', 'prev_sat_sign', 'flip', 'length_before',
                      'length_after', 'out'):
                self.assertEqual(len(entry[k]), 3)
            self.assertTrue(all(type(v) is bool for v in entry['saturated'] + entry['flip']))
            self.assertTrue(all(type(v) is int for v in entry['prev_sat_sign']))
            self.assertTrue(all(type(v) is float for v in entry['length_before'] + entry['length_after']))
            self.assertEqual(entry['length_before'], [lengths[max(t - 1, 0)], L0, L0])
            self.assertEqual(entry['length_after'], [lengths[t], L0, L0])
            self.assertEqual(entry['out'], out[0].tolist())
            self.assertEqual(entry['rule'], 'halve_on_flip')
            self.assertEqual(entry['l0_m'], 0.10)
            self.assertEqual(entry['min_step_m'], 0.01)
            self.assertIs(entry['bisect'], True)
            self.assertEqual(entry['executed_prev'], None if t == 0 else executed.tolist())
            commanded = executed = out[0].clone()
        # a base that leaves no dict: a fresh one is started
        for left in (None, 'text', [1]):
            rule = rule_for([[SAT, 0.0, 0.0]])
            ctx = make_ctx(0, extras={} if left is None else {'planner_log': left})
            rule.plan(ctx)
            self.assertEqual(list(ctx.extras['planner_log']), ['length_rule'])
        # non-finite values are logged as None, never NaN
        rule = rule_for([[float('nan'), float('inf'), 0.0]])
        ctx = make_ctx(1, [0.0, 0.0, 0.0], [float('nan'), 0.0, 0.0])
        rule.plan(ctx)
        json.dumps(ctx.extras['planner_log'], allow_nan=False)

    def test_11_registry(self):
        expected = {
            'sat10_cem_b': (CEMRelease, False, 'flip'), 'bisect_cem_b': (CEMRelease, True, 'flip'),
            'halvesat_cem_b': (CEMRelease, True, 'saturated'),
            'sat10_gna_b': (GNCubeA, False, 'flip'), 'bisect_gna_b': (GNCubeA, True, 'flip'),
            'halvesat_gna_b': (GNCubeA, True, 'saturated'),
            'sat10_gnb_b': (GNCubeB, False, 'flip'), 'bisect_gnb_b': (GNCubeB, True, 'flip'),
            'halvesat_gnb_b': (GNCubeB, True, 'saturated'),
        }
        self.assertEqual(list(PLANNERS), list(expected))  # nine names, in this order
        for name, (base_cls, bisect, shrink) in expected.items():
            with self.subTest(name):
                self.assertRegex(name, re.compile(r'^[a-z0-9_]+$'))
                planner = PLANNERS[name]()
                self.assertEqual(planner.name, name)
                self.assertIs(type(planner.base), base_cls)
                self.assertIs(planner.config.bisect, bisect)
                self.assertEqual(planner.config.shrink, shrink)
                self.assertEqual((planner.config.l0_m, planner.config.sat_m, planner.config.sat_tol_m,
                                  planner.config.min_step_m, planner.config.moved_m),
                                 (0.10, 0.05, 1e-5, 0.01, 1e-3))
                self.assertIsNot(PLANNERS[name]().base, planner.base)
                planner.reset(3)
                self.assertEqual(planner.length, [0.10] * 3)
                self.assertEqual(planner.prev_sat_sign, [0] * 3)
                self.assertEqual(planner.flipped, [False] * 3)
        for name in ('sat10_cem_b', 'bisect_cem_b', 'halvesat_cem_b'):
            for seed in (0, 1, 2):
                with self.subTest(name=name, seed=seed):
                    baseline, planner = CEMRelease(), PLANNERS[name]()
                    outer = np.random.get_state()
                    baseline.reset(seed)
                    planner.reset(seed)
                    self.assertEqual(planner.base.rng[0], baseline.rng[0])
                    np.testing.assert_array_equal(planner.base.rng[1], baseline.rng[1])
                    self.assertEqual(planner.base.rng[2:], baseline.rng[2:])
                    after = np.random.get_state()  # the global stream is untouched
                    np.testing.assert_array_equal(after[1], outer[1])
                    self.assertEqual(after[2:], outer[2:])

    def test_12_closed_loop_with_sign_oracle(self):
        class Oracle:
            """Plant r <- r - out per axis. Base: saturated sign of r if |r| >= 5 cm else r itself;
            always_sat: +/-10 cm with the sign of r, whatever |r|."""
            name = 'oracle'

            def __init__(self, always_sat=False):
                self.always_sat = always_sat

            def reset(self, seed):
                pass

            def plan(self, ctx):
                a = []
                for r in ctx.extras['r']:
                    if self.always_sat:
                        a.append(0.10 if r >= 0 else -0.10)
                    else:
                        a.append((0.05 if r > 0 else -0.05) if abs(r) >= 0.05 else r)
                return torch.tensor([a], dtype=torch.float64)

        def loop(bisect, r_cm, always_sat=False, ticks=5):
            rule = LengthRule(Oracle(always_sat), LengthRuleConfig(bisect=bisect))
            rule.reset(0)
            r = [x / 100 for x in r_cm]
            trace, commanded = [], None
            for t in range(ticks):
                # executed = commanded, kept in float64
                ctx = make_ctx(t, commanded, commanded, {'r': list(r)}, dtype=torch.float64)
                commanded = rule.plan(ctx)[0].clone()
                r = [ri - ci for ri, ci in zip(r, commanded.tolist())]
                trace.append([abs(ri) * 100 for ri in r])  # |r| in cm after each tick
            return trace

        def assert_trace(got, want):
            self.assertEqual(len(got), len(want))
            for g_row, w_row in zip(got, want):
                for g_v, w_v in zip(g_row, w_row):
                    self.assertAlmostEqual(g_v, w_v, places=9, msg=got)

        # near start: x and y are unsaturated and land at once; z overshoots by L0 once, then lands
        assert_trace(loop(True, [3.0, -2.8, 5.8]),
                     [[0, 0, 4.2], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        # the reference arm does the same here: no flip between two saturated ticks is involved
        assert_trace(loop(False, [3.0, -2.8, 5.8]),
                     [[0, 0, 4.2], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
        # far start: every axis is within 1e-9 cm of the goal after 5 ticks; the first two ticks are
        # full 10 cm steps (x at |r| = 5 cm sits on the saturation threshold, so only the end is fixed)
        far = loop(True, [25.0, -18.0, 9.0])
        assert_trace(far[:2], [[15, 8, 1], [5, 2, 0]])
        assert_trace(far[4:], [[0, 0, 0]])
        # always-saturated worst case on x from 3 cm: lengths 10, 5, 5, 2.5, 2.5 cm
        worst = loop(True, [3.0, 0.0, 0.0], always_sat=True)
        for got, want in zip([row[0] for row in worst], [7.0, 2.0, 3.0, 0.5, 2.0]):
            self.assertAlmostEqual(got, want, places=9, msg=worst)
        # the same plant under the reference arm cycles 7, 3, 7, 3, 7
        cycling = loop(False, [3.0, 0.0, 0.0], always_sat=True)
        for got, want in zip([row[0] for row in cycling], [7.0, 3.0, 7.0, 3.0, 7.0]):
            self.assertAlmostEqual(got, want, places=9, msg=cycling)


class ShrinkScheduleTests(unittest.TestCase):
    """shrink='saturated' against the default shrink='flip'; x axis unless stated, float64."""

    def lengths(self, signs, shrink, executed=(), bisect=True):
        rule = rule_for(x_cmds(signs), bisect=bisect, shrink=shrink, dtype=torch.float64)
        outs, logs = run(rule, list(executed), len(signs))
        for o, s, l in zip(outs, signs, logs):
            if s:  # the executed sign is always the base sign
                self.assertEqual(o[0], s * l['length_after'][0] if bisect else s * L0)
            self.assertEqual(o[1:], [0.0, 0.0])
        return [abs(o[0]) for o in outs], logs, rule

    def test_13_example_sequences_under_both_schedules(self):
        cases = [
            ([1, -1, -1, -1, -1, -1], 'saturated', [0.10, 0.05, 0.025, 0.0125, 0.01, 0.01]),
            ([1, -1, -1, -1, -1, -1], 'flip', [0.10, 0.05, 0.05, 0.05, 0.05, 0.05]),
            ([1, 1, -1, -1, 1], 'saturated', [0.10, 0.10, 0.05, 0.025, 0.0125]),
            ([1, 1, -1, -1, 1], 'flip', [0.10, 0.10, 0.05, 0.05, 0.025]),
        ]
        for signs, shrink, want in cases:
            with self.subTest(signs=signs, shrink=shrink):
                got, logs, rule = self.lengths(signs, shrink)
                self.assertEqual(got, want)
                self.assertEqual([l['length_after'][0] for l in logs], want)
                self.assertEqual(rule.length, [want[-1], L0, L0])
        # the flip itself is the same event under both schedules, and it halves once
        for shrink in ('flip', 'saturated'):
            _, logs, _ = self.lengths([1, 1, -1, -1, 1], shrink)
            self.assertEqual([l['flip'][0] for l in logs], [False, False, True, False, True])
            self.assertEqual([l['flipped_before'][0] for l in logs], [False, False, False, True, True])
            self.assertEqual([l['flipped_after'][0] for l in logs], [False, False, True, True, True])
            self.assertEqual((logs[2]['length_before'][0], logs[2]['length_after'][0]), (0.10, 0.05))
        # before any flip a saturated run of one sign keeps L0 under 'saturated' too
        got, _, rule = self.lengths([-1] * 6, 'saturated')
        self.assertEqual(got, [0.10] * 6)
        self.assertEqual(rule.flipped, [False] * 3)

    def test_14_saturated_tick_without_movement_does_not_halve(self):
        still = [0.0005, 0.0005, 0.0005]  # norm 0.87 mm < moved_m
        # tick 1 flips (0.05); tick 1 did not move, so tick 2 keeps 0.05; tick 2 moved, tick 3 halves
        got, logs, rule = self.lengths([1, -1, -1, -1], 'saturated', [None, still, None])
        self.assertEqual(got, [0.10, 0.05, 0.05, 0.025])
        self.assertEqual(logs[2]['flipped_before'], [True, False, False])
        self.assertEqual(logs[2]['length_before'], logs[2]['length_after'])
        # same with an opposite-sign tick after the still tick: neither a flip nor a halving
        got, logs, _ = self.lengths([1, -1, 1, 1], 'saturated', [None, still, None])
        self.assertEqual(got, [0.10, 0.05, 0.05, 0.025])
        self.assertEqual([l['flip'][0] for l in logs], [False, True, False, False])
        # no feedback at all on a saturated tick after a flip: no halving either
        rule = rule_for(x_cmds([1, -1, -1, -1]), shrink='saturated', dtype=torch.float64)
        run(rule, [], 2)
        self.assertEqual(rule.plan(make_ctx(2))[0, 0].item(), -0.05)
        self.assertEqual(rule.length, [0.05, L0, L0])
        out = rule.plan(make_ctx(3, [-0.05, 0, 0], [-0.05, 0, 0], dtype=torch.float64))
        self.assertEqual(out[0, 0].item(), -0.025)
        # a flip refused for lack of movement does not set f: nothing halves afterwards
        got, _, rule = self.lengths([1, -1, -1, -1], 'saturated', [still])
        self.assertEqual(got, [0.10] * 4)
        self.assertEqual(rule.flipped, [False] * 3)

    def test_15_unsaturated_tick_after_a_flip_keeps_the_length(self):
        for last in (-1, 1):  # the sign of the saturated tick after the gap does not matter
            with self.subTest(last=last):
                got, logs, rule = self.lengths([1, -1, 0, last], 'saturated')
                self.assertEqual(got, [0.10, 0.05, 0.03, 0.025])
                self.assertEqual([l['length_after'][0] for l in logs], [0.10, 0.05, 0.05, 0.025])
                self.assertEqual(logs[3]['flip'], [False] * 3)  # p was 0: not a flip
                self.assertEqual(logs[3]['prev_sat_sign'], [0, 0, 0])
                # the default schedule keeps 0.05 on that tick
                got, _, _ = self.lengths([1, -1, 0, last], 'flip')
                self.assertEqual(got, [0.10, 0.05, 0.03, 0.05])

    def test_16_axes_are_independent(self):
        # x flips at tick 1; y is saturated with one sign throughout; z flips at tick 3
        cmds = [[SAT, SAT, SAT], [-SAT, SAT, SAT], [-SAT, SAT, SAT], [-SAT, SAT, -SAT]]
        rule = rule_for(cmds, shrink='saturated', dtype=torch.float64)
        outs, logs = run(rule, [], 4)
        self.assertEqual(outs, [[0.10, 0.10, 0.10], [-0.05, 0.10, 0.10],
                                [-0.025, 0.10, 0.10], [-0.0125, 0.10, -0.05]])
        self.assertEqual([l['flipped_after'] for l in logs],
                         [[False] * 3, [True, False, False], [True, False, False], [True, False, True]])
        self.assertEqual(rule.length, [0.0125, L0, 0.05])
        self.assertEqual(rule.flipped, [True, False, True])

    def test_17_reset_clears_flipped(self):
        rule = rule_for(x_cmds([1, -1, -1, -1, -1]), shrink='saturated', dtype=torch.float64)
        run(rule, [], 2)
        self.assertEqual((rule.length, rule.flipped), ([0.05, L0, L0], [True, False, False]))
        rule.reset(5)
        self.assertEqual((rule.length, rule.flipped), ([L0] * 3, [False] * 3))
        self.assertEqual(rule.prev_sat_sign, [0, 0, 0])
        # saturated ticks after movement, same sign as before the reset: full length, no halving
        commanded = [-0.05, 0.0, 0.0]
        for t in range(3):
            ctx = make_ctx(t, commanded, commanded, dtype=torch.float64)
            self.assertEqual(rule.plan(ctx)[0].tolist(), [-0.10, 0.0, 0.0])
            self.assertEqual(ctx.extras['planner_log']['length_rule']['flipped_before'], [False] * 3)
        self.assertEqual((rule.length, rule.flipped), ([L0] * 3, [False] * 3))

    def test_18_bisect_off_never_halves(self):
        for signs in ([1, -1, 1, -1, 1, -1], [1, -1, -1, -1, -1, -1]):
            got, logs, rule = self.lengths(signs, 'saturated', bisect=False)
            self.assertEqual(got, [0.10] * 6)
            for l in logs:
                self.assertEqual(l['flip'], [False] * 3)
                self.assertEqual(l['flipped_after'], [False] * 3)
                self.assertEqual(l['length_after'], [L0] * 3)
            self.assertEqual((rule.length, rule.flipped), ([L0] * 3, [False] * 3))

    def test_19_invalid_shrink_is_rejected(self):
        for bad in ('', 'Flip', 'sat', 'always', None, True):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    LengthRule(Scripted([]), LengthRuleConfig(shrink=bad))
        self.assertEqual(LengthRuleConfig().shrink, 'flip')
        self.assertEqual(LengthRule(Scripted([])).config.shrink, 'flip')

    def test_20_log_keys(self):
        names = {'flip': 'halve_on_flip', 'saturated': 'halve_on_saturated_after_flip'}
        for shrink, name in names.items():
            for bisect in (True, False):
                with self.subTest(shrink=shrink, bisect=bisect):
                    _, logs, _ = self.lengths([1, -1, -1], shrink, bisect=bisect)
                    for t, entry in enumerate(logs):
                        self.assertEqual(entry['rule'], name)
                        self.assertEqual(entry['shrink'], shrink)
                        self.assertIs(entry['bisect'], bisect)
                        for k in ('flipped_before', 'flipped_after'):
                            self.assertEqual(len(entry[k]), 3)
                            self.assertTrue(all(type(v) is bool for v in entry[k]))
                        self.assertEqual(entry['flipped_before'], [bisect and t >= 2, False, False])
                        self.assertEqual(entry['flipped_after'], [bisect and t >= 1, False, False])
                        self.assertEqual(json.loads(json.dumps(entry, allow_nan=False)), entry)
        # the logged lists are snapshots, not the live state
        rule = rule_for(x_cmds([1, -1]), shrink='saturated', dtype=torch.float64)
        _, logs = run(rule, [], 2)
        self.assertEqual(logs[0]['flipped_after'], [False] * 3)
        self.assertIsNot(logs[1]['flipped_after'], rule.flipped)

    def test_21_closed_loop_always_saturated_oracle(self):
        class Oracle:
            """Base command +/-10 cm with the sign of r, whatever |r| (always saturated, always right)."""
            name = 'oracle'

            def reset(self, seed):
                pass

            def plan(self, ctx):
                return torch.tensor([[0.10 if r >= 0 else -0.10 for r in ctx.extras['r']]],
                                    dtype=torch.float64)

        def final_abs_r(shrink, r_cm, ticks=5):
            rule = LengthRule(Oracle(), LengthRuleConfig(shrink=shrink))
            rule.reset(0)
            r, commanded = [x / 100 for x in r_cm], None
            for t in range(ticks):  # plant r <- r - out; executed = commanded, in float64
                ctx = make_ctx(t, commanded, commanded, {'r': list(r)}, dtype=torch.float64)
                commanded = rule.plan(ctx)[0].clone()
                r = [ri - ci for ri, ci in zip(r, commanded.tolist())]
            return [abs(ri) for ri in r]

        # executed lengths in cm, x from 3.0: 'flip' 10, 5, 5, 2.5, 2.5; 'saturated' 10, 5, 2.5, 1.25, 1
        want_cm = {'flip': (2.0, 2.2, 0.8), 'saturated': (0.25, 0.05, 0.55)}
        for shrink, want in want_cm.items():
            got = final_abs_r(shrink, [3.0, -2.8, 5.8])
            for axis in range(3):
                with self.subTest(shrink=shrink, axis=axis):
                    self.assertLess(abs(got[axis] - want[axis] / 100), 1e-9, msg=got)


if __name__ == '__main__':
    unittest.main()
