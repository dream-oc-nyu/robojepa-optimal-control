"""CPU checks for the planner-log proxy, its sidecar writer and the exporter's sidecar handling.

No GPU, no model, no kubectl. `run_planner.py` imports the harness and the model driver at module
level, so its source is executed here with those four modules replaced by stubs (the stubs are removed
from `sys.modules` again; nothing is registered). The GN planners and `bench.interface` are real.
"""
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch

from bench.interface import PlanContext
from bench.pod import export_e002
from planners.gn_cube_a import Planner as GNCubeA      # imported before the stubbed load below, so the
from planners.gn_cube_b import Planner as GNCubeB      # driver binds these same real classes


class FakeCEM:
    """Stands in for the CEM planners: sets no extras."""
    name = 'cem_fake'

    def reset(self, seed):
        self.seed = seed

    def plan(self, ctx):
        return torch.zeros(1, 3)


def load_driver():
    def module(name, **attributes):
        m = types.ModuleType(name)
        m.__dict__.update(attributes)
        return m

    stubs = {
        'bench.harness': module('bench.harness', run_unit=None),
        'bench.pod.run_unit': module('bench.pod.run_unit', load_model=None, ROOT=Path('unset'),
                                     SOURCES={'stub': 'source'}, EXPECTED={'stub': 'weights'}),
        'planners.cem_clampexec': module('planners.cem_clampexec', CEMClampExec=FakeCEM),
        'planners.cem_cap10': module('planners.cem_cap10', CEMCap10=FakeCEM),
    }
    path = Path(__file__).with_name('run_planner.py')
    driver = types.ModuleType('run_planner_under_test')
    driver.__file__ = str(path)
    with mock.patch.dict(sys.modules, stubs):
        exec(compile(path.read_text(), str(path), 'exec'), driver.__dict__)     # no bytecode file
    return driver


driver = load_driver()
LoggedPlanner, persist, jsonable = driver.LoggedPlanner, driver.persist_planner_log, driver.jsonable


class Toy:
    """z(a) = tanh(W clamp(a / 5 cm, -1, 1) + bias), two equal views, native-like .5/.5 mean |z - g|."""

    def __init__(self, target=(0.02, -0.03, 0.01), features=600):
        generator = torch.Generator().manual_seed(7)
        self.weight = torch.randn(features, 3, generator=generator) * 0.4
        self.bias = torch.linspace(-0.1, 0.1, features)
        self.goal = self.features(torch.tensor([target]))[0]

    def features(self, actions):
        return torch.tanh((actions / 0.05).clamp(-1, 1) @ self.weight.T + self.bias)

    def rollout(self, actions):
        return self.features(actions[:, 0])[:, None, :]

    def cost(self, z):
        return ((z - self.goal).abs().reshape(*z.shape[:-1], 2, -1).mean(-1) * 0.5).sum(-1)

    def context(self, tick=0):
        return PlanContext(self.rollout, self.cost, self.goal, 1, 0.30, 216.0, tick)


class Scripted:
    """Inner planner that puts a prepared object in extras (None = sets nothing)."""
    name = 'scripted'

    def __init__(self, logs):
        self.logs, self.seeds, self.plans = list(logs), [], []

    def reset(self, seed):
        self.seeds.append(seed)
        return 'reset-return'

    def plan(self, ctx):
        log = self.logs[len(self.plans)]
        if log is not None:
            ctx.extras['planner_log'] = log
        self.plans.append(torch.full((1, 3), float(ctx.tick)))
        return self.plans[-1]


def context(tick):
    return PlanContext(None, None, None, 1, 0.30, 216.0, tick)


def forbidden(*args, **kwargs):
    raise AssertionError('I/O, printing or encoding inside plan()')


def captured(function, *args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        result = function(*args)
    return result, out.getvalue().splitlines()


def log_for(tick):
    return dict(stop_reason='step_tolerance', rows_used=7 + tick, linearisations=1, accepted_steps=tick,
                rejected_steps=0, cost_at_zero=0.5, final_cost=0.25,
                iterations=[dict(point=[0.0, 0.0, 0.01], cost=0.25, accepted=True)], config=dict(max_rows=150))


class Workspace(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix='planner-log-test-'))
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.unit = self.dir/'unit.json'
        self.unit.write_text('{"complete": true}\n')
        self.sidecar = self.dir/'planner_log.json'

    def run_ticks(self, inner, ticks=5):
        proxy = LoggedPlanner(inner)
        proxy.reset(0)
        for tick in range(ticks):
            proxy.plan(context(tick))
        return proxy


class ProxyTests(Workspace):
    def test_forwards_name_reset_and_returns_the_inner_plan_object(self):
        inner = Scripted([log_for(0)])
        proxy = LoggedPlanner(inner)
        self.assertEqual(proxy.name, 'scripted')
        self.assertEqual(proxy.reset(2), 'reset-return')
        self.assertEqual(inner.seeds, [2])
        ctx = context(0)
        self.assertIs(proxy.plan(ctx), inner.plans[0])
        self.assertIs(ctx.extras['planner_log'], inner.logs[0])     # extras left as the planner set them

    def test_plan_keeps_references_and_does_no_io(self):
        logs = [log_for(tick) for tick in range(4)] + [None]
        inner = Scripted(logs)
        proxy = LoggedPlanner(inner)
        with mock.patch('builtins.open', forbidden), mock.patch('io.open', forbidden), \
                mock.patch('builtins.print', forbidden), mock.patch('json.dumps', forbidden), \
                mock.patch('json.dump', forbidden), mock.patch('os.replace', forbidden), \
                mock.patch('copy.deepcopy', forbidden), mock.patch('copy.copy', forbidden):
            for tick in range(5):
                proxy.plan(context(tick))
        self.assertEqual([tick for tick, _ in proxy.records], [0, 1, 2, 3, 4])
        for (_, kept), original in zip(proxy.records, logs):
            self.assertIs(kept, original)
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ['unit.json'])
        self.assertEqual(captured(persist, proxy, self.unit, 100, 0)[0], self.sidecar)
        self.assertTrue(self.sidecar.exists())

    def test_inner_exception_propagates_and_records_nothing(self):
        proxy = LoggedPlanner(Scripted([]))
        with self.assertRaises(IndexError):
            proxy.plan(context(0))
        self.assertEqual(proxy.records, [])


class WriterTests(Workspace):
    def test_five_ticks_exact_shape_and_summary_lines(self):
        logs = [log_for(tick) for tick in range(5)]
        logs[3] = None
        before = self.unit.read_bytes()
        proxy = self.run_ticks(Scripted(logs))
        written, lines = captured(persist, proxy, self.unit, 103, 2)
        self.assertEqual(written, self.sidecar)
        data = json.loads(self.sidecar.read_text(), parse_constant=self.fail)
        self.assertEqual(data, dict(planner='scripted', instance=103, seed=2,
                                    ticks=[dict(tick=tick, log=logs[tick]) for tick in range(5)]))
        self.assertEqual(list(data), ['planner', 'instance', 'seed', 'ticks'])
        self.assertIsNone(data['ticks'][3]['log'])
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ['planner_log.json', 'unit.json'])
        self.assertEqual(self.unit.read_bytes(), before)
        self.assertEqual(len(lines), 5)
        for tick, line in enumerate(lines):
            self.assertTrue(line.startswith('PLANNER_LOG {'))
            row = json.loads(line[len('PLANNER_LOG '):], parse_constant=self.fail)
            self.assertEqual((row['instance'], row['seed'], row['tick']), (103, 2, tick))
            if tick == 3:
                self.assertIsNone(row['log'])
                continue
            self.assertNotIn('iterations', row['log'])
            self.assertNotIn('config', row['log'])
            expected = {k: v for k, v in logs[tick].items() if k not in ('iterations', 'config')}
            self.assertEqual(row['log'], expected)
        self.assertIn('iterations', logs[0])                       # the planner's own dict is not edited

    def test_nonfinite_tensor_and_unknown_objects_become_strict_json(self):
        class Opaque:
            def __str__(self):
                return 'opaque-object'

        log = dict(status='nonfinite', cost_zero=float('nan'), cost_final=float('inf'), low=-math.inf,
                   point=torch.tensor([0.01, float('nan'), -0.05]), scalar=torch.tensor(0.5),
                   gradient=torch.ones(2, requires_grad=True), count=torch.tensor(3), flag=True,
                   pair=(1, 2.5), opaque=Opaque(), iterations=[dict(cost=float('nan'), lam=None)])
        log[5] = 'int key'
        try:
            import numpy as np
            log.update(array=np.array([1.5, np.inf]), np_scalar=np.float32(0.25), np_int=np.int64(4))
        except ImportError:
            np = None
        proxy = self.run_ticks(Scripted([log] * 5))
        written, lines = captured(persist, proxy, self.unit, 100, 0)
        self.assertEqual(written, self.sidecar)
        text = self.sidecar.read_text()
        got = json.loads(text, parse_constant=self.fail)['ticks'][0]['log']
        self.assertEqual(got['status'], 'nonfinite')
        self.assertIsNone(got['cost_zero'])
        self.assertIsNone(got['cost_final'])
        self.assertIsNone(got['low'])
        self.assertEqual(got['point'][1], None)
        self.assertAlmostEqual(got['point'][0], 0.01, places=6)
        self.assertEqual((got['scalar'], got['gradient'], got['count'], got['flag']), (0.5, [1.0, 1.0], 3, True))
        self.assertEqual((got['pair'], got['opaque'], got['5']), ([1, 2.5], 'opaque-object', 'int key'))
        self.assertEqual(got['iterations'], [dict(cost=None, lam=None)])
        if np is not None:
            self.assertEqual((got['array'], got['np_scalar'], got['np_int']), ([1.5, None], 0.25, 4))
        self.assertEqual(len(lines), 5)
        for line in lines:
            json.loads(line[len('PLANNER_LOG '):], parse_constant=self.fail)
        self.assertTrue(math.isnan(log['cost_zero']))               # source log untouched

    def test_all_none_logs_write_and_print_nothing(self):
        proxy = self.run_ticks(FakeCEM())
        self.assertEqual(captured(persist, proxy, self.unit, 100, 0), (None, []))
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ['unit.json'])

    def test_wrong_tick_count_leaves_an_existing_sidecar_untouched(self):
        self.sidecar.write_text('earlier run\n')
        stamp = self.sidecar.stat().st_mtime_ns
        for ticks in (0, 4, 6):
            with self.subTest(ticks=ticks):
                proxy = self.run_ticks(Scripted([log_for(0)] * 6), ticks=ticks)
                self.assertEqual(captured(persist, proxy, self.unit, 100, 0), (None, []))
                self.assertEqual(self.sidecar.read_text(), 'earlier run\n')
                self.assertEqual(self.sidecar.stat().st_mtime_ns, stamp)
        proxy = LoggedPlanner(Scripted([log_for(0)] * 5))
        for tick in (0, 1, 2, 4, 3):                                # five records, but not ticks 0..4 in order
            proxy.records.append((tick, log_for(tick)))
        self.assertEqual(captured(persist, proxy, self.unit, 100, 0), (None, []))
        self.assertEqual(self.sidecar.read_text(), 'earlier run\n')

    def test_write_failure_is_reported_and_harms_nothing(self):
        proxy = self.run_ticks(Scripted([log_for(tick) for tick in range(5)]))
        before = self.unit.read_bytes()
        self.sidecar.write_text('earlier run\n')
        with mock.patch.object(driver.os, 'replace', side_effect=OSError('disk full')):
            written, lines = captured(persist, proxy, self.unit, 100, 0)
        self.assertIsNone(written)
        self.assertEqual(lines[0], "PLANNER_LOG_ERROR OSError('disk full')")
        self.assertEqual(len([line for line in lines if line.startswith('PLANNER_LOG {')]), 5)
        self.assertEqual(self.sidecar.read_text(), 'earlier run\n')
        self.assertEqual(sorted(p.name for p in self.dir.iterdir()), ['planner_log.json', 'unit.json'])
        self.assertEqual(self.unit.read_bytes(), before)
        # Missing directory: also only a report.
        written, lines = captured(persist, proxy, self.dir/'absent'/'unit.json', 100, 0)
        self.assertIsNone(written)
        self.assertTrue(lines[0].startswith('PLANNER_LOG_ERROR FileNotFoundError('))
        self.assertFalse((self.dir/'absent').exists())

    def test_unserialisable_log_and_broken_stdout_never_raise(self):
        loop = {}
        loop['self'] = loop
        proxy = self.run_ticks(Scripted([loop] * 5))
        written, lines = captured(persist, proxy, self.unit, 100, 0)
        self.assertIsNone(written)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith('PLANNER_LOG_ERROR RecursionError('))
        self.assertFalse(self.sidecar.exists())
        proxy = self.run_ticks(Scripted([log_for(tick) for tick in range(5)]))
        with mock.patch('builtins.print', side_effect=BrokenPipeError()):
            self.assertEqual(persist(proxy, self.unit, 100, 0), self.sidecar)
        self.assertTrue(self.sidecar.exists())


class RealPlannerTests(Workspace):
    def check(self, planner_class, reason_key):
        toy, inner = Toy(), planner_class()
        proxy = LoggedPlanner(inner)
        self.assertEqual(proxy.name, inner.name)
        proxy.reset(1)
        direct = planner_class()
        direct.reset(1)
        for tick in range(5):
            ctx = toy.context(tick)
            plan = proxy.plan(ctx)
            self.assertIs(proxy.records[-1][1], ctx.extras['planner_log'])
            self.assertTrue(torch.equal(plan, direct.plan(toy.context(tick))))   # the proxy changes no plan
        logs = [log for _, log in proxy.records]
        self.assertEqual(len({id(log) for log in logs}), 5)        # a fresh dict per plan(): references suffice
        self.assertEqual(len({id(log['iterations']) for log in logs}), 5)
        frozen = json.dumps(jsonable(logs[0]), sort_keys=True)
        written, lines = captured(persist, proxy, self.unit, 101, 1)
        self.assertEqual(json.dumps(jsonable(logs[0]), sort_keys=True), frozen)
        data = json.loads(written.read_text(), parse_constant=self.fail)
        self.assertEqual((data['planner'], data['instance'], data['seed']), (inner.name, 101, 1))
        self.assertEqual([row['tick'] for row in data['ticks']], [0, 1, 2, 3, 4])
        for row, line in zip(data['ticks'], lines):
            self.assertIsInstance(row['log'], dict)
            self.assertIsInstance(row['log'][reason_key], str)
            self.assertIsInstance(row['log']['iterations'], list)
            self.assertGreaterEqual(row['log']['linearisations'], 1)
            summary = json.loads(line[len('PLANNER_LOG '):])['log']
            self.assertEqual(summary[reason_key], row['log'][reason_key])
            self.assertFalse({'iterations', 'config'} & set(summary))
            self.assertLess(len(line), 700)
        return data

    def test_gn_cube_a(self):
        data = self.check(GNCubeA, 'stop_reason')
        self.assertLess(data['ticks'][0]['log']['final_cost'], data['ticks'][0]['log']['cost_at_zero'])

    def test_gn_cube_b(self):
        data = self.check(GNCubeB, 'status')
        self.assertLess(data['ticks'][0]['log']['cost_final'], data['ticks'][0]['log']['cost_zero'])
        self.assertIn('config', data['ticks'][0]['log'])           # kept in the sidecar, dropped from the line

    def test_gn_cube_a_nonfinite_abort_is_visible(self):
        toy = Toy()
        proxy = LoggedPlanner(GNCubeA())
        proxy.reset(0)
        for tick in range(5):
            ctx = toy.context(tick)
            ctx.rollout = lambda actions: toy.rollout(actions) * float('nan')
            self.assertFalse(bool(proxy.plan(ctx).any()))            # the zero command
        written, lines = captured(persist, proxy, self.unit, 100, 0)
        row = json.loads(written.read_text(), parse_constant=self.fail)['ticks'][0]['log']
        self.assertEqual((row['stop_reason'], row['cost_at_zero'], row['final_cost']), ('nonfinite', None, None))
        self.assertIn('"stop_reason":"nonfinite"', lines[0])


class DriverMainTests(unittest.TestCase):
    """main() against a stand-in for the fixed harness: paths, skip behaviour, CEM silence."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='planner-log-main-'))
        self.addCleanup(shutil.rmtree, self.root, True)
        (self.root/'e002').mkdir()
        (self.root/'e002/budget.json').write_text('{"budget_s": 216.0}\n')
        self.toy = Toy()
        self.seen = []

    def fake_run_unit(self, wm, planner, instance, seed, budget_s, output, provenance):
        self.seen.append((type(planner).__name__, planner.name, instance, seed, budget_s, str(output)))
        output = Path(output)
        if output.exists():
            print(f'SKIP {output}', flush=True)
            return json.loads(output.read_text())
        output.parent.mkdir(parents=True)
        planner.reset(seed)
        commands = []
        for tick in range(5):
            commands.append(planner.plan(self.toy.context(tick))[0].tolist())
            assert not output.with_name('planner_log.json').exists()
        unit = dict(planner=planner.name, instance=instance, seed=seed, complete=True, commands=commands, **provenance)
        output.write_text(json.dumps(unit, indent=2) + '\n')
        return unit

    def main(self, *argv):
        with mock.patch.object(driver, 'run_unit', self.fake_run_unit), \
                mock.patch.object(driver, 'load_model', lambda: 'model'), \
                mock.patch.object(driver, 'ROOT', self.root), \
                mock.patch.object(sys, 'argv', ['run_planner', *argv]):
            return captured(driver.main)[1]

    LENGTH_RULE_NAMES = [
        'sat10_cem_a', 'bisect_cem_a', 'halvesat_cem_a',
        'sat10_gna_a', 'bisect_gna_a', 'halvesat_gna_a',
        'sat10_gnb_a', 'bisect_gnb_a', 'halvesat_gnb_a',
        'sat10_cem_b', 'bisect_cem_b', 'halvesat_cem_b',
        'sat10_gna_b', 'bisect_gna_b', 'halvesat_gna_b',
        'sat10_gnb_b', 'bisect_gnb_b', 'halvesat_gnb_b',
    ]

    def test_registry_order(self):
        self.assertEqual(list(driver.PLANNERS), [
            'cem_clampexec', 'cem_cap10', 'gn_cube_a', 'gn_cube_b',
        ] + self.LENGTH_RULE_NAMES)
        self.assertIs(driver.PLANNERS['gn_cube_a'], GNCubeA)
        self.assertIs(driver.PLANNERS['gn_cube_b'], GNCubeB)

    def test_length_rule_factories_build_named_planners(self):
        for name in self.LENGTH_RULE_NAMES:
            with self.subTest(planner=name):
                self.assertIn(name, driver.PLANNERS)
                self.assertEqual(driver.PLANNERS[name]().name, name)

    def test_launcher_covers_registry_and_stages_imported_planners(self):
        pod = Path(__file__).parent
        launcher = (pod/'launch_planner.sh').read_text()
        case = re.search(r'case "\$planner" in\n(.*?)\nesac', launcher, re.DOTALL)
        if case is None:
            self.fail('Launcher has no planner case statement')
        names = set()
        for patterns in re.findall(r'^\s*([^\n)]+)\)', case.group(1), re.MULTILINE):
            names.update(pattern.strip() for pattern in patterns.split('|'))
        for name in driver.PLANNERS:
            with self.subTest(planner=name):
                self.assertIn(name, names)
        staged = re.search(r'^for path in (.*?); do$', launcher, re.MULTILINE)
        if staged is None:
            self.fail('Launcher has no staged file list')
        staged_paths = set(shlex.split(staged.group(1)))
        modules = re.findall(r'^from planners\.(\w+) import ',
                             (pod/'run_planner.py').read_text(), re.MULTILINE)
        self.assertTrue(modules)
        for module in modules:
            with self.subTest(module=module):
                self.assertIn(f'planners/{module}.py', staged_paths)

    def test_gn_units_get_sidecars_and_a_rerun_touches_nothing(self):
        lines = self.main('--planner', 'gn_cube_a', '--seeds', '0,2')
        units = sorted((self.root/'e002/runs').rglob('unit.json'))
        self.assertEqual(len(units), 16)
        self.assertEqual([call[1:4] for call in self.seen],
                         [('gn_cube_a', i, s) for s in (0, 2) for i in range(100, 108)])
        self.assertEqual(self.seen[0][5], str(self.root/'e002/runs/gn_cube_a/inst100-seed0/unit.json'))
        self.assertEqual({call[4] for call in self.seen}, {216.0})
        for unit in units:
            sidecar = unit.with_name('planner_log.json')
            data = json.loads(sidecar.read_text())
            content = json.loads(unit.read_text())
            self.assertEqual((data['planner'], data['instance'], data['seed']),
                             ('gn_cube_a', content['instance'], content['seed']))
            self.assertEqual(sorted(p.name for p in unit.parent.iterdir()), ['planner_log.json', 'unit.json'])
        self.assertEqual(sum(line.startswith('PLANNER_LOG {') for line in lines), 80)
        self.assertFalse(any(line.startswith('PLANNER_LOG_ERROR') for line in lines))
        # The proxy leaves the content the harness would publish identical to a bare planner's.
        bare = GNCubeA()
        bare.reset(0)
        commands = [bare.plan(self.toy.context(tick))[0].tolist() for tick in range(5)]
        self.assertEqual(json.loads(units[0].read_text())['commands'], commands)

        first = units[0].with_name('planner_log.json')
        first.write_text('kept from the run that produced the unit\n')
        state = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (self.root/'e002/runs').rglob('*') if p.is_file()}
        lines = self.main('--planner', 'gn_cube_a', '--seeds', '0,2')
        self.assertEqual(sum(line.startswith('SKIP ') for line in lines), 16)
        self.assertFalse(any(line.startswith('PLANNER_LOG') for line in lines))
        self.assertEqual({p: (p.read_bytes(), p.stat().st_mtime_ns)
                          for p in (self.root/'e002/runs').rglob('*') if p.is_file()}, state)

    def test_cem_units_get_no_sidecar_and_no_lines(self):
        lines = self.main('--planner', 'cem_cap10')
        self.assertEqual(len(list((self.root/'e002/runs/cem_fake').rglob('unit.json'))), 8)
        self.assertEqual([p.name for p in (self.root/'e002/runs').rglob('*') if p.is_file() and p.name != 'unit.json'], [])
        self.assertFalse(any('PLANNER_LOG' in line for line in lines))

    def test_seed_validation_is_unchanged(self):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.main('--planner', 'gn_cube_b', '--seeds', '3')
        self.assertEqual(self.seen, [])


def entry(path, data=b'{}\n', size=None):
    return dict(path=path, size=len(data) if size is None else size, sha256=hashlib.sha256(data).hexdigest())


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp(prefix='planner-log-export-'))
        self.addCleanup(shutil.rmtree, self.dir, True)
        self.remote = self.dir/'remote'/'e002'
        self.files = {
            'runs/gn_cube_a/inst100-seed0/unit.json': b'{"unit": "a100"}\n',
            'runs/gn_cube_a/inst100-seed0/planner_log.json': b'{"ticks": [1]}\n',
            'runs/gn_cube_a/inst100-seed0/progress.json': b'{"stage": "executed"}\n',
            'runs/gn_cube_a/inst101-seed0/unit.json': b'{"unit": "a101"}\n',
            'runs/gn_cube_a/inst101-seed0/planner_log.json': b'x' * (export_e002.MAX_BYTES + 1),
            'runs/gn_cube_a/inst102-seed0/planner_log.json': b'{"orphan": true}\n',       # no unit.json beside it
            'runs/gn_cube_a/inst102-seed0/progress.json': b'{"stage": "predicted"}\n',
            'runs/cem_cap10/inst100-seed0/unit.json': b'{"unit": "c100"}\n',
            'budget.json': b'{"budget_s": 216.0}\n',
            'smoke.json': b'{"complete": true}\n',
        }
        for name, data in self.files.items():
            (self.remote/name).parent.mkdir(parents=True, exist_ok=True)
            (self.remote/name).write_bytes(data)

    def remote_manifest(self):
        before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in self.remote.rglob('*')}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(export_e002.manifest_code(str(self.remote)), {})
        self.assertEqual({p: (p.stat().st_size, p.stat().st_mtime_ns) for p in self.remote.rglob('*')}, before)
        return out.getvalue()

    def test_remote_listing_is_read_only_and_pairs_sidecars_with_units(self):
        manifest = json.loads(self.remote_manifest())
        self.assertEqual(sorted(row['path'] for row in manifest), sorted(
            name for name in self.files if 'progress' not in name and 'inst102' not in name))
        for row in manifest:
            data = self.files[row['path']]
            self.assertEqual(row['size'], len(data))
            oversize = len(data) > export_e002.MAX_BYTES
            self.assertEqual(row['sha256'], None if oversize else hashlib.sha256(data).hexdigest())
        code = export_e002.manifest_code(str(self.remote))
        for word in ('write', 'unlink', 'mkdir', 'rename', 'replace', 'open(', 'touch', 'rmtree', 'remove'):
            self.assertNotIn(word, code)

    def test_filter(self):
        unit, log = 'runs/gn_cube_a/inst100-seed0/unit.json', 'runs/gn_cube_a/inst100-seed0/planner_log.json'
        rows = [entry(unit), entry(log), entry('budget.json'), entry('smoke.json')]
        self.assertEqual(export_e002.select_exports(rows), (rows, []))
        big = dict(path=log, size=export_e002.MAX_BYTES + 1, sha256=None)
        self.assertEqual(export_e002.select_exports([entry(unit), big, entry('budget.json')]),
                         ([entry(unit), entry('budget.json')], [big]))
        at_cap = entry(log, size=export_e002.MAX_BYTES)
        self.assertEqual(export_e002.select_exports([entry(unit), at_cap]), ([entry(unit), at_cap], []))
        for bad in ([entry(log)],                                                         # no unit beside it
                    [entry(unit), entry('runs/gn_cube_a/inst101-seed0/planner_log.json')],
                    [entry(unit, size=export_e002.MAX_BYTES + 1)],                         # only sidecars may be skipped
                    [entry('budget.json', size=export_e002.MAX_BYTES + 1)],
                    [entry('/abs/unit.json')], [entry('runs/../../unit.json')],
                    [entry('runs/x/../../../planner_log.json')],
                    [entry(unit), entry('runs/gn_cube_a/inst100-seed0/planner_log.json.tmp')],
                    [entry('runs/gn_cube_a/inst100-seed0/unit.txt')]):
            with self.subTest(bad=bad[-1]['path']), self.assertRaises(ValueError):
                export_e002.select_exports(bad)

    def test_export_end_to_end_with_a_fake_kubectl(self):
        (self.dir/'bench').mkdir()
        (self.dir/'bench/interface.py').write_text('')
        copies, commands = [], []

        def check_output(command, text):
            commands.append(command)
            return 'pod-0' if 'get' in command else self.remote_manifest()

        def run(command):
            commands.append(command)
            source = command[-2].split(':', 1)[1]
            self.assertTrue(source.startswith(export_e002.ROOT + '/e002/'))
            name = source[len(export_e002.ROOT + '/e002/'):]
            copies.append(name)
            shutil.copyfile(self.remote/name, command[-1])
            return types.SimpleNamespace(returncode=0)

        def export():
            old = os.getcwd()
            os.chdir(self.dir)
            try:
                with mock.patch.object(export_e002.subprocess, 'check_output', check_output), \
                        mock.patch.object(export_e002.subprocess, 'run', run), \
                        mock.patch.object(export_e002.time, 'sleep', forbidden):
                    return captured(export_e002.main)[1]
            finally:
                os.chdir(old)

        lines = export()
        out = self.dir/export_e002.OUT
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith('WARNING planner log not exported'))
        self.assertIn('runs/gn_cube_a/inst101-seed0/planner_log.json', lines[0])
        self.assertEqual(json.loads(lines[1]), dict(verified_files=6, copied=6, already_verified=0, complete_units=3,
            planner_logs=1, planner_logs_skipped_oversize=1, output=str(export_e002.OUT)))
        exported = sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file())
        self.assertEqual(exported, ['budget.json', 'cem_cap10/inst100-seed0/unit.json', 'export-manifest.json',
            'gn_cube_a/inst100-seed0/planner_log.json', 'gn_cube_a/inst100-seed0/unit.json',
            'gn_cube_a/inst101-seed0/unit.json', 'smoke.json'])
        self.assertEqual((out/'gn_cube_a/inst100-seed0/planner_log.json').read_bytes(),
                         self.files['runs/gn_cube_a/inst100-seed0/planner_log.json'])
        self.assertEqual(len(list(out.rglob('unit.json'))), 3)
        self.assertEqual(len(json.loads((out/'export-manifest.json').read_text())), 6)
        for command in commands:                                     # get pods, one exec (the listing), cp
            self.assertTrue({'get', 'exec', 'cp'} & set(command))
        self.assertEqual(sum('exec' in command for command in commands), 1)

        del copies[:]
        (out/'gn_cube_a/inst100-seed0/planner_log.json').write_bytes(b'damaged')     # resumable, re-verified
        summary = json.loads(export()[1])
        self.assertEqual(copies, ['runs/gn_cube_a/inst100-seed0/planner_log.json'])
        self.assertEqual((summary['copied'], summary['already_verified'], summary['planner_logs']), (1, 5, 1))
        self.assertEqual((out/'gn_cube_a/inst100-seed0/planner_log.json').read_bytes(),
                         self.files['runs/gn_cube_a/inst100-seed0/planner_log.json'])


if __name__ == '__main__':
    unittest.main()
