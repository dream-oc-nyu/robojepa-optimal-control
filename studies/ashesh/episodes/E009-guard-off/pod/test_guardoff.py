"""CPU-only fakes: tests are code-path checks, never experiment evidence."""
import sys
sys.dont_write_bytecode = True

import importlib.util
import unittest
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_module(threshold=.15):
    module = ModuleType('fake_robot_env')
    module._REACH_JOINT_LIMIT_MARGIN_RAD = threshold
    exec('def hold(self, action):\n'
         '    self.seen_threshold = _REACH_JOINT_LIMIT_MARGIN_RAD\n'
         '    return action\n', module.__dict__)
    return module


class GuardTests(unittest.TestCase):
    def test_patch_requires_release_value_and_restores_after_failure(self):
        self.assertTrue((HERE / 'guardoff.py').exists(), 'driver implementation missing')
        driver = load('guardoff', HERE / 'guardoff.py')
        module = fake_module()
        with self.assertRaisesRegex(RuntimeError, 'simulator failed'):
            with driver.guard_disabled(module):
                self.assertEqual(module._REACH_JOINT_LIMIT_MARGIN_RAD, 0.)
                raise RuntimeError('simulator failed')
        self.assertEqual(module._REACH_JOINT_LIMIT_MARGIN_RAD, .15)
        for bad in (0., .150001, None):
            module._REACH_JOINT_LIMIT_MARGIN_RAD = bad
            with self.assertRaisesRegex(RuntimeError, '0.15'):
                with driver.guard_disabled(module):
                    self.fail('must not enter with an unexpected release constant')
            self.assertEqual(module._REACH_JOINT_LIMIT_MARGIN_RAD, bad)


    def test_bound_function_must_load_the_same_patched_global(self):
        from types import MethodType, SimpleNamespace
        driver = load('guardoff', HERE / 'guardoff.py')
        module, other = fake_module(), fake_module()
        inner = SimpleNamespace()
        method = MethodType(module.hold, inner)
        with self.assertRaisesRegex(RuntimeError, 'patched module global'):
            driver.verify_threshold(module, method)
        with driver.guard_disabled(module):
            driver.verify_threshold(module, method)
            method(None)
            self.assertEqual(inner.seen_threshold, 0.)
            with self.assertRaisesRegex(RuntimeError, 'patched module global'):
                driver.verify_threshold(module, MethodType(other.hold, inner))
            with self.assertRaisesRegex(RuntimeError, 'patched module global'):
                driver.verify_threshold(module, lambda action: action)


class RecorderTests(unittest.TestCase):
    def test_identity_no_extra_mutation_and_original_threshold(self):
        import numpy as np
        from types import SimpleNamespace
        driver = load('guardoff', HERE / 'guardoff.py')
        self.assertTrue(hasattr(driver, 'Recorder'), 'recorder implementation missing')
        for mutates in (False, True):
            inner = SimpleNamespace(_reach_task=True, _reach_home_eef_quat=[0, 0, 0, 1])
            margin = [.15]
            inner._get_arm_joint_pos_and_limits = lambda: (
                np.array([margin[0]] + [1.] * 6), np.zeros(7), np.ones(7) * 3)
            seen, returned = [], []

            def hold(action):
                seen.append(action)
                if mutates:
                    action[8:11] = .2  # Only the real hold, never the recorder, rewrites.
                result = action if mutates else action.copy()
                returned.append(result)
                return result

            inner._reach_apply_orientation_hold = hold
            recorder = driver.Recorder(SimpleNamespace(_env=inner))
            for value in (.15, .149, -.01):
                margin[0] = value
                action = np.arange(12, dtype=float)
                before = action.copy()
                result = inner._reach_apply_orientation_hold(action)
                self.assertIs(seen[-1], action)
                self.assertIs(result, returned[-1])
                if mutates:
                    before[8:11] = .2
                np.testing.assert_array_equal(action, before)
            self.assertEqual(recorder.columns['would_have_fired'], [False, True, True])
            self.assertEqual(recorder.columns['min_margin_rad'], [.15, .149, -.01])
            self.assertEqual(recorder.columns['limiting_joint'], [0, 0, 0])
            self.assertEqual(recorder.columns['limiting_side'], ['lower'] * 3)
            self.assertEqual(recorder.columns['incoming'], [[5., 6., 7.]] * 3)
            self.assertEqual(recorder.columns['outgoing'], [[5., 6., 7.]] * 3)
            recorder.begin_tick()
            self.assertTrue(all(not v for v in recorder.columns.values()))


class PersistenceTests(unittest.TestCase):
    def test_atomic_publish_and_scoped_paths(self):
        import json
        import os
        import tempfile
        from unittest.mock import patch
        driver = load('guardoff', HERE / 'guardoff.py')
        self.assertTrue(hasattr(driver, 'atomic_json'), 'persistence implementation missing')
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temp:
            root = Path(temp)
            target = driver.output_path(root, 100)
            self.assertEqual(target, root / 'e009/runs/cem_release_guardoff/inst100-seed0/unit.json')
            driver.atomic_json(root, target, {'complete': False, 'ticks': []})
            self.assertFalse(driver.complete_unit(target, 100))
            with patch.object(driver.os, 'replace', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    driver.atomic_json(root, target, {'complete': True})
            self.assertEqual(json.loads(target.read_text()), {'complete': False, 'ticks': []})
            self.assertEqual(list(target.parent.iterdir()), [target])
            for bad in (root / 'e002/runs/unit.json', root / 'e009/../escape.json'):
                with self.assertRaisesRegex(ValueError, 'e009'):
                    driver.atomic_json(root, bad, {})
            (root / 'outside').mkdir()
            (root / 'e009/link').symlink_to(root / 'outside', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'e009'):
                driver.atomic_json(root, root / 'e009/link/unit.json', {})
            with self.assertRaises(ValueError):
                driver.output_path(root, '../100')
            driver.atomic_json(root, target, {'value': float('nan')})
            self.assertIsNone(json.loads(target.read_text())['value'])


class HarnessTests(unittest.TestCase):
    def test_fixed_harness_flow_with_fake_compute_and_simulator(self):
        import ast
        import contextlib
        import io
        import json
        import math
        import os
        import tempfile
        import time
        from types import SimpleNamespace
        import numpy as np
        driver = load('guardoff', HERE / 'guardoff.py')
        self.assertTrue(hasattr(driver, 'run_instance'), 'harness integration missing')
        # Execute the ACTUAL fixed run_unit body with CPU stand-ins for tensors,
        # encoding and rollout. This catches changes to its writes/print/env API.
        candidates = [HERE.parents[2] / 'bench/harness.py']
        if os.environ.get('SESSION_ROOT'):
            candidates.append(Path(os.environ['SESSION_ROOT']) / 'e002/code/bench/harness.py')
        source = next((p for p in candidates if p.is_file()), None)
        self.assertIsNotNone(source, 'fixed harness source must be available')
        tree = ast.parse(source.read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('run_unit', 'atomic_json')]

        class Tensor(np.ndarray):
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return np.asarray(self)
        def tensor(value, **kwargs):
            return np.asarray(value, dtype=np.float32).view(Tensor)
        class Rollout:
            predictor_calls = predicted_samples = 0
            def __init__(self, *args, **kwargs): pass
            def prepare_cache(self): pass
            def __call__(self, value): return tensor([[1.]])
        module = fake_module()
        events = []
        class Env:
            def __init__(self):
                events.append(('build', module._REACH_JOINT_LIMIT_MARGIN_RAD))
                self._env = SimpleNamespace(_reach_task=True, _reach_home_eef_quat=None)
                self._env._get_arm_joint_pos_and_limits = lambda: (
                    np.array([.1] + [1.] * 6), np.zeros(7), np.ones(7) * 3)
                from types import MethodType
                self._env._reach_apply_orientation_hold = MethodType(module.hold, self._env)
                self.position = 0.
                self.closed = False
            def obs(self):
                p = np.zeros(14, dtype=np.float32)
                p[7] = self.position
                return dict(proprio_robocasa=p, reach_distance_to_goal=.1-self.position)
            def reset(self, **kw):
                events.append(('reset', kw))
                self._env._reach_home_eef_quat = [0, 0, 0, 1]
                return self.obs(), {}
            def step(self, action):
                sub = np.zeros(12)
                sub[5:8] = action['eef_translation']
                self._env._reach_apply_orientation_hold(sub)
                self.position += .01
                return self.obs(), 0., False, dict(sim_steps=1, blocking_converged=True,
                                                  success=False, reach_distance_to_goal=.1-self.position)
            def close(self):
                self.closed = True
        envs = []
        def build_env():
            env = Env()
            envs.append(env)
            return env
        harness = ModuleType('fake_compute_real_harness')
        harness.__dict__.update(json=json, math=math, os=os, Path=Path, time=time, np=np,
            torch=SimpleNamespace(as_tensor=tensor, float32=np.float32,
                no_grad=contextlib.nullcontext, zeros_like=lambda a: tensor(np.zeros_like(a)),
                cuda=SimpleNamespace(synchronize=lambda: None)),
            build_env=build_env, encode=lambda *a: tensor([1.]),
            make_state=lambda *a: None, goal_cost=lambda g: lambda z: float(z.item()) if z.ndim == 1 else z,
            bounded_plan=lambda p, ctx: p, Rollout=Rollout,
            PlanContext=lambda *a, **kw: None, Feedback=lambda *a: SimpleNamespace(rho=.5))
        exec(compile(tree, str(source), 'exec'), harness.__dict__)
        class Planner:
            name = 'cem_release'
            def reset(self, seed): events.append(('planner_seed', seed))
            def plan(self, ctx): return tensor([[.01, 0, 0]])
        original_writer = harness.atomic_json
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temp:
            root = Path(temp)
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                unit = driver.run_instance(harness, module, None, Planner(), root, 100, 2., {})
            target = driver.output_path(root, 100)
            published = json.loads(target.read_text())
            self.assertEqual(published, unit)
            self.assertTrue(driver.complete_unit(target, 100))
            self.assertEqual(len(unit['ticks']), 5)
            self.assertEqual(unit['planner'], 'cem_release')
            self.assertEqual(unit['planner_label'], 'cem_release_guardoff')
            self.assertIs(unit['reach_v0_unit'], False)
            self.assertEqual(unit['guard_threshold_rad'], 0.)
            self.assertEqual(unit['guard_threshold_original_rad'], .15)
            for tick in unit['ticks']:
                self.assertEqual(tick['substeps']['min_margin_rad'], [.1])
                self.assertEqual(tick['substeps']['would_have_fired'], [True])
                self.assertEqual(tick['controller_sim_steps'], 1)
                self.assertEqual(tick['rho'], .5)
                self.assertIn('planning_s', tick)
            self.assertTrue(envs[0].closed)
            self.assertEqual(events[0], ('build', 0.))
            self.assertEqual([e[1]['mode'] for e in events if e[0] == 'reset'], ['goal', 'init'])
            self.assertEqual(module._REACH_JOINT_LIMIT_MARGIN_RAD, .15)
            self.assertIs(harness.build_env, build_env)
            self.assertIs(harness.atomic_json, original_writer)
            self.assertNotIn('print', harness.__dict__)
            lines = [json.loads(s) for s in log.getvalue().splitlines() if s.startswith('{')]
            self.assertEqual(len(lines), 5)
            self.assertEqual(lines[-1]['would_have_fired_count'], 1)
            self.assertIn('COMPLETE ' + str(target), log.getvalue())
            with contextlib.redirect_stdout(io.StringIO()):
                driver.run_instance(harness, module, None, Planner(), root, 100, 2., {})
            self.assertEqual(len(envs), 1, 'complete unit must skip before environment build')
            published['ticks'].pop()
            target.write_text(json.dumps(published))
            self.assertFalse(driver.complete_unit(target, 100))
            with self.assertRaisesRegex(ValueError, 'Incomplete|incompatible'):
                driver.run_instance(harness, module, None, Planner(), root, 100, 2., {})


class ExportTests(unittest.TestCase):
    def test_verified_export_retries_preserves_old_file_and_resumes(self):
        import hashlib
        import os
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        if not (HERE / 'export.py').exists() and HERE.name == 'code':
            self.skipTest('exporter intentionally not staged on pod')
        self.assertTrue((HERE / 'export.py').exists(), 'exporter implementation missing')
        exporter = load('export_e009', HERE / 'export.py')
        payload = b'{"complete":true}\n'
        row = dict(path='inst100-seed0/unit.json', size=len(payload),
                   sha256=hashlib.sha256(payload).hexdigest())
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temp:
            out = Path(temp)
            target = out / row['path']
            target.parent.mkdir()
            target.write_bytes(b'old')
            attempts = []
            def transfer(argv):
                self.assertEqual(target.read_bytes(), b'old')
                attempts.append(argv)
                Path(argv[-1]).write_bytes(b'corrupt' if len(attempts) == 1 else payload)
                return SimpleNamespace(returncode=0)
            with patch.object(exporter.subprocess, 'run', side_effect=transfer), \
                    patch.object(exporter.time, 'sleep'):
                self.assertTrue(exporter.copy_verified('fake', '/fake', out, row))
                self.assertFalse(exporter.copy_verified('fake', '/fake', out, row))
            self.assertEqual(len(attempts), 2)
            self.assertEqual(target.read_bytes(), payload)
            target.write_bytes(b'old')
            with patch.object(exporter.subprocess, 'run', return_value=SimpleNamespace(returncode=1)), \
                    patch.object(exporter.time, 'sleep'):
                with self.assertRaisesRegex(RuntimeError, 'three attempts'):
                    exporter.copy_verified('fake', '/fake', out, row)
            self.assertEqual(target.read_bytes(), b'old')
            self.assertFalse(target.with_suffix('.json.part').exists())
        for changes in ({'path': '../escape.json'}, {'path': 'inst101-seed0/unit.json'},
                        {'size': 40_000_001}, {'sha256': 'bad'}):
            with self.assertRaises(ValueError):
                exporter.validate_entry(dict(row, **changes))


class AnalysisTests(unittest.TestCase):
    def test_cli_predictions_positive_negative_boundaries_and_missing(self):
        import copy
        import json
        import os
        import subprocess
        import tempfile
        import numpy as np
        analysis = HERE.parent / 'run.py'
        if not analysis.exists() and HERE.name == 'code':
            self.skipTest('analysis intentionally not staged on pod')
        self.assertTrue(analysis.exists(), 'analysis implementation missing')
        repo = HERE.parents[2]
        with tempfile.TemporaryDirectory(prefix='e009-synthetic-', dir=os.environ['TMPDIR']) as temp:
            root = Path(temp)
            runs, units, out = root / 'runs/cem_release_guardoff', root / 'units', root / 'out'
            command = [sys.executable, '-B', str(analysis), '--runs', str(runs),
                       '--units', str(units), '--out', str(out)]
            absent = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(absent.returncode, 0)
            self.assertIn('Guard-off units absent', absent.stderr)
            self.assertFalse(out.exists())
            guards = {}
            for instance, final in ((100, .03), (107, .06)):
                key = f'inst{instance}-seed0/unit.json'
                for planner in ('cem_release', 'cem_clampexec'):
                    source = repo / 'runs/E002-planner-loop' / planner / key
                    target = units / planner / key
                    target.parent.mkdir(parents=True)
                    target.write_bytes(source.read_bytes())
                unit = json.loads((units / 'cem_release' / key).read_text())
                unit.update(planner_label='cem_release_guardoff', guard_threshold_rad=0.,
                            guard_threshold_original_rad=.15, reach_v0_unit=False,
                            dataset_kind='synthetic-code-path-test')
                for tick in unit['ticks']:
                    tick.update(commanded=[.01, 0, 0], executed=[.001, 0, 0], controller_sim_steps=1,
                                substeps=dict(min_margin_rad=[.149], limiting_joint=[0],
                                    limiting_side=['lower'], incoming=[[.1, 0, 0]],
                                    outgoing=[[.1, 0, 0]], would_have_fired=[True]),
                                final_joint_state=dict(min_margin_rad=.149, limiting_joint=0,
                                                       limiting_side='lower'))
                unit['ticks'][-1]['distance_m'] = final
                guards[instance] = unit
            for negative in (False, True):
                data = copy.deepcopy(guards)
                if negative:
                    data[100]['ticks'][1]['distance_m'] += .016
                    data[100]['ticks'][3]['executed'] = [.0009, 0, 0]
                    for tick in data[107]['ticks']:
                        tick['substeps']['min_margin_rad'] = [.15]
                        tick['substeps']['would_have_fired'] = [False]
                        tick['final_joint_state']['min_margin_rad'] = .15
                    for unit in data.values():
                        unit['ticks'][-1]['distance_m'] = 0.
                for instance, unit in data.items():
                    target = runs / f'inst{instance}-seed0/unit.json'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(json.dumps(unit))
                proc = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                result = json.loads((out / 'result.json').read_text())
                holds = {k: v['holds'] for k, v in result['predictions'].items()}
                self.assertEqual(holds, {str(i): not negative for i in range(1, 5)})
                clauses = result['predictions']['3']['clauses']
                self.assertTrue(clauses['both_closer_than_guarded']['holds'])
                self.assertTrue(clauses['at_least_one_within_5cm']['holds'])
                self.assertEqual(clauses['at_most_one_within_2cm']['holds'], not negative)
                self.assertTrue(clauses['positive_progress_on_at_least_one']['holds'])
                table = np.loadtxt(out / 'trajectory.dat', skiprows=1)
                self.assertEqual(table.shape, (12, 5))
                self.assertEqual(len(result['input_sha256']), 6)
                self.assertGreater((out / 'fig-guardoff.tex').stat().st_size, 0)
                self.assertIn('compiled', result['figure'])
                if result['figure']['compiled']:
                    self.assertGreater((out / 'fig-guardoff.pdf').stat().st_size, 0)
                print(f'SYNTHETIC {"negative" if negative else "positive"}: {holds}; figure={result["figure"]}', flush=True)
            # Independently flip recovery clauses, including strict/inclusive boundaries.
            analyser = load('analysis_e009', analysis)
            natives = {i: json.loads((units / 'cem_release' / f'inst{i}-seed0/unit.json').read_text())
                       for i in (100, 107)}
            caps = {i: json.loads((units / 'cem_clampexec' / f'inst{i}-seed0/unit.json').read_text())
                    for i in (100, 107)}
            boundary = copy.deepcopy(guards)
            for i in (100, 107):
                boundary[i]['ticks'][-1]['distance_m'] = boundary[i]['initial_distance_m']
            p = analyser.evaluate(boundary, natives, caps)['predictions']['3']['clauses']
            self.assertFalse(p['positive_progress_on_at_least_one']['holds'])
            self.assertFalse(p['at_least_one_within_5cm']['holds'])
            boundary[100]['ticks'][-1]['distance_m'] = .02
            boundary[107]['ticks'][-1]['distance_m'] = .05
            p = analyser.evaluate(boundary, natives, caps)['predictions']['3']['clauses']
            self.assertEqual(p['at_most_one_within_2cm']['count'], 1)
            self.assertEqual(p['at_least_one_within_5cm']['count'], 2)
            for i in (100, 107):
                boundary[i]['ticks'][-1]['distance_m'] = natives[i]['ticks'][-1]['distance_m']
            p = analyser.evaluate(boundary, natives, caps)['predictions']
            self.assertFalse(p['3']['clauses']['both_closer_than_guarded']['holds'])
            for i in (100, 107):
                boundary[i]['ticks'][-1]['distance_m'] = caps[i]['ticks'][-1]['distance_m']
            self.assertFalse(analyser.evaluate(boundary, natives, caps)['predictions']['4']['holds'])
            from unittest.mock import patch
            with patch.object(analyser.shutil, 'which', return_value=None):
                result = analyser.evaluate(guards, natives, caps)
                analyser.write_artifacts(root / 'without-latex', result)
            self.assertEqual(result['figure'], {'compiled': False, 'reason': 'pdflatex not installed'})
            # Complete marker alone must not accept partial data or invalid flags.
            broken = copy.deepcopy(guards[100])
            broken['ticks'].pop()
            (runs / 'inst100-seed0/unit.json').write_text(json.dumps(broken))
            partial = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(partial.returncode, 0)
            self.assertIn('Invalid', partial.stderr)
        self.assertFalse(root.exists(), 'temporary synthetic data must be deleted')


class RunnerTests(unittest.TestCase):
    def test_local_shell_job_and_gpu_gates(self):
        import os
        import subprocess
        import tempfile
        import shlex
        # Rewrite only the hard-coded session root in a scratch COPY, never run
        # the deployment script against /workspace or contact a remote machine.
        with tempfile.TemporaryDirectory(prefix='e009-runner-', dir=os.environ['TMPDIR']) as temp:
            root = Path(temp)
            job = root / 'jobs/e009-guardoff'
            job.mkdir(parents=True)
            (root / 'e009/code').mkdir(parents=True)
            fakebin = root / 'bin'
            fakebin.mkdir()
            (root / 'session-env.sh').write_text('export SESSION_ROOT=' + shlex.quote(str(root)) + '\n')
            runner = (HERE / 'run.sh').read_text().replace(
                'ROOT=/workspace/robojepa-E001-20261112T003859Z', 'ROOT=' + shlex.quote(str(root)))
            (job / 'run.sh').write_text(runner)
            scripts = {
                'nvidia-smi': '#!/bin/bash\ncase "$*" in\n*query-gpu*) printf "%s\\n" "${FAKE_GPU_MEMORY:-0}";;\n*query-compute-apps=pid,*) :;;\n*query-compute-apps=pid*) printf "%s" "${FAKE_GPU_APPS:-}";;\nesac\n',
                'python': '#!/bin/bash\nprintf "fake Python started\\n"\nexit "${FAKE_PYTHON_EXIT:-0}"\n',
                'sleep': '#!/bin/bash\n/bin/sleep 0.01\n',
            }
            for name, code in scripts.items():
                path = fakebin / name
                path.write_text(code)
                path.chmod(0o700)
            env = dict(os.environ, PATH=str(fakebin) + ':' + os.environ['PATH'])
            # Prevent inherited shell startup files from recursing through bash fakes.
            env.pop('BASH_ENV', None)
            env.pop('ENV', None)
            def run(**extra):
                return subprocess.run(['bash', str(job / 'run.sh')], env=dict(env, **extra),
                                      capture_output=True, text=True, timeout=10)
            other = root / 'jobs/other'
            other.mkdir()
            (other / 'pid').write_text(str(os.getpid()))
            (other / 'state').write_text('running\n')
            proc = run()
            self.assertEqual(proc.returncode, 3, proc.stdout + proc.stderr)
            self.assertIn('Job already alive', proc.stdout)
            self.assertNotIn('fake Python started', proc.stdout)
            (other / 'state').write_text('complete\n')
            for override in ({'FAKE_GPU_MEMORY': '1'}, {'FAKE_GPU_APPS': '999'}):
                proc = run(**override)
                self.assertEqual(proc.returncode, 4, proc.stdout + proc.stderr)
                self.assertEqual((job / 'state').read_text().strip(), 'failed')
                self.assertNotIn('fake Python started', proc.stdout)
            for code, state in ((0, 'complete'), (7, 'failed')):
                proc = run(FAKE_PYTHON_EXIT=str(code))
                self.assertEqual(proc.returncode, code, proc.stdout + proc.stderr)
                self.assertEqual((job / 'state').read_text().strip(), state)
                self.assertIn('fake Python started', proc.stdout)
                self.assertTrue((job / 'python.pid').read_text().strip().isdigit())


if __name__ == '__main__':
    unittest.main(verbosity=2)
