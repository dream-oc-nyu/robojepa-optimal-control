"""CPU code-path tests only: fakes and temporary synthetic data, not evidence."""
import sys
sys.dont_write_bytecode = True

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WorkspaceTests(unittest.TestCase):
    def test_workspace_formula_hand_cases(self):
        self.assertTrue((HERE / 'replay.py').exists(), 'replay implementation missing')
        replay = load('replay', HERE / 'replay.py')
        cases = [([0, 0, 0], [1, 0, 0], 1.),
                 ([.39, 0, 0], [1, 0, 0], .2),
                 ([.41, 0, 0], [-1, 0, 0], 1.),
                 ([.41, 0, 0], [1, 0, 0], 0.),
                 ([.41, 0, 0], [0, 0, 0], 1.)]
        for offset, incoming, expected in cases:
            with self.subTest(offset=offset, incoming=incoming):
                self.assertAlmostEqual(replay.workspace_scale(incoming, offset, np.eye(3)), expected)


class FakeInner:
    def __init__(self, margin=.1):
        from types import SimpleNamespace
        self.margin = margin
        self._reach_task = True
        self._reach_home_eef_pos = np.zeros(3)
        self._reach_home_eef_quat = np.array([0., 0., 0., 1.])
        self.position = np.array([.02, 0., 0.])
        self._env = SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(ncon=3)))
        self.last_return = None

    def _get_arm_joint_pos_and_limits(self):
        return np.array([self.margin] + [1.] * 6), np.zeros(7), np.ones(7) * 3

    def _get_eef_pos(self):
        return self.position.copy()

    def _get_base_world_pose(self):
        return np.zeros(3), np.eye(3)

    def _get_base_world_R(self):
        return np.eye(3)

    def _get_eef_quat_world(self, q_ref=None):
        return self._reach_home_eef_quat.copy()

    def _reach_apply_orientation_hold(self, sub):
        if self.margin < .15:
            sub[5:8] = 0
        sub[8:11] = [.02, 0, 0]
        self.last_return = sub
        return sub


class FakeBlocking:
    def __init__(self, inner):
        self._env = inner

    def step_sub(self, sub):
        # Actual release call site: dynamic lookup on inner, not on the wrapper.
        return self._env._reach_apply_orientation_hold(sub)


class RecordingTests(unittest.TestCase):
    def test_instance_interception_is_noninterfering(self):
        replay = load('replay', HERE / 'replay.py')
        self.assertTrue(hasattr(replay, 'Recorder'), 'Recorder implementation missing')
        for margin, incoming in [( .1, [1., 2., 3.]), (.2, [1., 2., 3.]), (.1, [0., 0., 0.])]:
            with self.subTest(margin=margin, incoming=incoming):
                inner = FakeInner(margin)
                env = FakeBlocking(inner)
                recorder = replay.Recorder(env, rotation_error=lambda _: .003)
                sub = np.arange(12, dtype=float)
                sub[5:8] = incoming
                expected = sub.copy()
                FakeInner(margin)._reach_apply_orientation_hold(expected)
                returned = env.step_sub(sub)
                self.assertIs(returned, sub)
                self.assertIs(returned, inner.last_return)
                np.testing.assert_array_equal(sub, expected)
                cols = recorder.columns
                self.assertEqual(cols['guard_fired'], [margin < .15])
                self.assertEqual(cols['pos_zeroed'], [margin < .15 and np.linalg.norm(incoming) > 0])
                self.assertEqual(cols['incoming'], [incoming])
                self.assertEqual(cols['limiting_joint'], [0])
                self.assertEqual(cols['limiting_side'], ['lower'])
                self.assertEqual(cols['contact_count'], [3])
                self.assertEqual(cols['rotation_error_rad'], [.003])
                self.assertEqual(len(cols['joint_margins_rad'][0]), 7)

    def test_missing_contacts_explained_once(self):
        replay = load('replay', HERE / 'replay.py')
        self.assertTrue(hasattr(replay, 'Recorder'), 'Recorder implementation missing')
        inner = FakeInner()
        del inner._env.sim
        recorder = replay.Recorder(FakeBlocking(inner), rotation_error=lambda _: 0.)
        for _ in range(2):
            inner._reach_apply_orientation_hold(np.ones(12))
        self.assertEqual(recorder.columns['contact_count'], [None, None])
        self.assertIn('AttributeError', recorder.contact_unavailable_reason)

    def test_atomic_publish_sanitizes_nonfinite(self):
        from unittest.mock import patch
        replay = load('replay', HERE / 'replay.py')
        self.assertTrue(hasattr(replay, 'atomic_json'), 'atomic_json implementation missing')
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temp:
            target = Path(temp) / 'instance.json'
            replay.atomic_json(target, {'value': 1})
            with patch.object(replay.os, 'replace', side_effect=OSError('interrupted')):
                with self.assertRaises(OSError):
                    replay.atomic_json(target, {'value': 2})
            self.assertEqual(json.loads(target.read_text()), {'value': 1})
            replay.atomic_json(target, {'residual': float('nan'), 'array': np.array([1., np.inf])})
            self.assertEqual(json.loads(target.read_text()), {'residual': None, 'array': [1., None]})
            self.assertEqual([p.name for p in Path(temp).iterdir()], ['instance.json'])


class ReplayFlowTests(unittest.TestCase):
    def test_five_ticks_sequential_probes_and_finally_close(self):
        import contextlib
        import io
        replay = load('replay', HERE / 'replay.py')
        self.assertTrue(hasattr(replay, 'replay_instance'), 'replay_instance implementation missing')

        class Env(FakeBlocking):
            def __init__(self, fail=False):
                super().__init__(FakeInner(.2))
                self.resets, self.commands, self.closed = [], [], False
                self.fail = fail

            def obs(self):
                p = np.zeros(14, dtype=np.float32)
                p[7:10] = self._env.position
                return {'proprio_robocasa': p, 'reach_distance_to_goal': .1}

            def reset(self, **kwargs):
                self.resets.append(kwargs)
                return self.obs(), {}

            def step(self, action):
                a = action['eef_translation']
                assert a.dtype == np.float32
                self.commands.append(a.copy())
                if self.fail:
                    raise RuntimeError('sim failure')
                sub = np.zeros(12)
                sub[5:8] = a / .05
                self.step_sub(sub)
                self._env.position += a
                return self.obs(), 0., False, {'reach_distance_to_goal': .1, 'sim_steps': 1,
                    'blocking_converged': True, 'blocking_residual_m': float('nan')}

            def close(self):
                self.closed = True

        unit = {'initial_distance_m': .1, 'ticks': [dict(commanded=[.01, 0, 0],
                executed=[.01, 0, 0], distance_m=.1) for _ in range(5)]}
        env = Env()
        with contextlib.redirect_stdout(io.StringIO()):
            result = replay.replay_instance(lambda: env, unit, 100, 0, rotation_error=lambda _: 0.)
        self.assertTrue(env.closed)
        self.assertEqual([r['mode'] for r in env.resets], ['goal', 'init'])
        for reset in env.resets:
            self.assertEqual({k: v for k, v in reset.items() if k != 'mode'},
                             {'seed': 0, 'reach_task_seed': 100, 'reach_episode_idx': 0})
        self.assertEqual(len(result['ticks']), 5)
        self.assertEqual(len(result['probes']), 6)
        np.testing.assert_array_equal(env.commands[5:], np.eye(3, dtype=np.float32).repeat(2, axis=0) *
                                      np.array([.05, -.05] * 3, dtype=np.float32)[:, None])
        self.assertTrue(all(t['sim_steps'] == len(t['substeps']['guard_fired']) for t in result['ticks']))
        self.assertLess(result['ticks'][0]['executed_error_norm_m'], 1e-6)
        for before, after in zip(result['probes'], result['probes'][1:]):
            np.testing.assert_array_equal(before['eef_after_world'], after['eef_before_world'])
        bad = Env(fail=True)
        with self.assertRaisesRegex(RuntimeError, 'sim failure'):
            replay.replay_instance(lambda: bad, unit, 100, 0, rotation_error=lambda _: 0.)
        self.assertTrue(bad.closed)


class AnalysisTests(unittest.TestCase):
    def test_cli_synthetic_positive_and_negative_controls(self):
        import copy
        import hashlib
        import subprocess
        analysis = HERE.parent / 'run.py'
        self.assertTrue(analysis.exists(), 'analysis implementation missing')
        repo = HERE.parents[2]
        with tempfile.TemporaryDirectory(prefix='e008-code-path-', dir=os.environ['TMPDIR']) as temp:
            root = Path(temp)
            units_dir = root / 'units' / 'cem_release'
            instances, hashes = [], {}
            for instance in range(100, 108):
                source = repo / 'runs/E002-planner-loop/cem_release' / f'inst{instance}-seed0/unit.json'
                raw = source.read_bytes()
                target = units_dir / f'inst{instance}-seed0/unit.json'
                target.parent.mkdir(parents=True)
                target.write_bytes(raw)
                digest = hashlib.sha256(raw).hexdigest()
                hashes[f'inst{instance}-seed0'] = digest
                unit = json.loads(raw)
                unmoved = [i for i, t in enumerate(unit['ticks']) if np.linalg.norm(t['executed']) < .001
                           and np.linalg.norm(t['commanded']) >= .01]
                rows = []
                for i, tick in enumerate(unit['ticks']):
                    ratio = np.linalg.norm(tick['executed']) / np.linalg.norm(tick['commanded'])
                    cut = not unmoved and not tick['blocking_converged'] and .3 <= ratio <= .7
                    count = 200 if i in unmoved else 50 if cut else 1 if unmoved and i == unmoved[0]-1 else 0
                    guards = [False] * (200-count) + [True] * count
                    rows.append(dict(tick=i, kind='logged', commanded=tick['commanded'], executed=tick['executed'],
                        distance_m=tick['distance_m'], sim_steps=200, blocking_converged=tick['blocking_converged'],
                        target_projection_active=False,
                        final_joint_state=dict(min_margin_rad=.15 if cut else .1 if count else .3,
                                               limiting_joint=2, limiting_side='lower'),
                        substeps=dict(guard_fired=guards, pos_zeroed=guards.copy(), rotation_error_rad=[.002]*200,
                                      min_margin_rad=[.1 if g else .3 for g in guards], limiting_joint=[2]*200,
                                      limiting_side=['lower']*200, workspace_scale=[1.]*200)))
                probes = [dict(tick=j, kind='probe', executed=[0. if unmoved else .05, 0., 0.],
                               substeps=dict(workspace_scale=[1.])) for j in range(6)]
                instances.append(dict(instance=instance, seed=0, complete=True, input_unit_sha256=digest,
                                      initial_distance_m=unit['initial_distance_m'], ticks=rows, probes=probes))
            replay = dict(complete=True, planner='cem_release', seed=0,
                          dataset_kind='synthetic-code-path-test', instances=instances, input_sha256=hashes)
            for negative in (False, True):
                data = copy.deepcopy(replay)
                if negative:
                    data['instances'][0]['ticks'][3]['substeps']['pos_zeroed'] = [True]*189 + [False]*11
                    data['instances'][0]['probes'][0]['executed'] = [.001, 0., 0.]
                runs = root / ('negative' if negative else 'positive')
                (runs / 'cem_release').mkdir(parents=True)
                (runs / 'cem_release/replay.json').write_text(json.dumps(data))
                out = runs / 'out'
                proc = subprocess.run([sys.executable, '-B', str(analysis), '--runs', str(runs),
                    '--units', str(root / 'units'), '--out', str(out)], capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                result = json.loads((out / 'result.json').read_text())
                holds = {k: v['holds'] for k, v in result['predictions'].items()}
                self.assertEqual(holds, {str(i): not (negative and i in (2, 4)) for i in range(1, 7)})
                self.assertEqual(len(result['tick_table']), 40)
                self.assertEqual(result['predictions']['1']['matching_ticks'], 40)
                self.assertEqual(result['predictions']['4']['failed_probes_below_threshold'], 11 if negative else 12)
                self.assertEqual(len(result['input_sha256']), 9)
                for name in ('margin.dat', 'probes.dat', 'fig-margin.tex'):
                    self.assertGreater((out / name).stat().st_size, 0)
                self.assertFalse((out / 'fig-margin.pdf').exists(), 'synthetic figures must not compile')
                print(f'SYNTHETIC code-path control {"negative" if negative else "positive"}: {holds}', flush=True)


class ExportTests(unittest.TestCase):
    def test_hash_retry_atomic_replace_and_resume_without_network(self):
        import hashlib
        from types import SimpleNamespace
        from unittest.mock import patch
        exporter = load('export_e008', HERE / 'export.py')
        payload = b'{"complete":true}\n'
        row = dict(path='replay.json', size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as temp:
            out = Path(temp)
            target = out / 'replay.json'
            target.write_bytes(b'previous-complete-receipt')
            attempts = []

            def transfer(argv):
                self.assertEqual(target.read_bytes(), b'previous-complete-receipt')
                attempts.append(argv)
                Path(argv[-1]).write_bytes(b'corrupt' if len(attempts) == 1 else payload)
                return SimpleNamespace(returncode=0)

            # No subprocess is started: this tests checksum/publication logic only.
            with patch.object(exporter.subprocess, 'run', side_effect=transfer), patch.object(exporter.time, 'sleep'):
                self.assertTrue(exporter.copy_verified('fake-pod', '/fake-source', out, row))
                self.assertFalse(exporter.copy_verified('fake-pod', '/fake-source', out, row))
            self.assertEqual(len(attempts), 2)
            self.assertEqual(target.read_bytes(), payload)
            self.assertFalse((out / 'replay.json.part').exists())
            target.write_bytes(b'previous-complete-receipt')
            with patch.object(exporter.subprocess, 'run', return_value=SimpleNamespace(returncode=1)) as run, \
                 patch.object(exporter.time, 'sleep'):
                with self.assertRaisesRegex(RuntimeError, 'three attempts'):
                    exporter.copy_verified('fake-pod', '/fake-source', out, row)
                self.assertEqual(run.call_count, 3)
            self.assertEqual(target.read_bytes(), b'previous-complete-receipt')
            self.assertFalse((out / 'replay.json.part').exists())
        for bad in (dict(row, path='../escape.json'), dict(row, size=40_000_001), dict(row, sha256='bad')):
            with self.assertRaises(ValueError):
                exporter.validate_entry(bad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
