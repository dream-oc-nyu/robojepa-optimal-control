"""E009: native CEM through the fixed harness, with one runtime constant changed.

Only the environment factory, receipt writer and tick printer are intercepted;
planning, timing, feedback, encoding and controller actions remain in the fixed
harness. No torch/simulator imports until main(). Zero is a threshold, not an
unconditional bypass: a negative joint margin can still trigger the release rule.
"""
import sys
sys.dont_write_bytecode = True

from contextlib import contextmanager
import argparse
import builtins
import dis
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import numpy as np

SUBSTEP_FIELDS = ('min_margin_rad', 'limiting_joint', 'limiting_side',
                  'incoming', 'outgoing', 'would_have_fired')


def joint_state(inner):
    q, lower, upper = [np.asarray(a, dtype=float).copy()
                       for a in inner._get_arm_joint_pos_and_limits()]
    if any(a.shape != (7,) or not np.isfinite(a).all() for a in (q, lower, upper)):
        raise ValueError('Expected seven finite Panda joint positions and limits')
    low, high = q - lower, upper - q
    margins = np.minimum(low, high)
    joint = int(np.argmin(margins))
    return dict(min_margin_rad=float(margins[joint]), limiting_joint=joint,
                limiting_side='lower' if low[joint] <= high[joint] else 'upper')


class Recorder:
    """Observe the release hold with no action copies passed to it or rewrites."""
    def __init__(self, env):
        self.inner = inner = env._env
        if not inner._reach_task or inner._reach_home_eef_quat is None:
            raise RuntimeError('Reach orientation hold must be active after init reset')
        original = inner._reach_apply_orientation_hold
        self.begin_tick()

        def wrapper(action):
            row = joint_state(inner)
            row['incoming'] = np.asarray(action[5:8]).copy().tolist()
            row['would_have_fired'] = row['min_margin_rad'] < ORIGINAL_MARGIN
            result = original(action)
            row['outgoing'] = np.asarray(result[5:8]).copy().tolist()
            for key in SUBSTEP_FIELDS:
                self.columns[key].append(row[key])
            return result

        inner._reach_apply_orientation_hold = wrapper
        if inner._reach_apply_orientation_hold is not wrapper:
            raise RuntimeError('Hold interception failed to install')

    def begin_tick(self):
        self.columns = {key: [] for key in SUBSTEP_FIELDS}

ORIGINAL_MARGIN = .15
THRESHOLD_NAME = '_REACH_JOINT_LIMIT_MARGIN_RAD'


@contextmanager
def guard_disabled(module):
    original = getattr(module, THRESHOLD_NAME, None)
    if original != ORIGINAL_MARGIN:
        raise RuntimeError(f'Expected upstream joint margin 0.15 rad, got {original!r}')
    setattr(module, THRESHOLD_NAME, 0.)
    try:
        yield
    finally:
        setattr(module, THRESHOLD_NAME, original)


def verify_threshold(module, method):
    """Check the bound release function loads this module's patched global."""
    function = getattr(method, '__func__', method)
    reads_global = any(i.opname == 'LOAD_GLOBAL' and i.argval == THRESHOLD_NAME
                       for i in dis.get_instructions(function))
    if (not reads_global or function.__globals__ is not vars(module) or
            getattr(module, THRESHOLD_NAME) != 0.):
        raise RuntimeError('Reach hold does not read the patched module global at 0.0')


def confined(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    # Do not allow e009 itself to redirect into e002 via a symlink.
    if (root / 'e009').resolve() != root / 'e009' or not path.is_relative_to(root / 'e009'):
        raise ValueError('Output must remain under $SESSION_ROOT/e009 (no escaping symlinks)')
    return path


def output_path(root, instance):
    if instance not in (100, 107):
        raise ValueError('Only instances 100 and 107, seed 0, are permitted')
    return confined(root, Path(root) / f'e009/runs/cem_release_guardoff/inst{instance}-seed0/unit.json')


def json_safe(value):
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def atomic_json(root, path, value):
    path = confined(root, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(json_safe(value), stream, allow_nan=False, separators=(',', ':'))
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def complete_unit(path, instance):
    """Never treat an uninstrumented or partial harness receipt as a finished run."""
    try:
        unit = json.loads(Path(path).read_text())
        if (unit.get('complete') is not True or unit['instance'] != instance or unit['seed'] != 0 or
                unit['planner'] != 'cem_release' or unit['planner_label'] != 'cem_release_guardoff' or
                unit['guard_threshold_rad'] != 0. or unit['guard_threshold_original_rad'] != .15 or
                unit['reach_v0_unit'] is not False or len(unit['ticks']) != 5 or
                [t['tick'] for t in unit['ticks']] != list(range(5))):
            return False
        for tick in unit['ticks']:
            n = tick['controller_sim_steps']
            if not isinstance(n, int) or not 0 <= n <= 200:
                return False
            if any(len(tick['substeps'][k]) != n for k in SUBSTEP_FIELDS):
                return False
            for k in ('distance_m', 'commanded', 'executed', 'rho', 'planning_s', 'blocking_converged'):
                if k not in tick:
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


class RecordingEnv:
    """Transparent harness-facing adapter; installs instrumentation after init."""
    def __init__(self, env, module):
        self.env, self.module = env, module
        self.recorder = None
        self.records = []
        verify_threshold(module, env._env._reach_apply_orientation_hold)

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        if kwargs.get('mode') == 'init':
            verify_threshold(self.module, self.env._env._reach_apply_orientation_hold)
            self.recorder = Recorder(self.env)
        return result

    def step(self, action):
        if self.recorder is None or getattr(self.module, THRESHOLD_NAME) != 0.:
            raise RuntimeError('Recorder/guard threshold not ready')
        self.recorder.begin_tick()
        result = self.env.step(action)
        count = result[3]['sim_steps']
        if len(self.recorder.columns['min_margin_rad']) != count:
            raise RuntimeError('Hold interception count differs from controller sim_steps')
        self.records.append(dict(substeps=self.recorder.columns,
                                 final_joint_state=joint_state(self.env._env)))
        return result

    def close(self):
        return self.env.close()


def run_instance(harness, module, wm, planner, root, instance, budget_s, provenance):
    target = output_path(root, instance)
    if target.exists():
        if complete_unit(target, instance):
            print(f'SKIP {target}', flush=True)
            return json.loads(target.read_text())
        raise ValueError(f'Incomplete or incompatible published unit: {target}; inspect before retrying')
    original_build = harness.build_env
    observed = []

    def build():
        env = original_build()
        try:
            wrapped = RecordingEnv(env, module)
        except BaseException:
            env.close()
            raise
        observed.append(wrapped)
        return wrapped

    def enrich(row):
        record = observed[0].records[row['tick']]
        row.update(record)
        return row

    def writer(path, value):
        # Both progress.json and unit.json retain all the harness's fields.
        for row in value.get('ticks', value.get('completed_ticks', [])):
            enrich(row)
        atomic_json(root, path, value)

    def printer(*args, **kwargs):
        # Replace only the harness's tick JSON, not CEM output or other prints.
        if len(args) == 1 and isinstance(args[0], str) and args[0].startswith('{'):
            row = json.loads(args[0])
            if 'tick' in row and 'executed' in row:
                enrich(row)
                row.update(executed_norm_m=float(np.linalg.norm(row['executed'])),
                           steps=row['controller_sim_steps'],
                           min_margin_end_rad=row['final_joint_state']['min_margin_rad'],
                           would_have_fired_count=sum(row['substeps']['would_have_fired']))
                # Full substeps go to the atomic unit, not the heartbeat log.
                del row['substeps']
                return builtins.print(json.dumps(json_safe(row), allow_nan=False), **kwargs)
        return builtins.print(*args, **kwargs)

    metadata = dict(provenance)
    metadata.update(planner_label='cem_release_guardoff', guard_threshold_rad=0.,
                    guard_threshold_original_rad=ORIGINAL_MARGIN, reach_v0_unit=False,
                    guard_global_verified=True)
    with guard_disabled(module), patch.object(harness, 'build_env', build), \
            patch.object(harness, 'atomic_json', writer), patch.object(harness, 'print', printer, create=True):
        unit = harness.run_unit(wm, planner, instance, 0, budget_s, target, metadata)
    if not complete_unit(target, instance):
        raise RuntimeError(f'Incomplete instrumentation in final unit: {target}')
    return unit


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    root = Path(os.environ['SESSION_ROOT']).resolve()
    pending = []
    for instance in (100, 107):
        target = output_path(root, instance)
        if target.exists():
            if not complete_unit(target, instance):
                raise ValueError(f'Incomplete or incompatible published unit: {target}')
            print(f'SKIP {target}', flush=True)
        else:
            pending.append(instance)
    if not pending:
        return
    # All heavyweight imports are lazy; no code is written into staged e002.
    from bench import harness
    from bench.pod.run_unit import load_model, SOURCES, EXPECTED, ROOT
    from planners.cem_release import CEMRelease
    from robots.robocasa import robot_env
    if ROOT.resolve() != root:
        raise RuntimeError('Model loader and driver session roots differ')
    if getattr(robot_env, THRESHOLD_NAME) != ORIGINAL_MARGIN:
        raise RuntimeError('Expected upstream joint margin 0.15 rad before model loading')
    budget = json.loads((root / 'e002/budget.json').read_text())['budget_s']
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError('Expected positive finite calibrated budget_s')
    hashes = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in {
        'guardoff.py': __file__, 'harness.py': inspect.getfile(harness),
        'robot_env.py': robot_env.__file__}.items()}
    provenance = dict(source_commits=SOURCES, weight_sha256=EXPECTED, fp32=True,
                      controller='release ReachBlockingRoboCasaEnv, in-process', code_sha256=hashes)
    start = time.time()
    wm = load_model()
    print(f'MODEL_READY seconds={time.time()-start:.3f}', flush=True)
    for instance in pending:
        run_instance(harness, robot_env, wm, CEMRelease(), root, instance, budget, provenance)


if __name__ == '__main__':
    main()
