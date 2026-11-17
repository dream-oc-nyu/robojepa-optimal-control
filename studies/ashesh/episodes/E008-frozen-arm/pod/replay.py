"""E008: replay commands, without a model or planner, around the release hold.

Per controller call measure m=min_j min(q_j-lo_j, hi_j-q_j), the limiting
joint/side, ||eef-home||, ||log(R_home R_now^-1)||, workspace scale s in
||eef-home+s R_wb diag(output_max) u|| <= 0.40, and input/output u.
Per blocking tick measure the base-frame proprio displacement, world target
projection, and logged-trajectory discrepancies. Probes are sequential, not
six resets to the final state. Instrumentation never rewrites the action.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import numpy as np

WORKSPACE_R = .40
JOINT_MARGIN = .15
STEP_MAX_POS_NORM = .4
OSC_OUTPUT_MAX = np.array([.05, .05, .05, .5, .5, .5, 1.])


def workspace_scale(incoming, offset, rotation):
    """Exact step-0 branch order in robot_env.py:2027-2056 (before norm cap)."""
    a = np.asarray(offset, dtype=np.float64)
    d = np.asarray(rotation) @ (np.asarray(incoming, dtype=np.float64) * OSC_OUTPUT_MAX[:3])
    a_norm = float(np.linalg.norm(a))
    a_dot_d = float(np.dot(a, d))
    d_norm_sq = float(np.dot(d, d))
    if d_norm_sq < 1e-18:
        return 1.0
    if a_norm >= WORKSPACE_R - 1e-9:
        return 0.0 if a_dot_d > 0 else 1.0
    disc = a_dot_d * a_dot_d - d_norm_sq * (a_norm * a_norm - WORKSPACE_R * WORKSPACE_R)
    if disc < 0:
        return 1.0
    t_pos = (-a_dot_d + np.sqrt(disc)) / d_norm_sq
    return float(max(0., min(1., t_pos)))


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


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Same filesystem for atomic rename; no partially published receipts.
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(json_safe(value), stream, allow_nan=False, separators=(',', ':'))
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def joint_state(inner):
    qpos, lower, upper = [np.asarray(x, dtype=np.float64).copy()
                          for x in inner._get_arm_joint_pos_and_limits()]
    if any(x.shape != (7,) or not np.isfinite(x).all() for x in (qpos, lower, upper)):
        raise ValueError('Expected seven finite Panda arm joints and limits')
    low_margin, high_margin = qpos - lower, upper - qpos
    margins = np.minimum(low_margin, high_margin)
    joint = int(np.argmin(margins))
    return dict(qpos=qpos, lower=lower, upper=upper, joint_margins_rad=margins,
                min_margin_rad=float(margins[joint]), limiting_joint=joint,
                limiting_side='lower' if low_margin[joint] <= high_margin[joint] else 'upper')


def rotation_error(inner):
    # Lazy: CPU fakes can inject this scalar measurement without SciPy installed.
    from scipy.spatial.transform import Rotation
    home = inner._reach_home_eef_quat
    now = inner._get_eef_quat_world(q_ref=home)
    return float(np.linalg.norm((Rotation.from_quat(home) * Rotation.from_quat(now).inv()).as_rotvec()))


class Recorder:
    """Observe the instance method; return the original result by identity."""
    def __init__(self, env, rotation_error=rotation_error):
        self.inner = inner = env._env
        if not inner._reach_task or inner._reach_home_eef_quat is None or inner._reach_home_eef_pos is None:
            raise RuntimeError('Reach hold/home pose must be active after reset')
        self.orig = inner._reach_apply_orientation_hold
        self.rotation_error = rotation_error
        self.contact_unavailable_reason = None
        self.columns = {}

        def wrapper(sub):
            incoming = np.asarray(sub[5:8]).copy()
            row = joint_state(inner)
            eef = np.asarray(inner._get_eef_pos()).copy()
            offset = eef - np.asarray(inner._reach_home_eef_pos)
            # The hold uses this exact helper, rather than assuming a world-aligned base.
            scale = workspace_scale(incoming, offset, inner._get_base_world_R())
            contacts = None
            if self.contact_unavailable_reason is None:
                try:
                    contacts = int(inner._env.sim.data.ncon)
                except Exception as exc:
                    self.contact_unavailable_reason = f'{type(exc).__name__}: {exc}'
                    print(json.dumps({'contact_unavailable_reason': self.contact_unavailable_reason}), flush=True)
            row.update(incoming=incoming, eef_world=eef, home_offset=offset,
                       distance_from_home_m=float(np.linalg.norm(offset)),
                       rotation_error_rad=self.rotation_error(inner), workspace_scale=scale,
                       contact_count=contacts, guard_fired=row['min_margin_rad'] < JOINT_MARGIN)
            result = self.orig(sub)
            outgoing = np.asarray(result[5:8]).copy()
            row.update(outgoing=outgoing,
                       outgoing_rotation_norm=float(np.linalg.norm(result[8:11])),
                       pos_zeroed=bool(np.linalg.norm(outgoing) == 0 and np.linalg.norm(incoming) > 0))
            for key, value in row.items():
                self.columns.setdefault(key, []).append(json_safe(value))
            return result

        self.wrapper = wrapper
        inner._reach_apply_orientation_hold = wrapper
        if env._env._reach_apply_orientation_hold is not wrapper:
            raise RuntimeError('Instance instrumentation did not install on env._env')

    def begin_tick(self):
        self.columns = {}


def record_tick(env, recorder, obs, command, tick, kind, logged=None):
    inner = env._env
    a = np.asarray(command, dtype=np.float32)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError('Command must be a finite three-vector')
    before = np.asarray(inner._get_eef_pos()).copy()
    _, rotation = inner._get_base_world_pose()
    home = np.asarray(inner._reach_home_eef_pos).copy()
    raw_target = before + rotation @ a.astype(np.float64)
    offset = raw_target - home
    length = float(np.linalg.norm(offset))
    active = length > WORKSPACE_R
    projected = home + offset * (WORKSPACE_R / length) if active else raw_target.copy()
    recorder.begin_tick()
    new_obs, reward, done, info = env.step(dict(eef_translation=a, eef_rotation=np.zeros(3),
                                               gripper=np.array([-1.])))
    steps = int(info['sim_steps'])
    if len(recorder.columns.get('guard_fired', [])) != steps:
        raise RuntimeError('Hold interception count differs from controller sim_steps')
    executed = np.asarray(new_obs['proprio_robocasa'][7:10] - obs['proprio_robocasa'][7:10], dtype=np.float32)
    after = np.asarray(inner._get_eef_pos()).copy()
    final = joint_state(inner)
    row = dict(tick=tick, kind=kind, commanded=a, executed=executed,
               executed_norm_m=float(np.linalg.norm(executed.astype(float))),
               distance_m=float(info['reach_distance_to_goal']), sim_steps=steps,
               blocking_converged=bool(info['blocking_converged']),
               blocking_residual_m=info['blocking_residual_m'], reward=reward, done=done,
               eef_before_world=before, eef_after_world=after, base_rotation_world=rotation,
               home_eef_world=home, raw_target_world=raw_target, projected_target_world=projected,
               target_projection_active=active,
               distance_from_home_before_m=float(np.linalg.norm(before-home)),
               distance_from_home_after_m=float(np.linalg.norm(after-home)), final_joint_state=final,
               substeps=recorder.columns, logged_executed=None, logged_distance_m=None,
               executed_abs_error_m=None, executed_error_norm_m=None, distance_abs_error_m=None)
    if logged is not None:
        error = executed.astype(float) - np.asarray(logged['executed'], dtype=float)
        row.update(logged_executed=logged['executed'], logged_distance_m=logged['distance_m'],
                   executed_abs_error_m=np.abs(error), executed_error_norm_m=float(np.linalg.norm(error)),
                   distance_abs_error_m=abs(row['distance_m'] - logged['distance_m']))
    return new_obs, json_safe(row)


def replay_instance(build_env, unit, instance, seed, rotation_error=rotation_error):
    env = build_env()
    try:
        goal_obs, _ = env.reset(mode='goal', seed=0, reach_task_seed=instance, reach_episode_idx=0)
        obs, _ = env.reset(mode='init', seed=0, reach_task_seed=instance, reach_episode_idx=0)
        del goal_obs
        initial = float(obs['reach_distance_to_goal'])
        recorder = Recorder(env, rotation_error=rotation_error)
        result = dict(instance=instance, seed=seed, complete=False, initial_distance_m=initial,
                      logged_initial_distance_m=unit['initial_distance_m'],
                      initial_distance_difference_m=initial-unit['initial_distance_m'], ticks=[], probes=[])
        commands = [('logged', i, t['commanded'], t) for i, t in enumerate(unit['ticks'])]
        for axis in range(3):
            for sign in (1, -1):
                a = np.zeros(3, dtype=np.float32)
                a[axis] = sign * .05
                commands.append(('probe', axis * 2 + (sign < 0), a, None))
        for kind, tick, command, logged in commands:
            obs, row = record_tick(env, recorder, obs, command, tick, kind, logged)
            result['ticks' if kind == 'logged' else 'probes'].append(row)
            print(json.dumps(dict(instance=instance, seed=seed, kind=kind, tick=tick,
                executed_norm_m=row['executed_norm_m'], steps=row['sim_steps'],
                converged=row['blocking_converged'],
                guard_fired_substeps=sum(row['substeps'].get('guard_fired', [])),
                min_margin_end_rad=row['final_joint_state']['min_margin_rad'],
                limiting_joint=row['final_joint_state']['limiting_joint'],
                limiting_side=row['final_joint_state']['limiting_side']), allow_nan=False), flush=True)
        result['contact_unavailable_reason'] = recorder.contact_unavailable_reason
        result['complete'] = True
        return result
    finally:
        env.close()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planner', default='cem_release')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--instances', default=','.join(map(str, range(100, 108))))
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if not re.fullmatch('[a-z0-9_]+', args.planner):
        parser.error('Invalid planner name')
    instances = sorted(set(map(int, args.instances.split(','))))
    if not instances or not set(instances) <= set(range(100, 108)) or args.seed not in (0, 1, 2):
        parser.error('Only registered instances 100..107 and seeds 0..2')
    root = Path(os.environ['SESSION_ROOT']).resolve()
    out = (args.out or root / 'e008/out').resolve()
    if not out.is_relative_to(root / 'e008'):
        parser.error('--out must remain under $SESSION_ROOT/e008')
    # This fixed import pulls torch transitively; this script never constructs a model,
    # encodes an image, calls a planner, or uses torch itself.
    from bench.harness import build_env
    from robots.robocasa import robot_env
    from robots.robocasa import blocking_robot_env
    import inspect
    constants = dict(workspace_radius_m=WORKSPACE_R, joint_margin_rad=JOINT_MARGIN,
                     step_max_pos_norm=STEP_MAX_POS_NORM, osc_output_max=OSC_OUTPUT_MAX.tolist())
    if (robot_env._REACH_WORKSPACE_R != WORKSPACE_R or
        robot_env._REACH_JOINT_LIMIT_MARGIN_RAD != JOINT_MARGIN or
        robot_env._REACH_STEP_MAX_POS_NORM != STEP_MAX_POS_NORM or
        not np.array_equal(robot_env._GOAL_OSC_OUTPUT_MAX, OSC_OUTPUT_MAX)):
        raise RuntimeError('Pinned release constants do not match')
    code_hashes = {name: sha256(path) for name, path in {
        'replay.py': __file__, 'harness.py': inspect.getfile(build_env),
        'robot_env.py': robot_env.__file__, 'blocking_robot_env.py': blocking_robot_env.__file__}.items()}
    receipts, hashes, source_commits = [], {}, None
    for instance in instances:
        key = f'inst{instance}-seed{args.seed}'
        source = root / 'e002/runs' / args.planner / key / 'unit.json'
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        unit = json.loads(raw)
        if not unit.get('complete') or len(unit['ticks']) != 5 or unit['instance'] != instance or unit['seed'] != args.seed:
            raise ValueError(f'Invalid input unit: {source}')
        if unit['planner'] != args.planner or [t['tick'] for t in unit['ticks']] != list(range(5)):
            raise ValueError(f'Input identity/tick mismatch: {source}')
        if unit['source_commits']['robo_jepa'] != 'f93106f5f4a451fd526d51778aa8ab7c852eb824':
            raise ValueError('Input release commit mismatch')
        if source_commits is not None and unit['source_commits'] != source_commits:
            raise ValueError('Mixed source commits in input units')
        source_commits = unit['source_commits']
        hashes[key] = digest
        target = out / args.planner / (key + '.json')
        receipt = json.loads(target.read_text()) if target.exists() else None
        if receipt and receipt.get('complete'):
            if (receipt.get('input_unit_sha256') != digest or receipt.get('code_sha256') != code_hashes or
                receipt.get('instance') != instance or receipt.get('seed') != args.seed or
                receipt.get('planner') != args.planner or len(receipt.get('ticks', [])) != 5 or
                len(receipt.get('probes', [])) != 6):
                raise ValueError(f'Stale/incompatible complete receipt, refusing reuse: {target}')
            print(f'SKIP {target}', flush=True)
        else:
            receipt = replay_instance(build_env, unit, instance, args.seed)
            receipt.update(planner=args.planner, input_unit_sha256=digest, input_unit_path=str(source),
                           source_commits=source_commits, constants=constants, code_sha256=code_hashes)
            atomic_json(target, receipt)
        receipts.append(receipt)
    atomic_json(out / args.planner / 'replay.json', dict(complete=True, planner=args.planner,
        seed=args.seed, dataset_kind='simulator-replay', source_commits=source_commits,
        constants=constants, code_sha256=code_hashes, input_sha256=hashes, instances=receipts))


if __name__ == '__main__':
    main()
