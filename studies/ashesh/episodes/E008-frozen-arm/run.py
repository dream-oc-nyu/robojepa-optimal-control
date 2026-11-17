"""CPU-only evaluation of E008's six preregistered predictions (no pooled pass).

Replay fidelity uses ||executed_replay-executed_log||_2 and absolute distance
error. Failures are derived from logged ticks with ||executed|| < .001 m and
||commanded|| >= .01 m. A zeroed-by-guard substep requires BOTH recorded flags.
Synthetic fixtures are code-path tests only and never trigger PDF compilation.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INSTANCES = tuple(range(100, 108))


def read_input(path, hashes):
    raw = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def norm(vector):
    a = np.asarray(vector, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError('Expected finite displacement vector')
    return float(np.linalg.norm(a))


def count_guard(tick):
    return sum(tick['substeps'].get('guard_fired', []))


def evaluate(replay, units):
    receipts = {r['instance']: r for r in replay['instances']}
    if len(replay['instances']) != 8 or set(receipts) != set(INSTANCES):
        raise ValueError('Analysis requires exactly eight distinct instances 100..107')
    unmoved = {i: [j for j, t in enumerate(units[i]['ticks'])
                  if norm(t['executed']) < .001 and norm(t['commanded']) >= .01] for i in INSTANCES}
    failed = [i for i in INSTANCES if unmoved[i]]
    starts, fidelity, table, mechanisms, entries, probes, cuts = [], [], [], [], [], [], []
    workspace = {'logged': [0, 0], 'probe': [0, 0]}
    projections = 0
    for instance in INSTANCES:
        rec, unit = receipts[instance], units[instance]
        if not rec.get('complete') or len(rec['ticks']) != 5 or len(rec['probes']) != 6:
            raise ValueError(f'Incomplete instance {instance}')
        if [t['tick'] for t in rec['ticks']] != list(range(5)) or [t['tick'] for t in rec['probes']] != list(range(6)):
            raise ValueError('Tick/probe order is not canonical')
        start_error = abs(rec['initial_distance_m'] - unit['initial_distance_m'])
        if not np.isfinite(start_error):
            raise ValueError('Nonfinite start distance')
        starts.append(dict(instance=instance, absolute_error_m=start_error, within_1e_minus6_m=start_error <= 1e-6))
        first_guard = None
        for j, (tick, logged) in enumerate(zip(rec['ticks'], unit['ticks'])):
            cols = tick['substeps']
            n = tick['sim_steps']
            if n < 0 or n > 200 or any(len(v) != n for v in cols.values()):
                raise ValueError('Substep column lengths differ from sim_steps or exceed cap')
            for field in ('guard_fired', 'pos_zeroed', 'min_margin_rad', 'limiting_joint', 'limiting_side',
                          'rotation_error_rad', 'workspace_scale'):
                if n and field not in cols:
                    raise ValueError(f'Missing substep field {field}')
            if not np.isfinite(cols.get('rotation_error_rad', [])).all():
                raise ValueError('Nonfinite rotation error')
            # Do not trust precomputed errors or norms: recompute against immutable units.
            if not np.array_equal(np.asarray(tick['commanded'], dtype=np.float32),
                                  np.asarray(logged['commanded'], dtype=np.float32)):
                raise ValueError('Replayed commands differ from logged float32 commands')
            displacement_error = norm(np.asarray(tick['executed']) - np.asarray(logged['executed']))
            distance_error = abs(tick['distance_m'] - logged['distance_m'])
            if not np.isfinite(distance_error):
                raise ValueError('Nonfinite distance')
            fidelity.append(dict(instance=instance, tick=j, executed_error_norm_m=displacement_error,
                                 distance_abs_error_m=distance_error,
                                 within_1mm=displacement_error <= .001 and distance_error <= .001))
            final = tick['final_joint_state']
            if not np.isfinite(final['min_margin_rad']):
                raise ValueError('Nonfinite final joint margin')
            table.append(dict(instance=instance, tick=j, min_margin_end_rad=final['min_margin_rad'],
                              limiting_joint=final['limiting_joint'], limiting_side=final['limiting_side'],
                              executed_norm_m=norm(tick['executed']), guard_fired_substeps=count_guard(tick),
                              substeps=n))
            guards = cols.get('guard_fired', [])
            if first_guard is None and any(guards):
                k = guards.index(True)
                first_guard = dict(tick=j, substep=k, joint=cols['limiting_joint'][k],
                                   side=cols['limiting_side'][k], min_margin_rad=cols['min_margin_rad'][k])
            if j in unmoved[instance]:
                zeroed_guard = sum(bool(g and z) for g, z in zip(guards, cols.get('pos_zeroed', [])))
                max_rotation = max(cols.get('rotation_error_rad', []), default=None)
                mechanisms.append(dict(instance=instance, tick=j, substeps=n,
                    pos_zeroed_substeps=sum(cols.get('pos_zeroed', [])), guard_fired_substeps=sum(guards),
                    zeroed_by_guard_substeps=zeroed_guard, max_rotation_error_rad=max_rotation,
                    holds=zeroed_guard >= 190 and max_rotation is not None and max_rotation < .01))
            command_norm = norm(logged['commanded'])
            ratio = norm(logged['executed']) / command_norm if command_norm else 0.
            if instance not in failed and not logged['blocking_converged'] and .3 <= ratio <= .7:
                following = count_guard(rec['ticks'][j+1]) if j+1 < 5 else None
                margin_error = abs(final['min_margin_rad'] - .15)
                cuts.append(dict(instance=instance, tick=j, logged_executed_commanded_ratio=ratio,
                    guard_fired_substeps=sum(guards), min_margin_end_rad=final['min_margin_rad'],
                    margin_distance_from_threshold_rad=margin_error, following_tick_guard_substeps=following,
                    holds=10 <= sum(guards) <= 190 and margin_error <= .02 and following == 0))
            projections += bool(tick['target_projection_active'])
        if instance in failed:
            moving = [j for j in range(unmoved[instance][0]) if norm(unit['ticks'][j]['executed']) >= .001]
            last_moving = moving[-1] if moving else None
            entries.append(dict(instance=instance, first_guard=first_guard, last_moving_tick=last_moving,
                                in_last_moving_tick=first_guard is not None and first_guard['tick'] == last_moving))
        for tick in rec['probes']:
            value = norm(tick['executed'])
            probes.append(dict(instance=instance, probe=tick['tick'], axis=('+x', '-x', '+y', '-y', '+z', '-z')[tick['tick']],
                               failed_instance=instance in failed, executed_norm_m=value,
                               below_1e_minus4_m=value < 1e-4, at_least_4cm=value >= .04))
        for kind, rows in (('logged', rec['ticks']), ('probe', rec['probes'])):
            for tick in rows:
                scales = np.asarray(tick['substeps'].get('workspace_scale', []), dtype=float)
                if not np.isfinite(scales).all() or np.any((scales < 0) | (scales > 1)):
                    raise ValueError('Invalid recomputed workspace scale')
                workspace[kind][0] += int(np.count_nonzero(scales < 1))
                workspace[kind][1] += len(scales)
    start_matches = sum(r['within_1e_minus6_m'] for r in starts)
    tick_matches = sum(r['within_1mm'] for r in fidelity)
    joint_sides = {(e['first_guard']['joint'], e['first_guard']['side']) for e in entries if e['first_guard']}
    failed_probes = [p for p in probes if p['failed_instance']]
    other_probes = [p for p in probes if not p['failed_instance']]
    n_failed = sum(p['below_1e_minus4_m'] for p in failed_probes)
    n_other = sum(p['at_least_4cm'] for p in other_probes)
    scaled = sum(v[0] for v in workspace.values())
    predictions = {
        '1': dict(holds=start_matches == 8 and tick_matches >= 38, matching_starts=start_matches,
                  total_starts=8, matching_ticks=tick_matches, total_ticks=40, starts=starts, ticks=fidelity),
        '2': dict(holds=len(mechanisms) == 5 and all(r['holds'] for r in mechanisms),
                  unmoved_ticks=len(mechanisms), expected_unmoved_ticks=5, ticks=mechanisms),
        '3': dict(holds=len(entries) == 2 and all(e['in_last_moving_tick'] for e in entries) and len(joint_sides) == 1,
                  failed_instances=failed, entries=entries,
                  common_joint_side=list(next(iter(joint_sides))) if len(joint_sides) == 1 else None),
        '4': dict(holds=len(failed_probes) == 12 and n_failed == 12 and len(other_probes) == 36 and n_other >= 34,
                  failed_probes_below_threshold=n_failed, failed_probes_total=len(failed_probes),
                  other_probes_at_least_4cm=n_other, other_probes_total=len(other_probes), probes=probes),
        '5': dict(holds=len(cuts) == 2 and all(c['holds'] for c in cuts), cut_ticks=len(cuts), ticks=cuts),
        '6': dict(holds=scaled == 0 and projections <= 4, scaled_substeps=scaled,
                  total_substeps=sum(v[1] for v in workspace.values()),
                  workspace_by_kind={k: dict(scaled=v[0], total=v[1]) for k, v in workspace.items()},
                  target_projection_ticks=projections, total_logged_ticks=40),
    }
    return dict(predictions=predictions, tick_table=table, probe_table=probes,
                definitions=dict(failed_instance='any logged unmoved tick',
                                 displacement_error='Euclidean norm of replay minus logged displacement',
                                 joint_index='zero-based, source robot.robot_joints order',
                                 probe_protocol='sequential +x,-x,+y,-y,+z,-z, no resets between probes'))


def write_artifacts(out, result, replay):
    out.mkdir(parents=True, exist_ok=True)
    lines = ['instance tick min_margin_end_rad limiting_joint limiting_side executed_norm_m guard_fired_substeps substeps']
    for r in result['tick_table']:
        lines.append(' '.join(str(r[k]) for k in lines[0].split()))
    (out / 'margin.dat').write_text('\n'.join(lines) + '\n')
    lines = ['instance probe executed_norm_m failed_instance']
    for r in result['probe_table']:
        lines.append(f"{r['instance']} {r['probe']} {r['executed_norm_m']:.12g} {int(r['failed_instance'])}")
    (out / 'probes.dat').write_text('\n'.join(lines) + '\n')
    tex = [r'\documentclass[tikz,border=6pt]{standalone}', r'\usepackage{pgfplots}',
           r'\usepgfplotslibrary{groupplots}', r'\pgfplotsset{compat=1.18}', r'\begin{document}',
           r'\begin{tikzpicture}', r'\begin{groupplot}[group style={group size=1 by 3,vertical sep=2.3cm},',
           r'width=16cm,height=6cm,xlabel={Cumulative controller substep},ylabel={Minimum margin (rad)},',
           r'label style={font=\Large},tick label style={font=\large},title style={font=\Large}]']
    by_id = {r['instance']: r for r in replay['instances']}
    for instance in (100, 107, 103):
        points, boundaries, cumulative = [], [], 0
        for tick in by_id[instance]['ticks']:
            values = tick['substeps'].get('min_margin_rad', [])
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite substep margin')
            points.extend((cumulative+k, v) for k, v in enumerate(values))
            cumulative += len(values)
            boundaries.append(cumulative)
        boundary_text = ','.join(map(str, boundaries[:-1]))
        tex.append(r'\nextgroupplot[title={Instance ' + str(instance) + r'},xmin=0,xmax=' + str(max(1, cumulative)) +
                   r',extra x ticks={' + boundary_text + r'},extra x tick labels={},' +
                   r'extra x tick style={grid=major,major grid style={densely dotted,gray}}]')
        tex.append(r'\addplot[blue,thick,no marks] coordinates {' + ' '.join(f'({x},{y:.10g})' for x, y in points) + '};')
        tex.append(r'\addplot[black,dashed,thick] coordinates {(0,0.15)(' + str(max(1, cumulative)) + ',0.15)};')
    tex.extend([r'\end{groupplot}', r'\end{tikzpicture}', r'\end{document}'])
    (out / 'fig-margin.tex').write_text('\n'.join(tex) + '\n')
    synthetic = replay.get('dataset_kind') == 'synthetic-code-path-test'
    compiler = shutil.which('pdflatex')
    result['figure'] = dict(compiled=False, reason='synthetic code-path test' if synthetic else 'pdflatex not installed')
    if not synthetic and compiler:
        proc = subprocess.run([compiler, '-interaction=nonstopmode', '-halt-on-error', 'fig-margin.tex'],
                              cwd=out, capture_output=True, text=True)
        (out / 'fig-margin-build.log').write_text(proc.stdout + proc.stderr)
        result['figure'] = dict(compiled=proc.returncode == 0 and (out / 'fig-margin.pdf').exists(),
                               returncode=proc.returncode)
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if not synthetic and compiler and not result['figure']['compiled']:
        raise RuntimeError(f'Figure compilation failed; see {out}/fig-margin-build.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--planner', default='cem_release')
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs/E008-frozen-arm')
    parser.add_argument('--units', type=Path, default=ROOT / 'runs/E002-planner-loop')
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'out')
    args = parser.parse_args()
    if not re.fullmatch('[a-z0-9_]+', args.planner):
        parser.error('Invalid planner name')
    hashes = {}
    replay = read_input(args.runs / args.planner / 'replay.json', hashes)
    if not replay.get('complete') or replay['planner'] != args.planner:
        raise ValueError('Incomplete replay or mismatched planner')
    if replay.get('dataset_kind') == 'synthetic-code-path-test':
        scratch = Path(os.environ['TMPDIR']).resolve()
        if not all(p.resolve().is_relative_to(scratch) for p in (args.runs, args.units, args.out)):
            raise ValueError('Synthetic inputs and outputs must stay under TMPDIR')
    seed = replay['seed']
    units = {}
    for instance in INSTANCES:
        key = f'inst{instance}-seed{seed}'
        path = args.units / args.planner / key / 'unit.json'
        units[instance] = unit = read_input(path, hashes)
        if (not unit.get('complete') or unit['instance'] != instance or unit['seed'] != seed or
            unit['planner'] != args.planner or len(unit['ticks']) != 5 or
            [t['tick'] for t in unit['ticks']] != list(range(5))):
            raise ValueError(f'Invalid logged unit: {path}')
        if hashes[str(path)] != replay['input_sha256'][key]:
            raise ValueError(f'Logged unit SHA256 differs from replay provenance: {path}')
    for rec in replay['instances']:
        key = f"inst{rec['instance']}-seed{seed}"
        if rec['seed'] != seed or rec['input_unit_sha256'] != replay['input_sha256'][key]:
            raise ValueError('Instance provenance mismatch')
    result = evaluate(replay, units)
    result.update(planner=args.planner, seed=seed, input_sha256=hashes,
                  dataset_kind=replay.get('dataset_kind', 'simulator-replay'),
                  source_commits=replay.get('source_commits'), constants=replay.get('constants'))
    write_artifacts(args.out, result, replay)
    print(f"E008 {args.planner}: {result['dataset_kind']} (six separate predictions)")
    for number, p in result['predictions'].items():
        print(f"Prediction {number}: holds={p['holds']}")
    print(f"Trajectory: {result['predictions']['1']['matching_starts']}/8 starts, "
          f"{result['predictions']['1']['matching_ticks']}/40 ticks; "
          f"unmoved ticks={result['predictions']['2']['unmoved_ticks']}")
    print(f'Outputs: {args.out}')


if __name__ == '__main__':
    main()
