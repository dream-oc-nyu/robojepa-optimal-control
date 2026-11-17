"""CPU/numpy analysis of E009's four preregistered predictions, clause by clause.

--runs is the guard-off planner directory; --units contains the two E002
planner directories. Tick -1 is the initial distance; 0..4 are post-step values.
Comparisons to native/cap use their unrounded raw finals, not the rounded
parenthetical values in episode.md. 'Within' means <=; 'closer/farther' is strict.
Synthetic tests must keep all inputs and outputs under TMPDIR.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INSTANCES = (100, 107)
FIELDS = ('min_margin_rad', 'limiting_joint', 'limiting_side',
          'incoming', 'outgoing', 'would_have_fired')


def norm(vector):
    a = np.asarray(vector, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError('Invalid displacement vector: expected three finite numbers')
    return float(np.linalg.norm(a))


def validate(unit, instance, planner, guardoff=False):
    if (unit.get('complete') is not True or unit.get('instance') != instance or unit.get('seed') != 0 or
            unit.get('planner') != planner or len(unit.get('ticks', [])) != 5 or
            [t.get('tick') for t in unit['ticks']] != list(range(5))):
        raise ValueError(f'Invalid/incomplete unit: {planner}, instance {instance}')
    distances = [unit['initial_distance_m']] + [t['distance_m'] for t in unit['ticks']]
    if not np.isfinite(distances).all() or min(distances) < 0 or distances[0] <= 0:
        raise ValueError('Invalid physical distances')
    if guardoff and (unit.get('planner_label') != 'cem_release_guardoff' or
                     unit.get('guard_threshold_rad') != 0. or
                     unit.get('guard_threshold_original_rad') != .15 or
                     unit.get('reach_v0_unit') is not False):
        raise ValueError('Invalid guard-off intervention metadata')
    for tick in unit['ticks']:
        norm(tick['commanded'])
        norm(tick['executed'])
        if not guardoff:
            continue
        cols, n = tick['substeps'], tick['controller_sim_steps']
        if type(n) is not int or not 0 <= n <= 200 or any(len(cols[k]) != n for k in FIELDS):
            raise ValueError('Invalid substep counts (must match controller_sim_steps <= 200)')
        if not np.isfinite(cols['min_margin_rad']).all():
            raise ValueError('Invalid substep margins')
        for j, margin in enumerate(cols['min_margin_rad']):
            if (type(cols['would_have_fired'][j]) is not bool or
                    cols['would_have_fired'][j] != (margin < .15)):
                raise ValueError('Invalid would_have_fired: must use original 0.15 rad threshold')
            if cols['limiting_joint'][j] not in range(7) or cols['limiting_side'][j] not in ('lower', 'upper'):
                raise ValueError('Invalid limiting joint/side')
            norm(cols['incoming'][j])
            norm(cols['outgoing'][j])
        if not np.isfinite(tick['final_joint_state']['min_margin_rad']):
            raise ValueError('Invalid final joint margin')


def prediction(clauses):
    return dict(holds=all(c['holds'] for c in clauses.values()), clauses=clauses)


def evaluate(guardoff, native, cap):
    fidelity, motion, margins, finals, trajectory = [], [], [], [], []
    for instance in INSTANCES:
        go, baseline, capped = guardoff[instance], native[instance], cap[instance]
        for tick in ((0, 1) if instance == 100 else (0,)):
            observed = go['ticks'][tick]['distance_m']
            reference = baseline['ticks'][tick]['distance_m']
            error = abs(observed - reference)
            fidelity.append(dict(instance=instance, tick=tick, guardoff_cm=100*observed,
                                 native_cm=100*reference, abs_error_cm=100*error,
                                 tolerance_cm=1.5, holds=error <= .015))
        all_margins = []
        for tick in go['ticks']:
            command, executed = norm(tick['commanded']), norm(tick['executed'])
            stalled = executed < .001 and command >= .01
            motion.append(dict(instance=instance, tick=tick['tick'], commanded_m=command,
                               executed_m=executed, stalled=stalled, holds=not stalled))
            all_margins.extend(tick['substeps']['min_margin_rad'])
            all_margins.append(tick['final_joint_state']['min_margin_rad'])
        minimum = float(min(all_margins))
        margins.append(dict(instance=instance, minimum_margin_rad=minimum,
                            threshold_rad=.15, holds=minimum < .15))
        initial, final = go['initial_distance_m'], go['ticks'][-1]['distance_m']
        native_final, cap_final = baseline['ticks'][-1]['distance_m'], capped['ticks'][-1]['distance_m']
        progress = 1 - final / initial
        finals.append(dict(instance=instance, initial_cm=100*initial, guardoff_cm=100*final,
                           native_cm=100*native_final, cap5_cm=100*cap_final, progress=progress,
                           closer_than_guarded=final < native_final, within_5cm=final <= .05,
                           within_2cm=final <= .02, positive_progress=progress > 0.,
                           farther_than_cap=final > cap_final))
        for tick in range(-1, 5):
            def distance(unit):
                return unit['initial_distance_m'] if tick == -1 else unit['ticks'][tick]['distance_m']
            trajectory.append(dict(instance=instance, tick=tick, native_cm=100*distance(baseline),
                                   guardoff_cm=100*distance(go), cap5_cm=100*distance(capped)))
    n5 = sum(f['within_5cm'] for f in finals)
    n2 = sum(f['within_2cm'] for f in finals)
    positive = sum(f['positive_progress'] for f in finals)
    native_stalls = {str(i): sum(norm(t['executed']) < .001 and norm(t['commanded']) >= .01
                                for t in native[i]['ticks']) for i in INSTANCES}
    predictions = {
        '1': prediction({'before_entry_within_1_5cm': dict(
            holds=all(r['holds'] for r in fidelity), rows=fidelity)}),
        '2': prediction({
            'no_unmoved_tick_on_command_at_least_1cm': dict(
                holds=all(r['holds'] for r in motion), executed_below_m=.001, command_at_least_m=.01,
                stalled_count=sum(r['stalled'] for r in motion), native_stalled_counts=native_stalls, rows=motion),
            'margin_below_0_15rad_on_both': dict(holds=all(r['holds'] for r in margins), rows=margins)}),
        '3': prediction({
            'both_closer_than_guarded': dict(holds=all(f['closer_than_guarded'] for f in finals),
                rows=[dict(instance=f['instance'], guardoff_cm=f['guardoff_cm'], native_cm=f['native_cm'],
                           holds=f['closer_than_guarded']) for f in finals]),
            'at_least_one_within_5cm': dict(holds=n5 >= 1, count=n5, required_minimum=1, threshold_cm=5.,
                rows=[dict(instance=f['instance'], guardoff_cm=f['guardoff_cm'], within=f['within_5cm']) for f in finals]),
            'at_most_one_within_2cm': dict(holds=n2 <= 1, count=n2, allowed_maximum=1, threshold_cm=2.,
                rows=[dict(instance=f['instance'], guardoff_cm=f['guardoff_cm'], within=f['within_2cm']) for f in finals]),
            'positive_progress_on_at_least_one': dict(holds=positive >= 1, count=positive, required_minimum=1,
                rows=[dict(instance=f['instance'], progress=f['progress'], positive=f['positive_progress']) for f in finals])}),
        '4': prediction({'both_farther_than_cap5': dict(holds=all(f['farther_than_cap'] for f in finals),
            rows=[dict(instance=f['instance'], guardoff_cm=f['guardoff_cm'], cap5_cm=f['cap5_cm'],
                       holds=f['farther_than_cap']) for f in finals])}),
    }
    return dict(predictions=predictions, finals=finals, trajectory=trajectory,
                interpretation_valid=predictions['1']['holds'],
                interpretation_note=('Read predictions 2-4 only after fidelity prediction 1 holds; '
                                     'otherwise investigate wrapper fidelity before interpreting recovery.'),
                definitions=dict(progress='1 - final_distance_m / initial_distance_m',
                    before_entry_ticks={'100': [0, 1], '107': [0]}, within='<= (inclusive)',
                    comparisons='Raw native and cap finals; episode.md parentheses are rounded descriptions',
                    minimum_margin='Minimum over all pre-hold substeps and all post-tick joint states',
                    tick_minus_one='Initial distance before any action',
                    intervention='Threshold exactly 0.0 rad; negative margins can still trigger the release rule'))


def write_artifacts(out, result):
    out.mkdir(parents=True, exist_ok=True)
    keys = ('instance', 'tick', 'native_cm', 'guardoff_cm', 'cap5_cm')
    lines = [' '.join(keys)] + [' '.join(str(row[k]) for k in keys) for row in result['trajectory']]
    (out / 'trajectory.dat').write_text('\n'.join(lines) + '\n')
    tex = [r'\documentclass[tikz,border=8pt]{standalone}', r'\usepackage{pgfplots}',
           r'\usepgfplotslibrary{groupplots}', r'\pgfplotsset{compat=1.18}', r'\begin{document}',
           r'\begin{tikzpicture}', r'\begin{groupplot}[group style={group size=1 by 2,vertical sep=3cm},',
           r'width=16cm,height=6.8cm,xlabel={Planning tick (after execution)},ylabel={Distance to goal (cm)},',
           r'xmin=-1,xmax=4,ymin=0,xtick={-1,0,1,2,3,4},xticklabels={Start,0,1,2,3,4},',
           r'label style={font=\Large},tick label style={font=\large},title style={font=\Large},',
           r'legend style={font=\large,at={(0.5,-0.32)},anchor=north,legend columns=3},',
           r'grid=major,major grid style={gray!20}]']
    for instance in INSTANCES:
        tex.append(r'\nextgroupplot[title={Instance ' + str(instance) + '}]')
        rows = [r for r in result['trajectory'] if r['instance'] == instance]
        for column, style, label in (('native_cm', 'black,dashed,mark=square*', 'Native'),
                                     ('guardoff_cm', 'blue,mark=*', 'Guard off'),
                                     ('cap5_cm', 'orange!80!black,dashdotted,mark=triangle*', '5 cm cap')):
            points = ' '.join(f"({r['tick']},{r[column]:.12g})" for r in rows)
            tex.append(r'\addplot[very thick,' + style + '] coordinates {' + points + '};')
            if instance == 107:
                tex.append(r'\addlegendentry{' + label + '}')
    tex += [r'\end{groupplot}', r'\end{tikzpicture}', r'\end{document}']
    (out / 'fig-guardoff.tex').write_text('\n'.join(tex) + '\n')
    compiler = shutil.which('pdflatex')
    pdf = out / 'fig-guardoff.pdf'
    pdf.unlink(missing_ok=True)  # Never report an earlier PDF as this build's output.
    result['figure'] = dict(compiled=False, reason='pdflatex not installed')
    if compiler:
        proc = subprocess.run([compiler, '-interaction=nonstopmode', '-halt-on-error', 'fig-guardoff.tex'],
                              cwd=out, capture_output=True, text=True)
        (out / 'fig-guardoff-build.log').write_text(proc.stdout + proc.stderr)
        result['figure'] = dict(compiled=proc.returncode == 0 and pdf.is_file(), returncode=proc.returncode)
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    if compiler and not result['figure']['compiled']:
        raise RuntimeError(f'Figure compilation failed: see {out}/fig-guardoff-build.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs/E009-guard-off/cem_release_guardoff')
    parser.add_argument('--units', type=Path, default=ROOT / 'runs/E002-planner-loop')
    parser.add_argument('--out', type=Path, default=Path(__file__).parent / 'out')
    args = parser.parse_args()
    missing = [args.runs / f'inst{i}-seed0/unit.json' for i in INSTANCES
               if not (args.runs / f'inst{i}-seed0/unit.json').is_file()]
    if missing:
        raise FileNotFoundError('Guard-off units absent; run/export E009 first: ' + ', '.join(map(str, missing)))
    hashes = {}
    def read(path):
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    guardoff, native, cap = {}, {}, {}
    for instance in INSTANCES:
        key = f'inst{instance}-seed0/unit.json'
        guardoff[instance] = read(args.runs / key)
        native[instance] = read(args.units / 'cem_release' / key)
        cap[instance] = read(args.units / 'cem_clampexec' / key)
        validate(guardoff[instance], instance, 'cem_release', guardoff=True)
        validate(native[instance], instance, 'cem_release')
        validate(cap[instance], instance, 'cem_clampexec')
    synthetic = any(u.get('dataset_kind') == 'synthetic-code-path-test' for u in guardoff.values())
    if synthetic:
        scratch = Path(os.environ['TMPDIR']).resolve()
        if not all(p.resolve().is_relative_to(scratch) for p in (args.runs, args.units, args.out)):
            raise ValueError('Synthetic inputs and outputs must remain under TMPDIR')
    result = evaluate(guardoff, native, cap)
    result.update(seed=0, reach_v0_unit=False, input_sha256=hashes,
                  dataset_kind='synthetic-code-path-test' if synthetic else 'guard-off-intervention')
    write_artifacts(args.out, result)
    for number, p in result['predictions'].items():
        for clause, values in p['clauses'].items():
            print(f'Prediction {number}, {clause}: ' + json.dumps(values, allow_nan=False))
    print(f'Outputs: {args.out}')


if __name__ == '__main__':
    main()
