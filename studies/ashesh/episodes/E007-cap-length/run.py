"""Seed-0 cap dose-response; numpy/stdlib only, with partial arms labelled."""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
INSTANCES = tuple(range(100, 108))
CAPS = {'cem_release': None, 'cem_cap10': 0.10, 'cem_clampexec': 0.05}
NEAR = (100, 107)
FAR = (101, 102, 103, 105, 106)
BOOTSTRAP_SEED = 20261113
BOOTSTRAP_DRAWS = 10000


def read_input(path, hashes):
    raw = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw.decode('utf-8')


def unit_metrics(unit, cap):
    """Use the fixed scorer's progress; minimum includes the initial distance."""
    ticks = unit['ticks']
    d0 = unit['initial_distance_m']
    distances = [t['distance_m'] for t in ticks]
    commanded = np.asarray([t['commanded'] for t in ticks], dtype=float)
    executed = np.asarray([t['executed'] for t in ticks], dtype=float)
    assert commanded.shape == executed.shape == (5, 3)
    assert d0 > 0 and np.isfinite([d0, *distances]).all()
    assert np.isfinite(commanded).all() and np.isfinite(executed).all()
    unmoved = (np.linalg.norm(executed, axis=1) < 0.001) & (
        np.linalg.norm(commanded, axis=1) >= 0.01)
    capped = (np.abs(commanded) > 0.05 if cap is None else
              np.abs(commanded) >= cap - 1e-9).any(axis=1)
    return {
        'D0_m': d0, 'distance_after_tick_m': distances,
        'final_distance_m': distances[-1], 'min_distance_m': min(d0, *distances),
        'progress': 1 - distances[-1] / d0,
        'unmoved_ticks': int(unmoved.sum()),
        'unmoved_tick_ids': [t['tick'] for t, flag in zip(ticks, unmoved) if flag],
        'non_converged_ticks': sum(not t['blocking_converged'] for t in ticks),
        'non_converged_tick_ids': [t['tick'] for t in ticks if not t['blocking_converged']],
        'at_or_above_cap_ticks': int(capped.sum()),
        'at_or_above_cap_tick_ids': [t['tick'] for t, flag in zip(ticks, capped) if flag],
    }


def summarize(metrics):
    progress = [m['progress'] for m in metrics.values()]

    def group_median(instances):
        values = [metrics[i]['final_distance_m'] for i in instances if i in metrics]
        return float(np.median(values)) if values else None

    return {
        'units_present': len(metrics),
        'status': 'complete' if len(metrics) == 8 else 'partial' if metrics else 'missing',
        'present_instances': sorted(metrics),
        'missing_instances': sorted(set(INSTANCES) - metrics.keys()),
        'mean_progress': float(np.mean(progress)) if progress else None,
        'median_progress': float(np.median(progress)) if progress else None,
        'positive_progress_instances': sum(p > 0 for p in progress),
        'final_within_2cm_instances': sum(m['final_distance_m'] <= 0.02 for m in metrics.values()),
        'total_unmoved_ticks': sum(m['unmoved_ticks'] for m in metrics.values()),
        'total_non_converged_ticks': sum(m['non_converged_ticks'] for m in metrics.values()),
        'total_at_or_above_cap_ticks': sum(m['at_or_above_cap_ticks'] for m in metrics.values()),
        'near_median_final_distance_m': group_median(NEAR),
        'near_instances_present': [i for i in NEAR if i in metrics],
        'far_median_final_distance_m': group_median(FAR),
        'far_instances_present': [i for i in FAR if i in metrics],
    }


def paired(metrics, baseline):
    """Arm minus baseline; bootstrap the mean of paired instance differences."""
    common = sorted(metrics.keys() & baseline.keys())
    if not common:
        return None
    progress = np.array([metrics[i]['progress'] - baseline[i]['progress'] for i in common])
    distance = np.array([metrics[i]['final_distance_m'] - baseline[i]['final_distance_m']
                         for i in common])
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(
        0, len(common), (BOOTSTRAP_DRAWS, len(common)))
    return {
        'paired_instances': len(common), 'instances': common,
        'per_instance': {i: {'delta_progress': float(p), 'delta_final_distance_cm': float(d * 100)}
                         for i, p, d in zip(common, progress, distance)},
        'mean_delta_progress': float(progress.mean()),
        'mean_delta_progress_ci95': np.quantile(progress[draws].mean(axis=1), [.025, .975]).tolist(),
        'bootstrap_draws': BOOTSTRAP_DRAWS, 'bootstrap_seed': BOOTSTRAP_SEED,
        'closer_instances': int((distance < -0.001).sum()),
        'farther_instances': int((distance > 0.001).sum()),
        'within_1mm_instances': int((np.abs(distance) <= 0.001).sum()),
        'closer_by_at_least_5cm_instances': int((distance <= -0.05).sum()),
        'final_within_1cm_of_baseline_instances': int((np.abs(distance) <= 0.01).sum()),
    }


def tick0_comparison(units, baseline, prior, cap):
    """Counterfactual uses the baseline command, never the capped arm's command."""
    unique, ambiguous = {}, {}
    for i, base in sorted(baseline.items()):
        g = prior[i]
        assert g['status'] in ('unique', 'ambiguous')
        clipped = np.clip(base['ticks'][0]['commanded'], -cap, cap)
        actual = units[i]['ticks'][0] if i in units else None
        row = {
            'goal_status': g['status'], 'ambiguous': g['status'] == 'ambiguous',
            'arm_present': actual is not None,
            'clipped_baseline_command_m': clipped.tolist(),
            'predicted_distance_m': None, 'realized_distance_m': actual['distance_m'] if actual else None,
            'absolute_prediction_error_cm': None, 'within_1cm': None,
            'max_command_deviation_mm': float(1000 * np.max(np.abs(
                np.asarray(actual['commanded']) - clipped))) if actual else None,
        }
        if g['status'] == 'unique':
            row['predicted_distance_m'] = float(np.linalg.norm(np.asarray(g['g_m']) - clipped))
            if actual:
                error = abs(actual['distance_m'] - row['predicted_distance_m'])
                row['absolute_prediction_error_cm'] = 100 * error
                row['within_1cm'] = error <= 0.01
            unique[i] = row
        else:
            row['exclusion_reason'] = 'Mirror-ambiguous goal; excluded from prediction agreement.'
            ambiguous[i] = row
    return {
        'unique_g': unique, 'ambiguous_g': ambiguous,
        'ambiguous_instances': sorted(ambiguous),
        'unique_instances_compared': sum(r['arm_present'] for r in unique.values()),
        'within_1cm_instances': sum(r['within_1cm'] is True for r in unique.values()),
    }


def analyze(runs):
    hashes, excluded, units, arms = {}, [], {}, {}
    prior_path = ROOT / 'episodes/E006-action-interface/out/prior.json'
    prior = {r['instance']: r for r in json.loads(read_input(prior_path, hashes))['instances']}
    for arm, cap in CAPS.items():
        units[arm] = {}
        for i in INSTANCES:
            path = runs / arm / f'inst{i}-seed0/unit.json'
            if not path.exists():
                continue
            unit = json.loads(read_input(path, hashes))
            assert (unit['planner'], unit['instance'], unit['seed']) == (arm, i, 0), path
            if unit.get('complete') is False or len(unit['ticks']) != 5:
                excluded.append({'path': str(path), 'reason': 'incomplete unit'})
                continue
            assert [t['tick'] for t in unit['ticks']] == list(range(5)), path
            units[arm][i] = unit
        metrics = {i: unit_metrics(u, cap) for i, u in units[arm].items()}
        arms[arm] = {'cap_m': cap, **summarize(metrics), 'per_instance': metrics,
                     'paired_vs_cem_release': None, 'tick0': None, 'decision': None}

    baseline = arms['cem_release']
    leaderboard = ROOT / 'bench/leaderboard.jsonl'
    rows = [json.loads(line) for line in read_input(leaderboard, hashes).splitlines() if line.strip()]
    last = [row for row in rows if row['planner'] == 'cem_release'][-1]
    expected = {int(i): p for i, p in last['per_instance'].items()}
    observed = baseline['per_instance']
    assert set(observed) == set(expected) == set(INSTANCES), 'Baseline/leaderboard instances differ'
    max_error = max(abs(observed[i]['progress'] - expected[i]) for i in expected)
    assert max_error <= 1e-9, f'Baseline progress differs from last leaderboard row: {max_error}'

    for arm, cap in CAPS.items():
        if cap is None:
            continue
        row = arms[arm]
        comparison = paired(row['per_instance'], observed)
        row['paired_vs_cem_release'] = comparison
        row['tick0'] = tick0_comparison(units[arm], units['cem_release'], prior, cap)
        if row['units_present'] == 8:
            assert comparison is not None
            row['decision'] = bool(row['positive_progress_instances'] == 8
                                   and comparison['mean_delta_progress'] > 0
                                   and row['median_progress'] >= baseline['median_progress'] - 0.03)

    return {
        'runs_root': str(runs), 'seed': 0, 'expected_instances': list(INSTANCES),
        'arms_present': [arm for arm, row in arms.items() if row['units_present']],
        'arms': arms, 'input_sha256': hashes, 'excluded_inputs': excluded,
        'baseline_leaderboard_check': {'passed': True, 'max_absolute_error': max_error,
                                       'row': 'last cem_release row', 'tolerance': 1e-9},
        'definitions': {
            'progress': '1 - final_distance_m / D0_m (fixed bench/score.py definition)',
            'minimum_distance': 'minimum over start and five post-tick distances',
            'unmoved': 'norm(executed) < 0.001 m and norm(commanded) >= 0.01 m',
            'at_or_above_cap': 'any abs(commanded) > 0.05 m for none; >= cap - 1e-9 for capped arms',
            'paired_difference': 'arm minus cem_release; negative distance difference is closer',
            'closer_farther': 'difference strictly beyond +/-0.001 m; boundary is a tie',
            'bootstrap': '95% percentile interval; resample common instances, mean paired progress difference',
            'tick0': 'norm(g_m - clip(baseline tick-0 commanded, -cap, cap)); unique goals only',
            'command_sanity': 'max absolute arm tick-0 command minus clipped baseline command, in mm',
            'decision': 'complete 8/8 only: all progress > 0; mean paired progress difference > 0; median >= baseline median - 0.03',
            'near_starts': list(NEAR), 'far_starts': list(FAR),
        },
    }


def write_tables(result, out):
    arms = result['arms']
    baseline = arms['cem_release']['per_instance']
    ordered = sorted(baseline, key=lambda i: (baseline[i]['D0_m'], i))
    lines = ['instance D0_cm final_none_cm final_cap10_cm final_cap5_cm progress_none progress_cap10 progress_cap5']
    for i in ordered:
        metrics = [arms[arm]['per_instance'].get(i) for arm in CAPS]
        values = [baseline[i]['D0_m'] * 100]
        values += [m['final_distance_m'] * 100 if m else float('nan') for m in metrics]
        values += [m['progress'] if m else float('nan') for m in metrics]
        lines.append(str(i) + ' ' + ' '.join(f'{v:.12g}' for v in values))
    (out / 'cap.dat').write_text('\n'.join(lines) + '\n')
    lines = ['arm instance tick distance_cm']
    for arm, row in arms.items():
        for i in ordered:
            if i not in row['per_instance']:
                continue
            m = row['per_instance'][i]
            for tick, distance in enumerate([m['D0_m'], *m['distance_after_tick_m']]):
                lines.append(f'{arm} {i} {tick} {100 * distance:.12g}')
    (out / 'trajectories.dat').write_text('\n'.join(lines) + '\n')
    return ordered


def write_figure(result, out, ordered):
    baseline = result['arms']['cem_release']['per_instance']
    labels = ','.join(r'{\shortstack{' + str(i) + r'\\(' +
                      f"{100 * baseline[i]['D0_m']:.1f}" + ')}}' for i in ordered)
    tex = r"""\documentclass[12pt,border=6pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\begin{tikzpicture}
\begin{axis}[
 width=16cm,height=9cm,
 font=\large, label style={font=\Large},
 xlabel={Instance (start distance, cm)}, ylabel={Final distance (cm)},
 ymode=log, log basis y=10, ymin=0.1,
 xmin=-0.5,xmax=7.5, xtick={0,1,2,3,4,5,6,7},
 xticklabels={LABELS},
 unbounded coords=discard, filter discard warning=false,
 ymajorgrids=true, grid style={gray!20},
 legend style={at={(0.5,1.02)},anchor=south,draw=none},legend columns=3,
]
\addplot[only marks,mark=*,mark size=3pt,black]
 table[x expr=\coordindex,y=final_none_cm] {cap.dat};
\addlegendentry{No cap}
\addplot[only marks,mark=square*,mark size=3pt,blue!75!black]
 table[x expr=\coordindex,y=final_cap10_cm] {cap.dat};
\addlegendentry{10 cm}
\addplot[only marks,mark=triangle*,mark size=3.5pt,orange!85!black]
 table[x expr=\coordindex,y=final_cap5_cm] {cap.dat};
\addlegendentry{5 cm}
\addplot[dashed,gray,thick,forget plot] coordinates {(-0.5,2) (7.5,2)};
\end{axis}
\end{tikzpicture}
\end{document}
""".replace('LABELS', labels)
    (out / 'fig-cap.tex').write_text(tex)
    latex = shutil.which('pdflatex')
    if not latex:
        (out / 'fig-cap.pdf').unlink(missing_ok=True)
        return {'status': 'not_compiled', 'reason': 'pdflatex not found'}
    (out / 'fig-cap.pdf').unlink(missing_ok=True)
    try:
        proc = subprocess.run([latex, '-interaction=nonstopmode', '-halt-on-error', 'fig-cap.tex'],
                              cwd=out, text=True, capture_output=True, timeout=120)
        if proc.returncode:
            raise RuntimeError('pdflatex failed:\n' + proc.stdout + proc.stderr)
        assert (out / 'fig-cap.pdf').is_file()
    finally:
        for suffix in ('.aux', '.log'):
            (out / ('fig-cap' + suffix)).unlink(missing_ok=True)
    return {'status': 'compiled', 'path': str(out / 'fig-cap.pdf')}


def print_summary(result):
    def number(value):
        return 'pending' if value is None else f'{value:.3f}'

    print('arm             units status      mean  median positive <=2cm unmoved decision')
    for arm, row in result['arms'].items():
        decision = 'pending' if row['decision'] is None else str(row['decision']).lower()
        if arm == 'cem_release':
            decision = '-'
        print(f"{arm:15s} {row['units_present']:>2}/8  {row['status']:8s} "
              f"{number(row['mean_progress']):>7s} {number(row['median_progress']):>7s} "
              f"{row['positive_progress_instances']:>8d} {row['final_within_2cm_instances']:>5d} "
              f"{row['total_unmoved_ticks']:>7d} {decision}")
    print('instance  progress_none  progress_cap10  progress_cap5')
    for i in INSTANCES:
        values = [number(row['per_instance'].get(i, {}).get('progress'))
                  for row in result['arms'].values()]
        print(f'{i:8d} ' + ' '.join(f'{v:>14s}' for v in values))
    for arm, row in result['arms'].items():
        pair = row['paired_vs_cem_release']
        if pair is not None:
            lo, hi = pair['mean_delta_progress_ci95']
            t0 = row['tick0']
            print(f"{arm}: paired n={pair['paired_instances']}, mean delta={pair['mean_delta_progress']:.3f}, "
                  f"95% CI [{lo:.3f}, {hi:.3f}]; tick-0 within 1 cm "
                  f"{t0['within_1cm_instances']}/{t0['unique_instances_compared']} unique goals")
    ambiguous = sorted({i for row in result['arms'].values() if row['tick0'] is not None
                        for i in row['tick0']['ambiguous_instances']})
    print('Tick-0 ambiguous goals excluded: ' + ', '.join(map(str, ambiguous)))
    print('Baseline leaderboard self-check: PASS (atol=1e-9)')
    print('PDF: ' + ('compiled' if result['figure']['status'] == 'compiled' else 'not compiled (pdflatex not found)'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs/E002-planner-loop')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'out',
                        help='Output directory; override to isolate synthetic code-path tests.')
    args = parser.parse_args()
    result = analyze(args.runs)
    args.out.mkdir(parents=True, exist_ok=True)
    ordered = write_tables(result, args.out)
    result['figure'] = write_figure(result, args.out, ordered)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print_summary(result)


if __name__ == '__main__':
    main()
