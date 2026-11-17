"""Seed-0 GN-in-cube analysis; CPU/numpy only, incomplete arms stay pending."""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.score import paired_bootstrap

INSTANCES = tuple(range(100, 108))
CEM_ARMS = ('cem_release', 'cem_clampexec')
GN_ARMS = ('gn_cube_a', 'gn_cube_b')
ARMS = CEM_ARMS + GN_ARMS
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261113


def read_input(path, hashes):
    raw = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    return raw.decode('utf-8')


def load_units(runs, hashes):
    units, excluded = {}, []
    for arm in ARMS:
        units[arm] = {}
        for i in INSTANCES:
            path = runs / arm / f'inst{i}-seed0/unit.json'
            if not path.exists():
                continue
            unit = json.loads(read_input(path, hashes))
            if (unit['planner'], unit['instance'], unit['seed']) != (arm, i, 0):
                raise ValueError(f'Unit identity mismatch: {path}')
            if not unit.get('complete') or len(unit['ticks']) != 5:
                excluded.append({'path': str(path), 'reason': 'incomplete unit'})
                continue
            ticks = unit['ticks']
            if [t['tick'] for t in ticks] != list(range(5)):
                raise ValueError(f'Expected ticks 0..4: {path}')
            numbers = [unit['initial_distance_m']]
            for t in ticks:
                numbers.extend(t[k] for k in ('distance_m', 'cost_before', 'cost_after',
                                               'cost_model_before', 'cost_predicted',
                                               'planning_s', 'predicted_samples'))
                if len(t['commanded']) != 3:
                    raise ValueError(f'Expected 3D command: {path}')
                numbers.extend(t['commanded'])
                if t['rho'] is not None:
                    numbers.append(t['rho'])
                if t['distance_m'] < 0 or t['planning_s'] < 0 or t['predicted_samples'] < 0:
                    raise ValueError(f'Negative distance/time/rollouts: {path}')
            if unit['initial_distance_m'] <= 0 or not np.isfinite(numbers).all():
                raise ValueError(f'Invalid unit numerics: {path}')
            units[arm][i] = unit
    return units, excluded


def progress(unit):
    return 1 - unit['ticks'][-1]['distance_m'] / unit['initial_distance_m']


def anchor_gate(units):
    """Check historical controls before creating or touching any output."""
    checks = {}
    for arm in CEM_ARMS:
        if set(units[arm]) != set(INSTANCES):
            raise ValueError(f'Anchor failed: {arm} requires all eight complete units')
    expected_decreases = [0.0371, 0.0423, 0.0467, 0.0309, 0.0319, 0.0384, 0.0264, 0.0242]
    expected_commands = [(2.93, 3.07, -5), (5, 5, 5), (5, -5, 3.84), (5, 5, 5),
                         (-3.56, -1.07, 2.85), (5, -4.8, 5), (-1.82, -5, 5), (5, -3.79, 5)]

    def check(name, observed, expected, tolerance):
        error = float(np.max(np.abs(np.asarray(observed) - np.asarray(expected))))
        checks[name] = {'observed': observed, 'expected': expected,
                        'max_absolute_error': error, 'tolerance': tolerance, 'passed': error <= tolerance}
        if error > tolerance:
            raise ValueError(f'Anchor failed: {name}: max error {error:.9g} > {tolerance}')

    ticks = [units['cem_clampexec'][i]['ticks'][0] for i in INSTANCES]
    check('clampexec_tick0_predicted_decrease',
          [t['cost_model_before'] - t['cost_predicted'] for t in ticks], expected_decreases, 5e-5)
    check('clampexec_tick0_command_cm',
          [(100 * np.asarray(t['commanded'])).tolist() for t in ticks], expected_commands, 0.006)
    for arm, expected_mean, expected_median in [('cem_release', 0.587, 0.9509411),
                                               ('cem_clampexec', 0.888, 0.9167)]:
        values = [progress(units[arm][i]) for i in INSTANCES]
        check(arm + '_mean_progress', float(np.mean(values)), expected_mean, 1e-3)
        check(arm + '_median_progress', float(np.median(values)), expected_median, 1e-3)
    return {'passed': True, 'checks': checks}


def planner_logs(runs, arm, instance, hashes):
    path = runs / arm / f'inst{instance}-seed0/planner_log.json'
    if arm not in GN_ARMS or not path.exists():
        return {}
    sidecar = json.loads(read_input(path, hashes))
    if (sidecar['planner'], sidecar['instance'], sidecar['seed']) != (arm, instance, 0):
        raise ValueError(f'Sidecar identity mismatch: {path}')
    logs = {}
    for item in sidecar['ticks']:
        tick, log = item['tick'], item['log']
        if tick not in range(5) or tick in logs:
            raise ValueError(f'Invalid/duplicate sidecar tick: {path}')
        logs[tick] = None if log is None else {
            'stop': log.get('stop_reason', log.get('status')),
            'rows': log.get('rows_used', log.get('rows')),
            'linearisations': log.get('linearisations'),
            'accepted_steps': log.get('accepted_steps'),
            'rejected_steps': log.get('rejected_steps'),
        }
    return logs


def unit_metrics(unit, logs):
    ticks = [{**t, 'distance_cm': 100 * t['distance_m'],
              'predicted_decrease': t['cost_model_before'] - t['cost_predicted'],
              'planner_log': logs.get(t['tick'])} for t in unit['ticks']]
    return {
        'initial_distance_m': unit['initial_distance_m'], 'progress': progress(unit),
        'final_distance_cm': ticks[-1]['distance_cm'],
        'distance_after_tick_cm': [t['distance_cm'] for t in ticks],
        'median_planning_s': float(np.median([t['planning_s'] for t in ticks])),
        'median_predicted_samples': float(np.median([t['predicted_samples'] for t in ticks])),
        'max_predicted_samples': max(t['predicted_samples'] for t in ticks), 'ticks': ticks,
    }


def summarize(metrics):
    values = [m['progress'] for m in metrics.values()]
    ticks = [t for m in metrics.values() for t in m['ticks']]
    n = len(metrics)
    return {
        'units_present': n, 'status': 'complete' if n == 8 else f'partial {n}/8' if n else 'missing',
        'present_instances': sorted(metrics), 'missing_instances': sorted(set(INSTANCES) - metrics.keys()),
        'mean_progress': float(np.mean(values)) if values else None,
        'median_progress': float(np.median(values)) if values else None,
        'ticks_present': len(ticks),
        'median_planning_s': float(np.median([t['planning_s'] for t in ticks])) if ticks else None,
        'min_predicted_samples': min(t['predicted_samples'] for t in ticks) if ticks else None,
        'max_predicted_samples': max(t['predicted_samples'] for t in ticks) if ticks else None,
        'per_instance': metrics,
    }


def paired_mean(metrics, baseline):
    """Secondary statistic: bootstrap the mean of paired instance differences."""
    common = sorted(metrics.keys() & baseline.keys())
    if not common:
        return None
    differences = np.array([metrics[i]['progress'] - baseline[i]['progress'] for i in common])
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(common),
                                                          (BOOTSTRAP_DRAWS, len(common)))
    return {
        'label': 'secondary, not the fixed headline', 'paired_instances': len(common),
        'instances': common, 'per_instance_delta_progress': dict(zip(common, differences.tolist())),
        'mean_delta_progress': float(differences.mean()),
        'ci95': np.quantile(differences[draws].mean(axis=1), [.025, .975]).tolist(),
        'bootstrap_draws': BOOTSTRAP_DRAWS, 'bootstrap_seed': BOOTSTRAP_SEED,
    }


def judge(row, baseline, cem_reference):
    metrics, reference = row['per_instance'], baseline['per_instance']
    complete = row['units_present'] == 8

    def verdict(condition):
        return bool(condition) if complete else None

    ratios, commands, distances, resets = {}, {}, {}, {}
    for i in INSTANCES:
        ratios[i] = commands[i] = distances[i] = None
        if i not in metrics:
            continue
        m, base = metrics[i], reference[i]
        t, b = m['ticks'][0], base['ticks'][0]
        ratios[i] = t['predicted_decrease'] / b['predicted_decrease']
        commands[i] = float(100 * np.linalg.norm(np.asarray(t['commanded']) - b['commanded']))
        distances[i] = abs(m['final_distance_cm'] - base['final_distance_cm'])
        differences = {key: t[key] - b[key] for key in ('cost_before', 'cost_model_before')}
        resets[i] = {'signed_differences_gn_minus_clampexec': differences,
                     'absolute_differences': {key: abs(d) for key, d in differences.items()},
                     'tolerance': 1e-4, 'flag': any(abs(d) > 1e-4 for d in differences.values())}
    row['reset_identity'] = resets
    row['mean_progress_minus_clampexec'] = (
        row['paired_mean_vs_cem_clampexec']['mean_delta_progress'] if metrics else None)
    row['mean_progress_minus_clampexec_scope'] = 'paired available instances; full-arm comparison only at 8/8'
    row['plan_change_mean_gain_exceeds_0_05'] = verdict(
        bool(metrics) and row['mean_progress_minus_clampexec'] > 0.05)
    count1 = sum(r is not None and r >= 0.90 for r in ratios.values())
    count2 = sum(d is not None and d <= 1.5 for d in commands.values())
    positive = sum(m['progress'] > 0 for m in metrics.values())
    close = sum(d is not None and d <= 1.5 for d in distances.values())
    ticks = [t for m in metrics.values() for t in m['ticks']]
    time_ok = bool(ticks) and row['median_planning_s'] <= 10
    rollouts_ok = bool(ticks) and row['max_predicted_samples'] <= 150
    mean_ok = bool(metrics) and row['mean_progress'] >= 0.80
    headline = row['paired_vs_cem_release']
    headline_ok = headline is not None and (headline['delta_median'] < 0 or
                                            headline['ci95'][0] <= 0 <= headline['ci95'][1])
    return {
        'P1': {'count': count1, 'observed_instances': len(metrics), 'total_instances': 8,
               'threshold': {'ratio_min': 0.90, 'count_min': 6}, 'ratios': ratios,
               'holds': verdict(count1 >= 6)},
        'P2': {'count': count2, 'observed_instances': len(metrics), 'total_instances': 8,
               'threshold': {'command_distance_cm_max': 1.5, 'count_min': 5},
               'command_distances_cm': commands, 'holds': verdict(count2 >= 5)},
        'P3': {'ticks_present': len(ticks), 'total_ticks': 40,
               'median_planning_s': row['median_planning_s'],
               'max_predicted_samples': row['max_predicted_samples'],
               'threshold': {'median_planning_s_max': 10, 'predicted_samples_max': 150},
               'planning_s_at_most_10_count': sum(t['planning_s'] <= 10 for t in ticks),
               'predicted_samples_at_most_150_count': sum(t['predicted_samples'] <= 150 for t in ticks),
               'time_holds': verdict(time_ok), 'rollouts_hold': verdict(rollouts_ok),
               'cem_reference': cem_reference, 'holds': verdict(time_ok and rollouts_ok)},
        'P4': {'observed_instances': len(metrics), 'total_instances': 8,
               'positive_progress': {'count': positive, 'threshold': {'progress_strictly_above': 0,
                    'count_required': 8}, 'holds': verdict(positive == 8)},
               'mean_progress': {'value': row['mean_progress'], 'threshold': 0.80,
                                 'holds': verdict(mean_ok)},
               'final_distance_agreement': {'count': close,
                    'threshold': {'absolute_difference_cm_max': 1.5, 'count_min': 5},
                    'absolute_differences_cm': distances, 'holds': verdict(close >= 5)},
               'holds': verdict(positive == 8 and mean_ok and close >= 5)},
        'P5': {'paired_instances': headline['paired_instances'] if headline else 0,
               'total_instances': 8, 'threshold': 'delta_median < 0 OR ci95[0] <= 0 <= ci95[1]',
               'paired_vs_cem_release': headline,
               'paired_vs_cem_clampexec': row['paired_vs_cem_clampexec'],
               'holds': verdict(headline_ok)},
    }


def analyze(runs, units, hashes, excluded, anchors):
    # Hash the analysis contract and unchanged fixed evaluator alongside receipts.
    for path in (Path(__file__), Path(__file__).with_name('episode.md'), ROOT / 'bench/score.py'):
        read_input(path, hashes)
    arms = {}
    for arm in ARMS:
        metrics = {i: unit_metrics(u, planner_logs(runs, arm, i, hashes)) for i, u in units[arm].items()}
        arms[arm] = summarize(metrics)
    reference = {
        'episode_rounded': {'median_planning_s': 211, 'predicted_samples_range': [10240, 15360]},
        'observed': {arm: {key: arms[arm][key] for key in
                    ('median_planning_s', 'min_predicted_samples', 'max_predicted_samples')}
                     for arm in CEM_ARMS},
    }
    for arm in GN_ARMS:
        row = arms[arm]
        for base in CEM_ARMS:
            row['paired_vs_' + base] = paired_bootstrap(
                {i: m['progress'] for i, m in row['per_instance'].items()},
                {i: m['progress'] for i, m in arms[base]['per_instance'].items()})
            row['paired_mean_vs_' + base] = paired_mean(row['per_instance'], arms[base]['per_instance'])
        row['predictions'] = judge(row, arms['cem_clampexec'], reference)
    kept, kept_reason = None, 'Both GN arms must have all eight complete units.'
    if all(arms[arm]['units_present'] == 8 for arm in GN_ARMS):
        a, b = (arms[arm]['mean_progress'] for arm in GN_ARMS)
        if a == b:
            kept_reason = 'Equal mean progress: neither implementation is higher.'
        else:
            kept = GN_ARMS[0] if a > b else GN_ARMS[1]
            kept_reason = 'Higher mean progress over all eight dev instances.'
    return {
        'runs_root': str(runs), 'seed': 0, 'expected_instances': list(INSTANCES),
        'arms': arms, 'input_sha256': hashes, 'excluded_inputs': excluded,
        'anchor_gate': anchors, 'kept': kept, 'kept_reason': kept_reason,
        'definitions': {
            'progress': '1 - ticks[-1].distance_m / initial_distance_m (fixed scorer)',
            'predicted_decrease': 'cost_model_before - cost_predicted',
            'command_distance': 'Euclidean norm of tick-0 GN minus cem_clampexec command, cm',
            'planning_time': 'median over all available ticks, not median of unit medians',
            'rollouts': 'predicted_samples, not predictor_calls or sidecar rollout_calls',
            'fixed_headline': 'bench.score.paired_bootstrap unchanged: difference of medians, paired instance resampling',
            'secondary': 'mean of paired progress differences; 10000 draws, seed 20261113; not the fixed headline',
            'partial': 'descriptive statistics on available complete units; all prediction holds null until 8/8',
            'figure': 'one point per available unit; median predicted_samples vs final distance_cm; both axes log',
        },
        'not_available_from_units': [
            'Planner accepted/rejected steps, linearisations and stop reason require optional planner_log.json.',
            'Latent residuals/Jacobians, finite-difference smoothness, axis cost sweeps and physical per-axis goal offsets are not in unit receipts.',
            'Matched-rollout cem_small, held-out seeds and causal attribution to curvature require separate experiments.',
        ],
    }


def figure_label_positions(arms):
    """Separate labels in log-y space; never jitter the measured coordinates."""
    positive = [m['final_distance_cm'] for row in arms.values() for m in row['per_instance'].values()
                if m['final_distance_cm'] > 0 and m['median_predicted_samples'] > 0]
    span = max(float(np.log10(max(positive) / min(positive))), 0.5) if positive else 1.0
    gap = span * 0.065
    labels = {}
    for arm, row in arms.items():
        ordered = sorted((m['final_distance_cm'], i) for i, m in row['per_instance'].items()
                         if m['final_distance_cm'] > 0 and m['median_predicted_samples'] > 0)
        if not ordered:
            labels[arm] = {}
            continue
        original = np.log10([distance for distance, _ in ordered])
        positions = original.copy()
        for j in range(1, len(positions)):
            positions[j] = max(positions[j], positions[j - 1] + gap)
        positions -= float(np.mean(positions - original))
        labels[arm] = {i: float(10 ** y) for (_, i), y in zip(ordered, positions)}
    return labels


def write_tables(result, out):
    arms = result['arms']
    label_positions = figure_label_positions(arms)
    lines = ['instance arm predicted_decrease_gn predicted_decrease_cem_clampexec ratio '
             'command_distance_cm median_planning_s median_predicted_samples max_predicted_samples']
    for i in INSTANCES:
        base = arms['cem_clampexec']['per_instance'][i]['ticks'][0]['predicted_decrease']
        for arm in GN_ARMS:
            row = arms[arm]
            m = row['per_instance'].get(i)
            values = [m['ticks'][0]['predicted_decrease'] if m else None, base,
                      row['predictions']['P1']['ratios'][i],
                      row['predictions']['P2']['command_distances_cm'][i]]
            values += [m[key] if m else None for key in
                       ('median_planning_s', 'median_predicted_samples', 'max_predicted_samples')]
            lines.append(f'{i} {arm} ' + ' '.join(number(v) for v in values))
    (out / 'search.dat').write_text('\n'.join(lines) + '\n')
    columns = ('progress', 'final_distance_cm', 'median_predicted_samples', 'label_distance_cm')
    lines = ['instance ' + ' '.join(f'{arm}_{key}' for arm in ARMS for key in columns)]
    for i in INSTANCES:
        values = [label_positions[arm].get(i) if key == 'label_distance_cm' else
                  arms[arm]['per_instance'].get(i, {}).get(key) for arm in ARMS for key in columns]
        lines.append(str(i) + ' ' + ' '.join(number(v) for v in values))
    (out / 'progress.dat').write_text('\n'.join(lines) + '\n')


def number(value):
    return 'nan' if value is None else f'{value:.12g}'


def write_figure(result, out):
    tex = r"""\documentclass[12pt,border=8pt]{standalone}
\usepackage{pgfplots}
\pgfplotsset{compat=1.18}
\begin{document}
\pgfplotstableread{progress.dat}\progressdata
\begin{tikzpicture}
\begin{axis}[
 width=19cm,height=12cm,
 font=\large,label style={font=\Large},
 xlabel={Rollouts per tick (unit median)},ylabel={Final distance (cm)},
 xmode=log,ymode=log,log basis x=10,log basis y=10,
 enlarge x limits=0.25,enlarge y limits=0.2,
 unbounded coords=discard,filter discard warning=false,
 grid=major,grid style={gray!20},
 legend style={at={(0.5,1.03)},anchor=south,draw=none,font=\large,column sep=12pt},legend columns=2,
]
"""
    styles = [('black', '*', -8, 'CEM release'),
              ('blue!75!black', 'square*', 8, 'CEM clamp-exec'),
              ('orange!85!black', 'triangle*', -8, 'GN A'),
              ('teal!75!black', 'diamond*', 8, 'GN B')]
    for arm, (color, mark, shift, label) in zip(ARMS, styles):
        tex += (r'\addplot[only marks,mark=' + mark + ',mark size=3pt,' + color + ']\n' +
                ' table[x=' + arm + '_median_predicted_samples,y=' + arm +
                '_final_distance_cm] {progress.dat};\n' + r'\addlegendentry{' + label + '}\n')
        for index, i in enumerate(INSTANCES):
            m = result['arms'][arm]['per_instance'].get(i)
            if not m or m['final_distance_cm'] <= 0 or m['median_predicted_samples'] <= 0:
                continue
            # Read both the point and its displaced label from the .dat table.
            for variable, column in [('px', 'median_predicted_samples'),
                                     ('py', 'final_distance_cm'), ('ly', 'label_distance_cm')]:
                tex += (r'\pgfplotstablegetelem{' + str(index) + '}{' + arm + '_' + column +
                        r'}\of\progressdata' + '\n' + '\\edef\\' + variable +
                        r'{\pgfplotsretval}' + '\n')
            anchor = 'east' if shift < 0 else 'west'
            tex += (r'\edef\labeldraw{\noexpand\draw[' + color + r',thin] ' +
                    r'(axis cs:\px,\py) -- ([xshift=' + str(shift) + r'pt]axis cs:\px,\ly)' +
                    ' node[anchor=' + anchor + ',font=\\noexpand\\normalsize,inner sep=2pt]{' +
                    str(i) + '};}\\labeldraw\n')
    tex += '\\end{axis}\n\\end{tikzpicture}\n\\end{document}\n'
    (out / 'fig-search.tex').write_text(tex)
    pdf = out / 'fig-search.pdf'
    pdf.unlink(missing_ok=True)
    # Log coordinates cannot represent zero. Report any such receipts explicitly.
    omitted = [{'arm': arm, 'instance': i} for arm, row in result['arms'].items()
               for i, m in row['per_instance'].items()
               if m['final_distance_cm'] <= 0 or m['median_predicted_samples'] <= 0]
    metadata = {'log_axis_nonpositive_units': omitted}
    latex = Path('/usr/bin/pdflatex')
    if not latex.is_file():
        return {**metadata, 'status': 'not_compiled', 'reason': '/usr/bin/pdflatex not found'}
    try:
        proc = subprocess.run([str(latex), '-interaction=nonstopmode', '-halt-on-error', 'fig-search.tex'],
                              cwd=out, text=True, capture_output=True, timeout=120)
        if proc.returncode or not pdf.is_file():
            pdf.unlink(missing_ok=True)
            return {**metadata, 'status': 'not_compiled',
                    'reason': f'pdflatex exit {proc.returncode}: ' + (proc.stdout + proc.stderr)[-4000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        pdf.unlink(missing_ok=True)
        return {**metadata, 'status': 'not_compiled', 'reason': str(exc)}
    finally:
        for suffix in ('.aux', '.log'):
            (out / ('fig-search' + suffix)).unlink(missing_ok=True)
    return {**metadata, 'status': 'compiled', 'path': str(pdf)}


def print_summary(result):
    for arm, row in result['arms'].items():
        print(f"{arm}: {row['status']}; mean={number(row['mean_progress'])}, "
              f"median={number(row['median_progress'])}")
        if arm not in GN_ARMS:
            continue
        for name, prediction in row['predictions'].items():
            print(f'  {name}: ' + json.dumps(prediction, sort_keys=True))
        for base in CEM_ARMS:
            print(f'  paired_vs_{base}: ' + json.dumps(row['paired_vs_' + base]))
            print(f'  paired_mean_vs_{base}: ' + json.dumps(row['paired_mean_vs_' + base]))
    print('kept: ' + json.dumps(result['kept']))
    print('Anchor gate: PASS')
    print('PDF: ' + result['figure']['status'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs/E002-planner-loop')
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parent / 'out',
                        help='Override to isolate synthetic code-path tests.')
    args = parser.parse_args()
    hashes = {}
    units, excluded = load_units(args.runs, hashes)
    anchors = anchor_gate(units)
    if not any(units[arm] for arm in GN_ARMS):
        print('No gn units: no output written.')
        return
    result = analyze(args.runs, units, hashes, excluded, anchors)
    # Validate JSON before the first output write, including optional sidecars.
    json.dumps(result, allow_nan=False)
    args.out.mkdir(parents=True, exist_ok=True)
    write_tables(result, args.out)
    result['figure'] = write_figure(result, args.out)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print_summary(result)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
