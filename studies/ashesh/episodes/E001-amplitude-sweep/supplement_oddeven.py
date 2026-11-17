"""Post-hoc reading of E001 (written after the fixed evaluator's output was seen).

Not part of the fixed evaluator and not a pre-registered quantity. It only
regroups `out/results.json` (the evaluator's per-candidate rows); no raw array
is read or changed.

For a symmetric command pair (+a u, -a u) with own-neutral cost change dC:
    odd(a, u)  = [dC(+a u) - dC(-a u)] / 2     direction-dependent response
    even(a, u) = [dC(+a u) + dC(-a u)] / 2     direction-independent response
u is the goal ray v_g ("onray") or one of the two orthogonal axes ("offray0/1").
G_odd = odd_pred / odd_real on the goal ray (null when |odd_real| is zero or
below 10x the instance's neutral-repeat noise, the evaluator's rule for G).

Usage: python supplement_oddeven.py   (writes out/oddeven.json, out/oddeven.dat,
out/signed_response.dat, out/fig_points.dat, out/fig_medians.dat)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / 'out'
AXES = {'onray': (0.001, 0.003, 0.01, 0.03), 'offray0': (0.01, 0.03), 'offray1': (0.01, 0.03)}
FIELDS = ('real', 'pred_cmd', 'pred_exec')


def pairs(instance):
    """One row per (axis, amplitude): odd/even parts of each cost response."""
    c = {r['name']: r for r in instance['candidates']}
    rows = []
    for axis, amplitudes in AXES.items():
        for a in amplitudes:
            plus, minus = c[f'{axis}-plus-{a:g}'], c[f'{axis}-minus-{a:g}']
            row = dict(instance=instance['instance'], axis=axis, commanded_mm=a * 1000,
                       executed_mm=(plus['executed_mm'] + minus['executed_mm']) / 2,
                       noise=instance['neutral_repeat_noise'])
            for f in FIELDS:
                row['odd_' + f] = (plus['dC_' + f] - minus['dC_' + f]) / 2
                row['even_' + f] = (plus['dC_' + f] + minus['dC_' + f]) / 2
            rows.append(row)
    return rows


def spread(v):
    v = np.asarray(v, float)
    return dict(n=int(v.size), median=float(np.median(v)), min=float(v.min()), max=float(v.max()))


def bootstrap_median(v, draws=2000, seed=0):
    """Percentile 95% interval of the median, resampling whole instances."""
    v = np.asarray(v, float)
    idx = np.random.default_rng(seed).integers(0, v.size, (draws, v.size))
    return np.quantile(np.median(v[idx], axis=1), [.025, .975]).tolist()


def main():
    result = json.loads((OUT / 'results.json').read_text())
    rows = [r for i in result['instances'] for r in pairs(i)]
    summary = {'note': 'post-hoc regrouping of the fixed evaluator output; exploratory', 'onray': [], 'offray': []}
    for a in AXES['onray']:
        sel = [r for r in rows if r['axis'] == 'onray' and r['commanded_mm'] == a * 1000]
        # Same validity rule as the evaluator's G: null below 10x neutral-repeat noise or at zero.
        g = [r['odd_pred_cmd'] / r['odd_real'] for r in sel
             if r['odd_real'] != 0 and abs(r['odd_real']) >= 10 * r['noise']]
        summary['onray'].append(dict(
            commanded_mm=a * 1000, executed_mm=spread([r['executed_mm'] for r in sel]),
            odd_real=spread([r['odd_real'] for r in sel]), odd_pred_cmd=spread([r['odd_pred_cmd'] for r in sel]),
            even_real=spread([r['even_real'] for r in sel]), even_pred_cmd=spread([r['even_pred_cmd'] for r in sel]),
            odd_sign_agree=int(sum(np.sign(r['odd_pred_cmd']) == np.sign(r['odd_real']) for r in sel)),
            odd_real_above_10x_noise=int(sum(abs(r['odd_real']) >= 10 * r['noise'] for r in sel)),
            G_odd=spread(g) if g else None,
            G_odd_median_ci95=bootstrap_median(g) if len(g) >= 4 else None))
    for a in (0.01, 0.03):
        sel = [r for r in rows if r['axis'] != 'onray' and r['commanded_mm'] == a * 1000]
        summary['offray'].append(dict(
            commanded_mm=a * 1000, n_pairs=len(sel),
            even_real=spread([r['even_real'] for r in sel]), even_pred_cmd=spread([r['even_pred_cmd'] for r in sel]),
            even_pred_negative=int(sum(r['even_pred_cmd'] < 0 for r in sel)),
            even_real_negative=int(sum(r['even_real'] < 0 for r in sel)),
            odd_real_abs=spread([abs(r['odd_real']) for r in sel]),
            odd_pred_abs=spread([abs(r['odd_pred_cmd']) for r in sel])))
    # Sign census over single candidates with commanded amplitude >= 1 cm.
    census = {}
    for i in result['instances']:
        for r in i['candidates']:
            if r['kind'] in ('onray', 'offray') and r['commanded_mm'] >= 9.99:
                name = r['name']
                group = ('toward' if name.startswith('onray-plus') else
                         'away' if name.startswith('onray-minus') else 'lateral')
                e = census.setdefault(group, dict(n=0, pred_cmd_decrease=0, real_decrease=0, D_decrease=0))
                e['n'] += 1
                e['pred_cmd_decrease'] += r['dC_pred_cmd'] < 0
                e['real_decrease'] += r['dC_real'] < 0
                neutral = next(c for c in i['candidates'] if c['name'] == i['neutral'])
                e['D_decrease'] += r['D_mm'] < neutral['D_mm']
    summary['sign_census_ge_1cm'] = census
    (OUT / 'oddeven.json').write_text(json.dumps(summary, indent=2) + '\n')
    cols = ['instance', 'axis', 'commanded_mm', 'executed_mm'] + [p + f for p in ('odd_', 'even_') for f in FIELDS]
    (OUT / 'oddeven.dat').write_text(' '.join(cols) + '\n' + ''.join(
        ' '.join(str(r[c]) if isinstance(r[c], str) else format(r[c], '.17g') for c in cols) + '\n' for r in rows))
    # Signed displacement along the goal ray (toward the goal positive), one row per on-ray candidate.
    cols = ['instance', 'signed_commanded_mm', 'signed_executed_mm', 'dC_real', 'dC_pred_cmd', 'dC_pred_exec']
    lines = []
    for i in result['instances']:
        for r in i['candidates']:
            if r['kind'] == 'onray':
                s = 1.0 if '-plus-' in r['name'] else -1.0
                lines.append(' '.join([i['instance']] + [format(v, '.17g') for v in (
                    s * r['commanded_mm'], s * r['executed_mm'], r['dC_real'], r['dC_pred_cmd'], r['dC_pred_exec'])]))
    (OUT / 'signed_response.dat').write_text(' '.join(cols) + '\n' + '\n'.join(lines) + '\n')
    # Plot tables for fig-signed-response.tex: evenly spaced command categories (not to scale).
    xpos = {-30.0: 1, -10.0: 2, -3.0: 3, -1.0: 4, 1.0: 5, 3.0: 6, 10.0: 7, 30.0: 8}
    points = []
    for i in result['instances']:
        for r in i['candidates']:
            if r['kind'] == 'onray':
                sign = 1.0 if '-plus-' in r['name'] else -1.0
                points.append((xpos[sign * round(r['commanded_mm'])], r['dC_real'], r['dC_pred_cmd']))
            elif r['kind'] == 'offray':
                points.append((9.7 if round(r['commanded_mm']) == 10 else 10.7, r['dC_real'], r['dC_pred_cmd']))
    (OUT / 'fig_points.dat').write_text('xpos dC_real dC_pred_cmd\n' + ''.join(
        f'{x:g} {a:.17g} {b:.17g}\n' for x, a, b in points))
    med = sorted({x for x, _, _ in points})
    (OUT / 'fig_medians.dat').write_text('xpos n dC_real dC_pred_cmd\n' + ''.join(
        f"{x:g} {sum(q[0] == x for q in points)} {np.median([q[1] for q in points if q[0] == x]):.17g} "
        f"{np.median([q[2] for q in points if q[0] == x]):.17g}\n" for x in med))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
