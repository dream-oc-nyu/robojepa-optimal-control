"""Fixed E001 mathematics. No inference; only load_instance performs I/O."""
from __future__ import annotations

import numpy as np
from itertools import combinations
from typing import Any

SELECTORS = {'realized': 'real', 'predicted-cmd': 'pred_cmd', 'predicted-exec': 'pred_exec'}
SCOPES = ('all', 'onray', 'offray')


def divide(a, b):
    """a/b, or None for an exactly zero denominator (never a fabricated zero)."""
    return None if b == 0 else float(a / b)


def validate_instance(x):
    """Assert E001 array types/shapes; feature grids may be tiny for fixtures.

    Require exactly two neutrals and one goal, unique whitespace-free names,
    finite values, and matching (N,2,H,W,C) float32 features. H,W,C are not
    fixed to the released encoder, so the same mathematics tests tiny arrays.
    """
    required = {'names', 'kind', 'commanded_xyz', 'executed_delta14', 'ee_world',
                'sim_seconds', 'rewritten', 'z_real', 'z_pred_cmd', 'z_pred_exec', 'images'}
    assert required <= x.keys(), f'missing fields: {required - x.keys()}'
    names, kind = x['names'], x['kind']
    assert names.ndim == 1 and names.dtype.kind == 'U', 'names must be Unicode (N,)'
    n = len(names)
    assert len(set(names)) == n and all(s and not any(c.isspace() for c in s) for s in names), 'invalid names'
    assert kind.shape == (n,) and kind.dtype.kind == 'U', 'kind must be Unicode (N,)'
    assert set(kind) <= {'neutral', 'onray', 'offray', 'goal'}, 'unknown kind'
    assert np.sum(kind == 'neutral') == 2 and np.sum(kind == 'goal') == 1, 'need two neutrals and one goal'
    for key, shape in [('commanded_xyz', (n, 3)), ('executed_delta14', (n, 14)),
                       ('ee_world', (n, 3)), ('sim_seconds', (n,))]:
        assert x[key].shape == shape and x[key].dtype == np.float64, f'{key}: require f64 {shape}'
        assert np.isfinite(x[key]).all(), f'{key}: nonfinite values'
    assert (x['sim_seconds'] >= 0).all(), 'negative sim_seconds'
    assert (x['commanded_xyz'][kind == 'neutral'] == 0).all(), 'neutral commands must be zero'
    assert x['rewritten'].shape == (n,) and x['rewritten'].dtype == bool, 'invalid rewritten'
    assert x['images'].shape == (n, 2, 240, 320, 3) and x['images'].dtype == np.uint8, 'invalid images'
    shape = x['z_real'].shape
    assert len(shape) == 5 and shape[:2] == (n, 2) and all(s > 0 for s in shape), 'invalid feature shape'
    for key in ('z_real', 'z_pred_cmd', 'z_pred_exec'):
        assert x[key].shape == shape and x[key].dtype == np.float32, f'{key}: mismatched shape or dtype'
        assert np.isfinite(x[key]).all(), f'{key}: nonfinite features'


def load_instance(path):
    """Load exactly one pickle-free E001 npz; reject malformed arrays."""
    with np.load(path, allow_pickle=False) as f:
        x = {k: f[k] for k in f.files}
    validate_instance(x)
    return x


def concordance(costs, distances):
    """Count strict same-sign pair differences / all unordered pairs.

    Ties are exact float64 equality, excluded from concordant/discordant but
    retained in total. cost_ties/distance_ties are exclusive; both_ties separate.
    """
    a, b = np.asarray(costs, np.float64), np.asarray(distances, np.float64)
    assert a.ndim == 1 and a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    counts = dict(concordant=0, discordant=0, total=0, cost_ties=0, distance_ties=0, both_ties=0)
    for i, j in combinations(range(len(a)), 2):
        sa, sb = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
        key = ('both_ties' if sa == sb == 0 else 'cost_ties' if sa == 0 else
               'distance_ties' if sb == 0 else 'concordant' if sa == sb else 'discordant')
        counts[key] += 1
        counts['total'] += 1
    return {**counts, 'fraction': divide(counts['concordant'], counts['total'])}


def selector(rows, field, neutral_distance):
    """argmin C (first input row breaks ties); progress relative to subset best D."""
    con = concordance([r['C_' + field] for r in rows], [r['D_mm'] for r in rows])
    if not rows:
        return dict(concordance=con, selected=None, selected_ties=[], progress_captured=None, regret_mm=None)
    pick = min(rows, key=lambda r: r['C_' + field])
    best = min(r['D_mm'] for r in rows)
    return dict(concordance=con, selected=pick['name'],
                selected_ties=[r['name'] for r in rows if r['C_' + field] == pick['C_' + field]],
                progress_captured=divide(neutral_distance - pick['D_mm'], neutral_distance - best),
                regret_mm=pick['D_mm'] - best)


def evaluate(x, name='instance') -> dict[str, Any]:
    """Compute C, own-neutral dC, response norms/cosines, G and physical regret.

    G = dC_pred/dC_real is null at zero or |dC_real| < 10*repeat noise.
    All distances/amplitudes are unsigned mm. Rankings exclude goal/repeat;
    onray/offray subsets contain strictly that kind (no neutral).
    """
    validate_instance(x)
    neutral, repeat = np.flatnonzero(x['kind'] == 'neutral')
    goal = int(np.flatnonzero(x['kind'] == 'goal')[0])
    g = x['z_real'][goal]
    costs = {p: [cost(z, g) for z in x['z_' + p]] for p in SELECTORS.values()}
    noise = abs(costs['real'][repeat] - costs['real'][neutral])
    rows = []
    for i, candidate in enumerate(x['names']):
        cmd = float(np.linalg.norm(x['commanded_xyz'][i]) * 1000)
        exe = float(np.linalg.norm(x['ee_world'][i] - x['ee_world'][neutral]) * 1000)
        row: dict[str, Any] = dict(name=str(candidate), kind=str(x['kind'][i]), commanded_mm=cmd, executed_mm=exe,
                   D_mm=float(np.linalg.norm(x['ee_world'][i] - x['ee_world'][goal]) * 1000),
                   ratio=None if x['kind'][i] == 'neutral' else divide(exe, cmd))
        responses = {}
        for p in SELECTORS.values():
            row['C_' + p] = costs[p][i]
            row['dC_' + p] = costs[p][i] - costs[p][neutral]
            responses[p] = (x['z_' + p][i].astype(np.float64) - x['z_' + p][neutral].astype(np.float64)).ravel()
            row['feature_response_' + p] = float(np.abs(responses[p]).mean())
        for variant in ('cmd', 'exec'):
            a, b = responses['pred_' + variant], responses['real']
            row['cos_' + variant + '_real'] = divide(float(a @ b), float(np.linalg.norm(a) * np.linalg.norm(b)))
            row['G_' + variant] = (None if abs(row['dC_real']) < 10 * noise else
                                    divide(row['dC_pred_' + variant], row['dC_real']))
        rows.append(row)
    eligible = [r for i, r in enumerate(rows) if i not in (repeat, goal)]
    rankings = {scope: {s: selector(eligible if scope == 'all' else [r for r in eligible if r['kind'] == scope],
                                  p, rows[neutral]['D_mm']) for s, p in SELECTORS.items()} for scope in SCOPES}
    uf = min(eligible, key=lambda r: r['C_pred_cmd'])
    ue = min(eligible, key=lambda r: r['C_real'])
    us = min(eligible, key=lambda r: r['D_mm'])
    return dict(instance=name, neutral=str(x['names'][neutral]), goal=str(x['names'][goal]),
                feature_shape=list(g.shape), neutral_repeat_noise=noise, gain_noise_threshold=10*noise,
                candidates=rows, rankings=rankings,
                regret_split=dict(U_F=uf['name'], U_E=ue['name'], U_star=us['name'],
                                  predictor_interface_mm=uf['D_mm'] - ue['D_mm'],
                                  representation_cost_mm=ue['D_mm'] - us['D_mm'],
                                  total_mm=uf['D_mm'] - us['D_mm']))


def adapt_e000(real, pred_cmd, pred_exec, endpoints, sim_seconds):
    """Build E001 arrays in memory from E000 named arrays, without I/O.

    Historical replay has no sealing/rewritten telemetry: the boolean placeholder
    is false, explicitly UNKNOWN, never evidence of sealed predictions. Scoring
    ignores rewritten. Original images, poses, commands and deltas are retained.
    """
    names = ['base0', 'base1', 'minus0.01', 'minus0.003', 'minus0.001',
             'plus0.001', 'plus0.003', 'plus0.01', 'goal']
    x = dict(names=np.array(names), kind=np.array(['neutral', 'neutral'] + ['onray']*6 + ['goal']),
             commanded_xyz=np.stack([endpoints[n]['commanded_total_xyz'] for n in names]),
             executed_delta14=np.stack([endpoints[n]['achieved_delta'] for n in names]),
             ee_world=np.stack([endpoints[n]['world_pose'][:3] for n in names]),
             images=np.stack([endpoints[n]['images'] for n in names]),
             sim_seconds=np.array([sim_seconds[n] for n in names], np.float64),
             rewritten=np.zeros(len(names), bool))
    for key, data in [('z_real', real), ('z_pred_cmd', pred_cmd), ('z_pred_exec', pred_exec)]:
        x[key] = np.stack([data[n] for n in names])
    validate_instance(x)
    return x


def cost(z, goal):
    """C(z) = mean |float64(z) - float64(goal)| over every feature entry."""
    z, goal = np.asarray(z, dtype=np.float64), np.asarray(goal, dtype=np.float64)
    assert z.shape == goal.shape and z.size, 'feature/goal shape mismatch'
    assert np.isfinite(z).all() and np.isfinite(goal).all(), 'nonfinite features'
    return float(np.abs(z - goal).mean())
