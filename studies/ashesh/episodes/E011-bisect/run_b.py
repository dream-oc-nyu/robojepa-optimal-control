#!/usr/bin/env python3
"""E011 Part B: d = final distance (m); p = fixed scorer progress; mu = mean p.

B1: d_b(107) <= .02 AND d_s(107) > .04 AND x-flips_s(107) >= 3.
B2: max_i(d_b(i)-d_s(i)) <= .01 AND max_far d_b(i) <= .02.
B3: #{p_b>0} = 8 AND mu_b >= .90 AND mu_b-mu_s >= .03
    AND mu_b > mu_cem_clampexec (read from receipts, not the rounded .888).
B4: sum guard_fired over logged replay substeps = 0, for BOTH b and s.
B5: median_b >= .95 AND CI95_low <= 0 <= CI95_high versus cem_release,
    using bench.score.summarize unchanged (no leaderboard writes).
C1: #{i: max_{t,axis}|commanded_b-commanded_h| < .0001} >= 5.
C2: d_b(107)-d_h(107) >= .01 AND d_h(107) <= .02.
C3: max_far(d_h-d_b) <= .01 AND mu_h >= mu_b.
Here b=bisect (revised halve-on-flip), s=sat10, h=halvesat; far={103,105,106}.
B1-B3 with h substituted for b are SECONDARY, not episode predictions.

Geometry: p_0=0, p_{t+1}=p_t+executed_t, ||g-p_t||=D_t, r_t=g-p_t.
Reuse Part A's pooled fit and E006 prior. A true bracket at t requires
sgn(e_{t-1,i})=sgn(r_{t-1,i}) and |e_{t-1,i}|>|r_{t-1,i}|.
Far flips have |r_{t,i}|>=.10. sat10 flips use Part A's base-command rule:
consecutive |base_i|>=.05-1e-5, opposite signs, ||e_{t-1}||>=.001.
Other arms use the sidecar's flip bit, never infer it from shrunken outputs.

All predicates need eight complete seed-0 units in every required arm; missing
predicate inputs imply pending, even if an observed clause already fails.
Only numpy and stdlib; all outputs are confined to --out. Example:
python3 -B episodes/E011-bisect/run_b.py --search gna --runs RUNS --out OUT
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from bench.score import summarize


@dataclass(frozen=True)
class Config:
    instances: tuple = tuple(range(100, 108))
    far_instances: tuple = (103, 105, 106)
    ticks: int = 5
    seed: int = 0
    near_m: float = .02
    beyond_m: float = .04
    distance_margin_m: float = .01
    mean_gain: float = .03
    mean_min: float = .90
    median_min: float = .95
    command_same_m: float = .0001
    far_axis_m: float = .10
    moved_m: float = .001
    saturation_m: float = .05
    saturation_tolerance_m: float = 1e-5
    min_x_flips: int = 3
    min_same_instances: int = 5
    positive_progress: float = 0.
    max_guard_substeps: int = 0
    interval_zero: float = 0.
    replay_fidelity_m: float = .001
    replay_start_m: float = 1e-6
    sidecar_match_m: float = 1e-7
    latex_timeout_s: int = 120
    reference_arms: tuple = ('cem_release', 'cem_cap10', 'cem_clampexec')
    rules: tuple = ('sat10', 'bisect', 'halvesat')


def import_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PART_A = import_path('e011_part_a_for_b', HERE / 'run.py')
REPLAY = import_path('e008_for_e011_part_b', HERE.parent / 'E008-frozen-arm/run.py')


def read_json(path, hashes):
    raw = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def valid_unit(u, cfg):
    if not u.get('complete') or len(u.get('ticks', [])) != cfg.ticks:
        return 'incomplete unit'
    if [t.get('tick') for t in u['ticks']] != list(range(cfg.ticks)):
        return 'ticks are not 0..4'
    try:
        a, e = PART_A.arrays(u)
        values = [u['initial_distance_m']]
        for t in u['ticks']:
            values.extend(t[k] for k in ('distance_m', 'planning_s', 'budget_overrun_s',
                                         'release_reward', 'release_success'))
            if t['rho'] is not None:
                values.append(t['rho'])
        if (a.shape != (cfg.ticks, 3) or e.shape != a.shape or
                not all(np.isfinite(v).all() for v in (a, e, values)) or
                u['initial_distance_m'] <= 0 or any(t['distance_m'] < 0 for t in u['ticks'])):
            return 'invalid unit numerics'
    except (KeyError, TypeError, ValueError) as exc:
        return f'missing/invalid unit input: {exc}'
    return None


def load_units(runs, names, cfg, hashes, missing):
    """Part A loader; pool ALL arms and seeds, score only requested seed-0 arms."""
    pool, excluded = [], []
    for directory in sorted(p for p in runs.glob('*') if p.is_dir()):
        for u in PART_A.load_arm(runs, directory.name, directory.name == '_calibration'):
            hashes[u['path']] = u['sha256']
            path = Path(u['path'])
            expected_dir = f"inst{u['instance']}-seed{u['seed']}"
            if path.parent.name != expected_dir or (not u['calibration'] and u['planner'] != directory.name):
                raise ValueError(f'Unit identity mismatch: {path}')
            if u['instance'] not in cfg.instances:
                excluded.append({'path': str(path), 'reason': 'outside dev instances'})
                continue
            reason = valid_unit(u, cfg)
            if reason:
                excluded.append({'path': str(path), 'reason': reason})
                missing.append(f'{path}: {reason}')
            else:
                pool.append(u)
    units = {a: {u['instance']: u for u in pool if u['group'] == a and
                 not u['calibration'] and u['seed'] == cfg.seed} for a in names}
    for a in names:
        absent = sorted(set(cfg.instances) - units[a].keys())
        if absent:
            missing.append(f'{a}: complete seed-0 units missing for {absent}')
    return units, pool, excluded


def length_logs(u, hashes, missing, cfg):
    path = Path(u['path']).with_name('planner_log.json')
    if not path.is_file():
        missing.append(f'{path}: sidecar not run/exported')
        return {}
    data = read_json(path, hashes)
    if tuple(data.get(k) for k in ('planner', 'instance', 'seed')) != tuple(u[k] for k in ('planner', 'instance', 'seed')):
        raise ValueError(f'Sidecar identity mismatch: {path}')
    logs = {}
    vectors = ('base_command', 'saturated', 'flip', 'length_before', 'length_after')
    for item in data.get('ticks', []):
        t = item.get('tick')
        if t not in range(cfg.ticks) or t in logs:
            raise ValueError(f'Invalid/duplicate sidecar tick: {path}')
        log = (item.get('log') or {}).get('length_rule')
        if not isinstance(log, dict) or any(k not in log for k in vectors):
            missing.append(f'{path}: length_rule fields missing at tick {t}')
            continue
        if any(np.shape(log[k]) != (3,) or not np.isfinite(log[k]).all() for k in vectors):
            raise ValueError(f'Invalid length_rule vector: {path} tick {t}')
        if any(type(v) is not bool for k in ('saturated', 'flip') for v in log[k]):
            raise ValueError(f'Expected boolean saturation/flip: {path}')
        # Out and feedback are audit cross-checks when present, not substituted data.
        for key, expected in (('out', u['ticks'][t]['commanded']),
                              ('executed_prev', None if t == 0 else u['ticks'][t-1]['executed'])):
            if key in log and expected is not None:
                if (np.shape(log[key]) != (3,) or not np.isfinite(log[key]).all() or
                        np.max(np.abs(np.asarray(log[key]) - expected)) > cfg.sidecar_match_m):
                    raise ValueError(f'Sidecar {key} disagrees with unit: {path} tick {t}')
        logs[t] = log
    absent = sorted(set(range(cfg.ticks)) - logs.keys())
    if absent:
        missing.append(f'{path}: usable length_rule ticks missing {absent}')
    return logs


def flip_at(u, logs, t, axis, rule, cfg):
    if t not in logs:
        return None
    if rule != 'sat10':
        return bool(logs[t]['flip'][axis])
    if t == 0:
        return False
    if t-1 not in logs:
        return None
    prev, now = logs[t-1]['base_command'][axis], logs[t]['base_command'][axis]
    cut = cfg.saturation_m - cfg.saturation_tolerance_m
    return bool(abs(prev) >= cut and abs(now) >= cut and np.sign(prev) != np.sign(now)
                and np.linalg.norm(u['ticks'][t-1]['executed']) >= cfg.moved_m)


def axis_rows(u, logs, g, rule, cfg):
    """Only unique-g instances enter per-axis diagnostics; absent log fields stay null."""
    a, e = PART_A.arrays(u)
    remaining = g - np.vstack([np.zeros(3), np.cumsum(e, axis=0)])[:-1]
    rows = []
    for t in range(cfg.ticks):
        log = logs.get(t, {})
        for i, axis in enumerate(('x', 'y', 'z')):
            flip = flip_at(u, logs, t, i, rule, cfg)
            bracket = None
            if flip and t:
                rp, ep = remaining[t-1, i], e[t-1, i]
                bracket = bool(np.sign(ep) == np.sign(rp) and abs(ep) > abs(rp))
            row = dict(arm=u['group'], instance=u['instance'], seed=u['seed'], tick=t, axis=axis,
                       commanded_m=float(a[t, i]), executed_m=float(e[t, i]),
                       remaining_m=float(remaining[t, i]), flip=flip,
                       flip_source='part-A base commands' if rule == 'sat10' else 'logged flip',
                       true_bracket=bracket,
                       far_axis_flip=None if flip is None else bool(flip and abs(remaining[t, i]) >= cfg.far_axis_m))
            for key in ('base_command', 'saturated', 'length_before', 'length_after'):
                row[key] = log[key][i] if key in log else None
            row['executed_prev_m'] = float(e[t-1, i]) if t else None
            row['remaining_prev_m'] = float(remaining[t-1, i]) if t else None
            rows.append(row)
    return rows


def replay_metrics(root, arm, units, hashes, missing, cfg):
    """Count E008 guard_fired on the five logged ticks, never diagnostic probes.

    Missing/partial or mismatched replay evidence cannot establish no guard.
    Provenance and command/fidelity checks prevent accepting a stale replay.
    """
    path = root / arm / 'replay.json'
    if not path.is_file():
        missing.append(f'{path}: replay not run')
        return dict(status='not run', guard_substeps=None, ticks=[], reasons=['replay absent'])
    data = read_json(path, hashes)
    reasons, rows = [], []
    if data.get('planner') != arm or data.get('seed') != cfg.seed:
        reasons.append('replay identity mismatch')
    if not data.get('complete'):
        reasons.append('replay marked incomplete')
    receipts = data.get('instances', [])
    ids = [r.get('instance') for r in receipts]
    if len(ids) != len(set(ids)) or set(ids) != set(cfg.instances):
        reasons.append('replay requires eight distinct dev instances')
    for rec in receipts:
        inst = rec.get('instance')
        u = units.get(inst)
        if u is None:
            reasons.append(f'{inst}: complete input unit absent')
            continue
        key = f'inst{inst}-seed{cfg.seed}'
        if (data.get('input_sha256', {}).get(key) != u['sha256'] or
                rec.get('input_unit_sha256') != u['sha256']):
            reasons.append(f'{inst}: replay/unit SHA256 mismatch or missing')
        if rec.get('seed') != cfg.seed or not rec.get('complete'):
            reasons.append(f'{inst}: incomplete replay instance or wrong seed')
        start = rec.get('initial_distance_m')
        if start is None or not np.isfinite(start) or abs(start-u['initial_distance_m']) > cfg.replay_start_m:
            reasons.append(f'{inst}: replay start differs or missing')
        ticks = rec.get('ticks', [])
        if [t.get('tick') for t in ticks] != list(range(cfg.ticks)):
            reasons.append(f'{inst}: replay ticks incomplete')
            continue
        for t, original in zip(ticks, u['ticks']):
            guards = t.get('substeps', {}).get('guard_fired')
            if (not isinstance(guards, list) or t.get('sim_steps') != len(guards) or
                    any(type(v) is not bool for v in guards)):
                reasons.append(f'{inst}/{t["tick"]}: guard flags missing/invalid')
                continue
            if not np.array_equal(np.asarray(t.get('commanded'), dtype=np.float32),
                                  np.asarray(original['commanded'], dtype=np.float32)):
                reasons.append(f'{inst}/{t["tick"]}: replay commands differ')
            e, distance = t.get('executed'), t.get('distance_m')
            if (np.shape(e) != (3,) or not np.isfinite(e).all() or distance is None or
                    not np.isfinite(distance) or abs(distance-original['distance_m']) > cfg.replay_fidelity_m or
                    np.linalg.norm(np.asarray(e)-original['executed']) > cfg.replay_fidelity_m):
                reasons.append(f'{inst}/{t["tick"]}: replay fidelity missing/outside tolerance')
            rows.append(dict(arm=arm, instance=inst, tick=t['tick'],
                             guard_substeps=int(REPLAY.count_guard(t)), substeps=len(guards)))
    if reasons:
        missing.append(f'{path}: ' + '; '.join(reasons))
    return dict(status='pending' if reasons else 'complete',
                guard_substeps=sum(r['guard_substeps'] for r in rows) if not reasons else None,
                observed_guard_substeps=sum(r['guard_substeps'] for r in rows), ticks=rows, reasons=reasons)


def judgment(arms, required, numbers, clauses, extra=()):
    missing = [f'{a} {arms[a]["units_present"]}/8' for a in required if arms[a]['status'] != 'complete']
    missing += list(extra)
    holds = None if missing or any(v is None for v in clauses.values()) else all(clauses.values())
    return dict(verdict='pending' if holds is None else 'holds' if holds else 'fails',
                holds=holds, numbers=numbers, clauses=clauses, missing=missing)


def b123(arms, units, logs, target, sat, cfg):
    row, control = arms[target], arms[sat]
    d = lambda a, i: arms[a]['per_instance'].get(i, {}).get('final_distance_cm')
    td, sd = d(target, 107), d(sat, 107)
    flips = None
    if 107 in units[sat]:
        bits = [flip_at(units[sat][107], logs.get((sat, 107), {}), t, 0, 'sat10', cfg)
                for t in range(1, cfg.ticks)]
        if all(b is not None for b in bits):
            flips = sum(b is True for b in bits)
    b1 = judgment(arms, (target, sat), dict(target_107_cm=td, sat10_107_cm=sd, sat10_x_flips=flips),
                  dict(target_within=td <= 100*cfg.near_m if td is not None else None,
                       sat10_beyond=sd > 100*cfg.beyond_m if sd is not None else None,
                       sat10_flips=flips >= cfg.min_x_flips if flips is not None else None),
                  ['sat10/107 base-command transitions missing'] if flips is None else [])
    deltas = {i: d(target, i)-d(sat, i) for i in cfg.instances
              if d(target, i) is not None and d(sat, i) is not None}
    far = {i: d(target, i) for i in cfg.far_instances}
    b2 = judgment(arms, (target, sat), dict(delta_cm=deltas, far_cm=far),
                  dict(no_regression=all(v <= 100*cfg.distance_margin_m for v in deltas.values()),
                       far_within=all(v is not None and v <= 100*cfg.near_m for v in far.values())))
    mean, other, clamp = row['mean_progress'], control['mean_progress'], arms['cem_clampexec']['mean_progress']
    gain = mean-other if mean is not None and other is not None else None
    b3 = judgment(arms, (target, sat, 'cem_clampexec'),
                  dict(positive=row['positive_progress_count'], mean=mean, gain_vs_sat10=gain, clamp_mean=clamp),
                  dict(all_positive=row['positive_progress_count'] == len(cfg.instances),
                       mean_min=mean >= cfg.mean_min if mean is not None else None,
                       gain_min=gain >= cfg.mean_gain if gain is not None else None,
                       above_clamp=mean > clamp if mean is not None and clamp is not None else None))
    return dict(B1=b1, B2=b2, B3=b3)


def predictions(arms, units, logs, selected, cfg):
    sat, bisect, half = (selected[r] for r in cfg.rules)
    out = b123(arms, units, logs, bisect, sat, cfg)
    replay = {a: arms[a]['replay'] for a in (sat, bisect)}
    out['B4'] = judgment(arms, (sat, bisect),
                         {a: dict(status=r['status'], guard_substeps=r['guard_substeps']) for a, r in replay.items()},
                         {a: r['guard_substeps'] == cfg.max_guard_substeps if r['guard_substeps'] is not None else None
                          for a, r in replay.items()},
                         [f'{a} replay {r["status"]}' for a, r in replay.items() if r['status'] != 'complete'])
    score = arms[bisect]['fixed_scorer']
    median = score.get('median_progress') if score else None
    paired = score.get('paired_vs_cem_release') if score else None
    interval = paired['ci95'] if paired else None
    out['B5'] = judgment(arms, (bisect, 'cem_release'), dict(median=median, paired=paired),
                         dict(median_min=median >= cfg.median_min if median is not None else None,
                              interval_contains_zero=interval[0] <= cfg.interval_zero <= interval[1] if interval else None))
    command_diffs = {i: float(np.max(np.abs(PART_A.arrays(units[bisect][i])[0] -
                                           PART_A.arrays(units[half][i])[0])))
                     for i in sorted(units[bisect].keys() & units[half].keys())}
    same = sum(v < cfg.command_same_m for v in command_diffs.values())
    out['C1'] = judgment(arms, (bisect, half), dict(same_instances=same, max_command_difference_m=command_diffs),
                         dict(enough_same=same >= cfg.min_same_instances))
    b = arms[bisect]['per_instance']
    h = arms[half]['per_instance']
    db = b.get(107, {}).get('final_distance_cm')
    dh = h.get(107, {}).get('final_distance_cm')
    improvement = db-dh if db is not None and dh is not None else None
    out['C2'] = judgment(arms, (bisect, half), dict(bisect_107_cm=db, halvesat_107_cm=dh, improvement_cm=improvement),
                         dict(closer=improvement >= 100*cfg.distance_margin_m if improvement is not None else None,
                              within=dh <= 100*cfg.near_m if dh is not None else None))
    far = {i: h[i]['final_distance_cm']-b[i]['final_distance_cm'] for i in cfg.far_instances if i in b and i in h}
    bm, hm = arms[bisect]['mean_progress'], arms[half]['mean_progress']
    out['C3'] = judgment(arms, (bisect, half), dict(far_delta_cm=far, bisect_mean=bm, halvesat_mean=hm),
                         dict(far_no_regression=all(v <= 100*cfg.distance_margin_m for v in far.values()),
                              mean_not_less=hm >= bm if hm is not None and bm is not None else None))
    return out, dict(label='secondary; B1-B3 clauses for halvesat, not episode predictions',
                     predictions=b123(arms, units, logs, half, sat, cfg))


def analyze(runs, replays, search, impl, cfg):
    selected = {r: f'{r}_{search}_{impl}' for r in cfg.rules}
    names = cfg.reference_arms + tuple(selected.values())
    hashes, missing = {}, []
    for path in (Path(__file__), HERE / 'episode.md', HERE / 'run.py', PART_A.PRIOR_PATH,
                 ROOT / 'bench/score.py', HERE.parent / 'E008-frozen-arm/run.py'):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    units, pool, excluded = load_units(runs, names, cfg, hashes, missing)
    scored = summarize([u for group in units.values() for u in group.values()])
    prior = PART_A.load_prior()
    geometry = [PART_A.fit_geometry(i, pool, [], prior, PART_A.Config())
                for i in cfg.instances if any(u['instance'] == i for u in pool)]
    goals = {g['instance']: np.array(g['g_cm']) / 100 for g in geometry if g['status'] == 'unique'}
    exclusions = [dict(instance=g['instance'], reason=f"goal offset {g['status']}: {g['ambiguity_kind']}")
                  for g in geometry if g['status'] != 'unique']
    exclusions += [dict(instance=i, reason='no geometry inputs') for i in cfg.instances
                   if not any(g['instance'] == i for g in geometry)]
    arms, logs, axes = {}, {}, []
    for a in names:
        score = scored.get(a)
        if units[a]:
            assert score is not None
        metrics = {i: dict(final_distance_cm=100*u['ticks'][-1]['distance_m'],
                           progress=scored[a]['per_instance'][i]) for i, u in units[a].items()}
        values = [m['progress'] for m in metrics.values()]
        n = len(metrics)
        arms[a] = dict(status='complete' if n == len(cfg.instances) else 'partial' if n else 'missing',
                       units_present=n, judgment_status='ready' if n == len(cfg.instances) else 'pending',
                       missing_instances=sorted(set(cfg.instances)-metrics.keys()), per_instance=metrics,
                       mean_progress=float(np.mean(values)) if values else None,
                       positive_progress_count=sum(p > cfg.positive_progress for p in values),
                       fixed_scorer=score)
        if a in selected.values():
            rule = next(r for r, name in selected.items() if name == a)
            for i, u in units[a].items():
                logs[a, i] = length_logs(u, hashes, missing, cfg)
                if i in goals:
                    axes.extend(axis_rows(u, logs[a, i], goals[i], rule, cfg))
            rows = [r for r in axes if r['arm'] == a]
            arms[a]['diagnostics'] = dict(
                status='complete' if n == len(cfg.instances) and all(
                    i in goals and len(logs[a, i]) == cfg.ticks for i in units[a]) else 'pending',
                scope='observed seed-0 flips on unique-g instances with sidecar data only',
                observed_flips=sum(r['flip'] is True for r in rows),
                true_brackets=sum(r['true_bracket'] is True for r in rows),
                false_flips=sum(r['true_bracket'] is False for r in rows),
                far_axis_false_flips=sum(r['true_bracket'] is False and r['far_axis_flip'] is True for r in rows),
                missing_axis_rows=sum(r['flip'] is None for r in rows),
                excluded_instances=sorted(set(units[a])-goals.keys()))
            arms[a]['replay'] = replay_metrics(replays, a, units[a], hashes, missing, cfg)
    pred, secondary = predictions(arms, units, logs, selected, cfg)
    return dict(search=search, impl=impl, config=asdict(cfg),
                inputs=dict(runs=str(runs), replays=str(replays), sha256=hashes),
                anchor_gate={a: dict(status=arms[a]['status'], units=arms[a]['units_present'],
                                    mean_progress=arms[a]['mean_progress'],
                                    fixed_scorer=arms[a]['fixed_scorer']) for a in cfg.reference_arms},
                definitions=dict(progress='bench.score.summarize per_instance (seed 0); no reimplementation',
                                 complete='8 complete seed-0 units; fixed scorer complete_24_units stays unchanged',
                                 guard='E008 count_guard on logged replay ticks only; probes excluded',
                                 moved='whole executed vector norm >= 1 mm (Part A), not axis displacement',
                                 same_commands='maximum absolute commanded component difference < 0.1 mm',
                                 geometry='all available complete arms/seeds and calibration under --runs',
                                 boundary='within/at least/no more than inclusive; beyond/above strictly greater'),
                arms=arms, geometry=geometry, axis_table=axes, predictions=pred, secondary=secondary,
                excluded_inputs=excluded, excluded_instances=exclusions, missing=missing)


def number(value):
    if value is None:
        return 'nan'
    if isinstance(value, bool):
        return str(int(value))
    return format(value, '.12g') if isinstance(value, (float, int)) else str(value)


def write_dat(path, columns, rows):
    path.write_text(' '.join(columns) + '\n' + ''.join(
        ' '.join(number(r.get(k)) for k in columns) + '\n' for r in rows))


def write_tables(result, out):
    arms = result['arms']
    for a, row in arms.items():
        write_dat(out / f'{a}.dat', ('instance', 'final_distance_cm', 'progress'),
                  [dict(instance=i, **m) for i, m in row['per_instance'].items()])
    summary = [dict(arm=a, status=r['status'], units=r['units_present'], mean_progress=r['mean_progress'],
                    positive=r['positive_progress_count'],
                    median_progress=(r['fixed_scorer'] or {}).get('median_progress'),
                    diagnostic_status=r.get('diagnostics', {}).get('status'),
                    **{k: v for k, v in r.get('diagnostics', {}).items() if k != 'status'})
               for a, r in arms.items()]
    for row in summary:
        paired = (arms[row['arm']]['fixed_scorer'] or {}).get('paired_vs_cem_release') or {}
        interval = paired.get('ci95', [None, None])
        row.update(delta_median=paired.get('delta_median'), ci95_low=interval[0], ci95_high=interval[1],
                   paired_instances=paired.get('paired_instances'))
    write_dat(out / 'arms.dat', ('arm', 'status', 'units', 'mean_progress', 'positive', 'median_progress',
                                 'delta_median', 'ci95_low', 'ci95_high', 'paired_instances', 'diagnostic_status',
                                 'observed_flips', 'true_brackets', 'false_flips', 'far_axis_false_flips'), summary)
    columns = ('arm', 'instance', 'seed', 'tick', 'axis', 'base_command', 'saturated', 'flip',
               'length_before', 'length_after', 'commanded_m', 'executed_m', 'remaining_m',
               'executed_prev_m', 'remaining_prev_m', 'true_bracket', 'far_axis_flip')
    write_dat(out / 'axis.dat', columns, result['axis_table'])
    write_dat(out / 'geometry.dat', ('instance', 'status', 'gx_cm', 'gy_cm', 'gz_cm', 'rms_residual_mm'),
              [dict(g, **dict(zip(('gx_cm', 'gy_cm', 'gz_cm'), g['g_cm']))) for g in result['geometry']])
    write_dat(out / 'replay.dat', ('arm', 'instance', 'tick', 'guard_substeps', 'substeps'),
              [t for a in arms.values() for t in a.get('replay', {}).get('ticks', [])])
    write_dat(out / 'judgments.dat', ('scope', 'prediction', 'verdict', 'holds'),
              [dict(scope=scope, prediction=k, verdict=p['verdict'], holds=p['holds'])
               for scope, predictions_ in (('primary', result['predictions']),
                                          ('secondary', result['secondary']['predictions']))
               for k, p in predictions_.items()])


def write_figure(result, out, cfg):
    names = ['cem_release', 'cem_cap10'] + [f'{r}_{result["search"]}_{result["impl"]}' for r in cfg.rules]
    tex = [r'\documentclass[tikz,border=8pt]{standalone}', r'\usepackage{pgfplots}',
           r'\pgfplotsset{compat=1.18}', r'\begin{document}', r'\begin{tikzpicture}',
           r'\begin{axis}[width=17cm,height=10cm,xlabel={Instance},ylabel={Final distance (cm)},',
           r'ymode=log,xtick={100,101,102,103,104,105,106,107},scaled x ticks=false,',
           r'unbounded coords=discard,filter discard warning=false,grid=major,',
           r'legend style={at={(0.5,1.03)},anchor=south},legend columns=3]']
    omitted = []
    for a, style in zip(names, ('black,mark=*', 'blue,mark=square*', 'orange,mark=triangle*',
                               'teal,mark=diamond*', 'red,mark=x')):
        tex += [r'\addplot[only marks,' + style + r'] table[x=instance,y=final_distance_cm] {' + a + '.dat};',
                r'\addlegendentry{' + a.replace('_', r'\_') + '}']
        omitted.extend(dict(arm=a, instance=i) for i, m in result['arms'][a]['per_instance'].items()
                       if m['final_distance_cm'] <= 0)
    tex += [r'\end{axis}', r'\end{tikzpicture}', r'\end{document}']
    (out / 'fig-partb.tex').write_text('\n'.join(tex) + '\n')
    pdf = out / 'fig-partb.pdf'
    pdf.unlink(missing_ok=True)
    metadata = dict(compiled=False, log_axis_nonpositive_units=omitted)
    compiler = shutil.which('pdflatex')
    if not compiler:
        return dict(metadata, reason='pdflatex not installed')
    try:
        proc = subprocess.run([compiler, '-interaction=nonstopmode', '-halt-on-error', 'fig-partb.tex'],
                              cwd=out, capture_output=True, text=True, timeout=cfg.latex_timeout_s)
        if proc.returncode or not pdf.is_file():
            pdf.unlink(missing_ok=True)
            return dict(metadata, reason='pdflatex failed', log_tail=(proc.stdout+proc.stderr)[-4000:])
    except (OSError, subprocess.TimeoutExpired) as exc:
        pdf.unlink(missing_ok=True)
        return dict(metadata, reason=str(exc))
    finally:
        for ext in ('aux', 'log'):
            (out / f'fig-partb.{ext}').unlink(missing_ok=True)
    return dict(compiled=True, log_axis_nonpositive_units=omitted, pdf_bytes=pdf.stat().st_size)


def print_summary(result, cfg):
    print(f'E011 Part B | search={result["search"]} impl={result["impl"]} | seed=0')
    for scope, preds in (('', result['predictions']), ('secondary ', result['secondary']['predictions'])):
        for name, p in preds.items():
            text = json.dumps(p['numbers'], separators=(',', ':'))
            print(f'{scope}{name}: {p["verdict"]} | {text}')
    print('Final distance cm / progress (pending = no complete unit):')
    print('instance ' + ' '.join(result['arms']))
    for i in cfg.instances:
        cells = []
        for row in result['arms'].values():
            m = row['per_instance'].get(i)
            cells.append(f'{m["final_distance_cm"]:.3f}/{m["progress"]:.4f}' if m else 'pending')
        print(str(i) + ' ' + ' '.join(cells))
    print('Axis exclusions: ' + (', '.join(str(r['instance']) for r in result['excluded_instances']) or 'none'))
    print('Figure: ' + ('compiled' if result['figure']['compiled'] else result['figure']['reason']))


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or '').split('\n')[0])
    parser.add_argument('--search', choices=('cem', 'gna', 'gnb'), required=True)
    parser.add_argument('--impl', choices=('a', 'b'), default='a')
    parser.add_argument('--runs', type=Path, default=ROOT / 'runs/E002-planner-loop')
    parser.add_argument('--replays', type=Path, default=ROOT / 'runs/E008-frozen-arm')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    out = (args.out or HERE / f'out-partB-{args.search}-{args.impl}').resolve()
    cfg = Config()
    result = analyze(args.runs.resolve(), args.replays.resolve(), args.search, args.impl, cfg)
    json.dumps(result, allow_nan=False)
    out.mkdir(parents=True, exist_ok=True)
    write_tables(result, out)
    result['figure'] = write_figure(result, out, cfg)
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print_summary(result, cfg)


if __name__ == '__main__':
    main()
