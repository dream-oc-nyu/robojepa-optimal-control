"""CPU/numpy scoring: median seed progress per instance, then median over instances.

The paired bootstrap resamples instances (not correlated repeated planner seeds).
Only atomically published unit.json files enter scoring. Partial sets stay labelled.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def paired_bootstrap(a, b, draws=10000):
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    x, y = np.array([a[k] for k in keys]), np.array([b[k] for k in keys])
    indices = np.random.default_rng(20261112).integers(0, len(keys), (draws, len(keys)))
    deltas = np.median(x[indices], axis=1) - np.median(y[indices], axis=1)
    return {"paired_instances": len(keys), "delta_median": float(np.median(x)-np.median(y)),
            "ci95": np.quantile(deltas, [.025, .975]).tolist()}


def summarize(units):
    result = {}
    for planner in sorted({u['planner'] for u in units}):
        group = [u for u in units if u['planner'] == planner]
        per_instance = {i: float(np.median([1-u['ticks'][-1]['distance_m']/u['initial_distance_m']
                        for u in group if u['instance'] == i])) for i in sorted({u['instance'] for u in group})}
        ticks = [t for u in group for t in u['ticks']]
        rho = [t['rho'] for t in ticks if t['rho'] is not None and np.isfinite(t['rho'])]
        complete = {(u['instance'], u['seed']) for u in group} == {(i,s) for i in range(100,108) for s in range(3)}
        result[planner] = {
            'units': len(group), 'complete_24_units': complete, 'per_instance': per_instance,
            'median_progress': float(np.median(list(per_instance.values()))),
            'mean_release_return': float(np.mean([sum(.99**i*t['release_reward'] for i,t in enumerate(u['ticks'])) for u in group])),
            'final_success_rate': float(np.mean([u['ticks'][-1]['release_success'] for u in group])),
            'rho_quantiles_05_50_95': np.quantile(rho,[.05,.5,.95]).tolist() if rho else None,
            'rho_count':len(rho), 'planning_s_median':float(np.median([t['planning_s'] for t in ticks])),
            'planning_s_total':float(sum(t['planning_s'] for t in ticks)),
            'overrun_ticks':sum(t['budget_overrun_s'] > 0 for t in ticks),
        }
    baseline = result.get('cem_release', {}).get('per_instance', {})
    for name, row in result.items():
        row['paired_vs_cem_release'] = paired_bootstrap(row['per_instance'], baseline) if name != 'cem_release' else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=Path('runs/E002-planner-loop'))
    parser.add_argument('--out', type=Path, default=Path('episodes/E002-planner-loop/out'))
    parser.add_argument('--leaderboard', type=Path, default=Path('bench/leaderboard.jsonl'))
    args = parser.parse_args()
    units, seen, hashes = [], set(), []
    for path in sorted(args.runs.rglob('unit.json')):
        raw = path.read_bytes(); u = json.loads(raw)
        if u.get('calibration',False): continue
        key = u['planner'], u['instance'], u['seed']
        if key in seen:
            raise ValueError(f'duplicate unit {key}: {path}')
        if not u.get('complete') or len(u['ticks']) != 5:
            raise ValueError(f'incomplete receipt {path}')
        seen.add(key); units.append(u); hashes.append(hashlib.sha256(raw).hexdigest())
    results = summarize(units)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'results.json').write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
    lines = ['# E002 reach-v0 results', '', 'Seed medians within instance; median across instances. Partial sets are not completed comparisons.', '',
             '| Planner | Units / 24 | Median progress | Discounted release return | Final success | Median planning s | Overrun ticks |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        lines.append(f"| {name} | {r['units']} | {r['median_progress']:.6f} | {r['mean_release_return']:.6f} | {r['final_success_rate']:.4f} | {r['planning_s_median']:.3f} | {r['overrun_ticks']} |")
    lines += ['', '| Instance | '+' | '.join(results)+' |', '|---|'+'---:|'*len(results)]
    for i in range(100,108):
        lines.append(f'| {i} | '+' | '.join(f"{r['per_instance'][i]:.6f}" if i in r['per_instance'] else 'pending' for r in results.values())+' |')
    for name,r in results.items():
        lines += ['', f"## {name}", f"rho 5/50/95%: {r['rho_quantiles_05_50_95']} (n={r['rho_count']}).",
                  f"Paired bootstrap versus CEM: `{r['paired_vs_cem_release']}`."]
        (args.out/f'{name}-progress.dat').write_text('instance progress\n'+''.join(f'{i} {p:.9g}\n' for i,p in r['per_instance'].items()))
        group_ticks = [t for u in units if u['planner']==name for t in u['ticks']]
        (args.out/f'{name}-ticks.dat').write_text('index planning_s rho distance_m\n'+''.join(
            f"{j} {t['planning_s']:.9g} {t['rho'] if t['rho'] is not None else 'nan'} {t['distance_m']:.9g}\n" for j,t in enumerate(group_ticks)))
    (args.out/'results.md').write_text('\n'.join(lines)+'\n')
    fingerprint = hashlib.sha256('\n'.join(sorted(hashes)).encode()).hexdigest()
    args.leaderboard.parent.mkdir(parents=True,exist_ok=True)
    prior = [json.loads(l) for l in args.leaderboard.read_text().splitlines()] if args.leaderboard.exists() else []
    with args.leaderboard.open('a') as f:
        for name,r in results.items():
            if not any(p.get('input_sha256')==fingerprint and p['planner']==name for p in prior):
                f.write(json.dumps({'benchmark':'reach-v0','planner':name,'input_sha256':fingerprint,'scored_unix':time.time(),**r},allow_nan=False)+'\n')
    print(json.dumps(results, indent=2))

if __name__ == '__main__': main()
