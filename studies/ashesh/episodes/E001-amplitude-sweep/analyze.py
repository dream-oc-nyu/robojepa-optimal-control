"""CPU-only fixed evaluator: python analyze.py INPUT_DIR --out OUTPUT_DIR.

Or: python analyze.py --e000 --out out-E000. NumPy + stdlib only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from readings import SCOPES, SELECTORS, adapt_e000, divide, evaluate, load_instance


@dataclass(frozen=True)
class Config:
    """Fixed bins [lo,hi) in executed mm; cluster-bootstrap settings."""
    gain_edges_mm: tuple = (0., 1., 3., 10., 30.)  # final bin [30,infinity)
    bootstrap_draws: int = 2000
    bootstrap_seed: int = 0


def load_e000(root):
    """Read retained pilot arrays, then use the pure E000 adapter."""
    physical = root / 'runs/20261029T230407Z-checkpoint/remote/physical'
    replay = root / 'runs/20261029T235530Z-action-contract/remote/replay'
    def read(path):
        with np.load(path, allow_pickle=False) as f:
            return {k: f[k] for k in f.files if k != 'history'}
    real = read(physical / 'encoded-observations.npz')
    endpoints = {n: read(physical / f'{n}-endpoint.npz') for n in real}
    trials = json.loads((physical / 'trials.json').read_text())
    return adapt_e000(real, read(replay / 'requested-endpoints.npz'),
                      read(replay / 'achieved_xyz-endpoints.npz'), endpoints,
                      {t['trial']: t['sim_seconds'] for t in trials})


def stats(values):
    """Median/min/max over defined values, retaining the effective sample count."""
    v = [x for x in values if x is not None]
    return dict(n=len(v), median=float(np.median(v)) if v else None,
                min=min(v) if v else None, max=max(v) if v else None)


def amplitude_summary(instances, key, fields, predicate) -> list[dict[str, Any]]:
    """Within each instance/bin take medians, then summarize across instances."""
    groups = {}
    for instance in instances:
        local = {}
        for r in instance['candidates']:
            if predicate(r):
                local.setdefault(key(r), []).append(r)
        for group, rows in local.items():
            entry = groups.setdefault(group, {'n_candidates': 0, 'n_instances': 0,
                                             **{f: [] for f in fields}})
            entry['n_candidates'] += len(rows)
            entry['n_instances'] += 1
            for f in fields:
                entry[f].append(stats([r[f] for r in rows])['median'])
    return [dict(group=k, n_candidates=v['n_candidates'], n_instances=v['n_instances'],
                 **{f: stats(v[f]) for f in fields}) for k, v in sorted(groups.items())]


def pool(instances, config=Config()) -> dict[str, Any]:
    """Pool pair counts; bootstrap entire instances, not correlated pairs.

    Median/min/max describe instance-level metrics. Ratio/gain summaries first
    take within-instance medians (both signs/directions pooled), then summarize
    those medians. Goal and neutrals do not enter amplitude summaries.
    """
    assert instances, 'no instances'
    rng = np.random.default_rng(config.bootstrap_seed)
    sampled = (rng.integers(0, len(instances), (config.bootstrap_draws, len(instances)))
               if len(instances) >= 4 else None)
    rankings = {}
    for scope in SCOPES:
        rankings[scope] = {}
        for selector in SELECTORS:
            rows = [r['rankings'][scope][selector] for r in instances]
            con: dict[str, Any] = {k: sum(r['concordance'][k] for r in rows) for k in
                   ('concordant', 'discordant', 'total', 'cost_ties', 'distance_ties', 'both_ties')}
            con['fraction'] = divide(con['concordant'], con['total'])
            con['bootstrap_ci95'] = None
            con['bootstrap_defined_draws'] = 0
            if sampled is not None:
                numerator = np.array([r['concordance']['concordant'] for r in rows])[sampled].sum(axis=1)
                denominator = np.array([r['concordance']['total'] for r in rows])[sampled].sum(axis=1)
                valid = denominator > 0
                con['bootstrap_defined_draws'] = int(valid.sum())
                if valid.any():
                    con['bootstrap_ci95'] = np.quantile(numerator[valid] / denominator[valid], [.025, .975]).tolist()
            rankings[scope][selector] = dict(concordance=con,
                concordance_fraction=stats([r['concordance']['fraction'] for r in rows]),
                progress_captured=stats([r['progress_captured'] for r in rows]),
                regret_mm=stats([r['regret_mm'] for r in rows]),
                selected=[{'instance': i['instance'], 'name': r['selected']} for i, r in zip(instances, rows)])
    moving = lambda r: r['kind'] in ('onray', 'offray')
    ratios = amplitude_summary(instances, lambda r: round(r['commanded_mm'], 6),
                               ['ratio', 'executed_mm'], moving)
    gains = amplitude_summary(instances, lambda r: int(np.searchsorted(config.gain_edges_mm, r['executed_mm'], side='right') - 1),
                              ['G_cmd', 'G_exec', 'executed_mm'], moving)
    for r in ratios:
        r['commanded_mm'] = r.pop('group')
    for r in gains:
        b = r.pop('group')
        r['bin_lo_mm'] = config.gain_edges_mm[b]
        r['bin_hi_mm'] = config.gain_edges_mm[b+1] if b+1 < len(config.gain_edges_mm) else None
    return dict(n_instances=len(instances), ratio_by_amplitude=ratios, gain_by_executed_bin=gains,
                rankings=rankings, regret_split={k: stats([r['regret_split'][k] for r in instances])
                    for k in ('predictor_interface_mm', 'representation_cost_mm', 'total_mm')})


def fmt(x):
    return 'null' if x is None else f'{x:.6g}'


def spread(s):
    return f"{fmt(s['median'])} [{fmt(s['min'])}, {fmt(s['max'])}]"


def markdown(result):
    """One compact table per fixed question; no per-candidate table duplication."""
    p = result['pooled']
    lines = ['# Fixed evaluator results', '', f"Instances: {p['n_instances']}. Values are median [min, max] over instances; amplitude rows first take each instance's median.",
             'Goal and neutrals are excluded from amplitude summaries. Amplitudes are unsigned norms in mm.', '',
             '## Interface: executed / commanded', '',
             '| Commanded mm | Instances | Candidates | Executed mm | Ratio |',
             '|---|---|---|---|---|']
    for r in p['ratio_by_amplitude']:
        lines.append(f"| {fmt(r['commanded_mm'])} | {r['n_instances']} | {r['n_candidates']} | {spread(r['executed_mm'])} | {spread(r['ratio'])} |")
    lines += ['', '## Gain: dC_pred / dC_real', '',
              '| Executed bin mm | Candidates | Cmd valid instances | G cmd | Exec valid instances | G exec |',
              '|---|---|---|---|---|---|']
    for r in p['gain_by_executed_bin']:
        hi = 'inf' if r['bin_hi_mm'] is None else fmt(r['bin_hi_mm'])
        lines.append(f"| [{fmt(r['bin_lo_mm'])}, {hi}) | {r['n_candidates']} | {r['G_cmd']['n']} | {spread(r['G_cmd'])} | {r['G_exec']['n']} | {spread(r['G_exec'])} |")
    lines += ['', '## Ranking and selection', '',
              '| Set | Selector | Ordered / pairs | Ties (cost / D / both) | Fraction [bootstrap 95% CI] | Progress captured | Regret mm | Selected |',
              '|---|---|---|---|---|---|---|---|']
    for scope in SCOPES:
        for selector, r in p['rankings'][scope].items():
            c = r['concordance']
            ci = c['bootstrap_ci95']
            ci_text = 'null' if ci is None else ', '.join(map(fmt, ci))
            picks = ', '.join(f"{v['instance']}:{v['name']}" for v in r['selected'] if v['name'] is not None) or 'null'
            lines.append(f"| {scope} | {selector} | {c['concordant']} / {c['total']} | {c['cost_ties']} / {c['distance_ties']} / {c['both_ties']} | {fmt(c['fraction'])} [{ci_text}] | {spread(r['progress_captured'])} | {spread(r['regret_mm'])} | {picks} |")
    split = p['regret_split']
    lines += ['', f"Regret split (mm): predictor+interface {spread(split['predictor_interface_mm'])}; representation+cost {spread(split['representation_cost_mm'])}; total {spread(split['total_mm'])}.", '',
              'Concordance = strictly concordant / all pairs; exact ties are separate, not successes. All-set includes the first neutral; ray subsets contain only their named kind. Argmin ties use input order. Subset progress uses the first-neutral distance and that subset\'s best D.',
              'Gain is null for zero dC_real or |dC_real| < 10 times neutral-repeat noise. Ratio/cosine/progress with zero denominators are null. JSON preserves null; plot tables use nan.',
              'Bootstrap: 2,000 resamples of whole instances, seed 0, percentile 95% interval; unavailable below four instances. Bins are [0,1), [1,3), [3,10), [10,30), [30,inf) mm; commanded amplitudes group at 1e-6 mm precision.',
              result['source_note'], '']
    return '\n'.join(lines)


def write_dat(path, columns, rows):
    """Whitespace pgfplots tables; nan denotes an undefined numeric quantity."""
    def token(v):
        if v is None:
            return 'nan'
        if isinstance(v, float):
            return format(v, '.17g')
        text = str(v)
        assert text and not any(c.isspace() for c in text), 'plot identifier contains whitespace'
        return text
    path.write_text(' '.join(columns) + '\n' + ''.join(' '.join(token(v) for v in row) + '\n' for row in rows))


def write_outputs(result, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (out / 'results.md').write_text(markdown(result))
    candidates = [dict(instance=i['instance'], **r) for i in result['instances'] for r in i['candidates']]
    columns = 'instance name kind commanded_mm executed_mm D_mm dC_real dC_pred_cmd dC_pred_exec'.split()
    write_dat(out / 'percandidate.dat', columns, [[r[k] for k in columns] for r in candidates])
    columns = 'instance name kind commanded_mm executed_mm ratio'.split()
    write_dat(out / 'ratio_vs_amplitude.dat', columns, [[r[k] for k in columns] for r in candidates])
    columns = 'instance name kind commanded_mm executed_mm G_cmd G_exec'.split()
    write_dat(out / 'gain_vs_amplitude.dat', columns, [[r[k] for k in columns] for r in candidates])
    columns = 'instance scope selector selected concordant total cost_ties distance_ties both_ties fraction progress_captured regret_mm'.split()
    rows = []
    for i in result['instances']:
        for scope in SCOPES:
            for selector, r in i['rankings'][scope].items():
                data = dict(instance=i['instance'], scope=scope, selector=selector, **r, **r['concordance'])
                rows.append([data[k] for k in columns])
    write_dat(out / 'selectors.dat', columns, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path)
    parser.add_argument('--e000', action='store_true')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.e000 == (args.directory is not None):
        parser.error('provide exactly one of directory or --e000')
    if args.e000:
        instances = [evaluate(load_e000(Path(__file__).resolve().parents[2]), 'E000')]
        metadata = None
        note = 'E000 is retrospective replay, not sealed E001 data. Its missing rewritten telemetry has a false adapter placeholder (unknown, not verified); no sealing claim is made.'
    else:
        paths = sorted(args.directory.glob('instance-*.npz'))
        if not paths:
            parser.error('no instance-*.npz files')
        metadata = json.loads((args.directory / 'meta.json').read_text())
        assert isinstance(metadata, dict), 'meta.json must be an object'
        instances = [evaluate(load_instance(path), path.stem) for path in paths]
        note = 'E001 raw arrays scored as supplied; meta.json is retained, but model provenance and prediction sealing are not independently verified by this evaluator.'
    config = Config()
    result = dict(schema_version=1, config=asdict(config), source_note=note, metadata=metadata,
                  instances=instances, pooled=pool(instances, config))
    write_outputs(result, args.out)
    print(f"Scored {len(instances)} instance(s); wrote results.json, results.md and four .dat files to {args.out}")


if __name__ == '__main__':
    main()
