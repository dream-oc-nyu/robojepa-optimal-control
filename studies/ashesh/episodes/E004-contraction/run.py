"""E004: restore predicted token dispersion, then score with the fixed E001 evaluator.

Run from the repository root: python3 episodes/E004-contraction/run.py
Only one complete instance is resident; retained foreign goals are small copies.
All reductions and costs use float64. E003 supplies the unchanged scoring path.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True  # imported read-only episodes must not acquire pyc files
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'episodes' / 'E001-amplitude-sweep'))
from readings import cost, concordance, selector, load_instance, divide  # fixed evaluator

_spec = importlib.util.spec_from_file_location('e003_helpers', ROOT / 'episodes/E003-goal-free-term/run.py')
e003 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = e003
_spec.loader.exec_module(e003)
VARIANTS = ('raw', 'real', 'foreign', 'disp_channel', 'disp_scalar', 'cosine', 'amp_penalty', 'norm_only')


@dataclass(frozen=True)
class Config:
    """Fixed populations, numerical conventions and pre-registered cutoffs."""
    data_dir: str = 'runs/E001-amplitude-sweep/scoring-input'
    out_dir: str = 'episodes/E004-contraction/out'
    n_instances: int = 8
    min_command_mm: float = 9.99
    moving_reference: str = 'onray-plus-0.001'
    bootstrap_draws: int = 2000
    bootstrap_seed: int = 0
    lambda_max: float = 3.00
    lambda_step: float = 0.01
    gain_noise_multiple: float = 10.0
    arm_tokens: int = 30
    p1_foreign_median_min: float = -0.0066
    p1_negative_fraction_max: float = 0.75
    p2_toward: int = 16
    p2_away_min: int = 12
    p2_lateral_min: int = 32
    p2_gain_range: tuple = (0.15, 0.35)
    p3_fraction_min: float = 0.78


def tokens(z):
    """(view, grid_y, grid_x, channel) -> float64 (view, token, channel)."""
    return np.asarray(z, np.float64).reshape(z.shape[0], -1, z.shape[-1])


def match_dispersion(z, reference_sigma, scalar=False):
    """mu(a) + (z(a)-mu(a))*sigma(0)/sigma(a), population token std."""
    mu = z.mean(axis=1, keepdims=True)
    sigma = z.std(axis=1)
    num, den = reference_sigma, sigma
    if scalar:
        num, den = num.mean(axis=1, keepdims=True), den.mean(axis=1, keepdims=True)
    factor = np.divide(num, den, out=np.ones_like(den), where=den != 0)
    return mu + (z - mu) * factor[:, None, :]


def unit_tokens(z):
    """z_t / ||z_t||_2; zero-length tokens are undefined and rejected."""
    norm = np.linalg.norm(z, axis=-1, keepdims=True)
    assert (norm > 0).all(), 'undefined cosine for zero-norm token'
    return z / norm


def cosine_cost(z, goal_unit):
    """1 - mean_t cos(z_t, g_t); no centering or clipping."""
    return float(1 - (unit_tokens(z) * goal_unit).sum(axis=-1).mean())


def load_reduced(path, goals, goal_units, cfg):
    """Reduce one archive to small cost matrices and descriptive statistics."""
    x = load_instance(path)
    names, kinds = x['names'].tolist(), x['kind'].tolist()
    inst = dict(names=names, kinds=kinds,
                classes=[e003.candidate_class(n, k) for n, k in zip(names, kinds)],
                commanded_mm=(np.linalg.norm(x['commanded_xyz'], axis=1) * 1000).tolist(),
                norm_m=np.linalg.norm(x['commanded_xyz'], axis=1), ee=x['ee_world'].copy())
    idx = e003.index_sets(inst, cfg)
    n, ng = len(names), len(goals)
    keys = ('raw', 'real', 'disp_channel', 'disp_scalar', 'cosine', 'exec_raw', 'exec_disp_channel', 'exec_disp_scalar')
    inst['K'] = {k: np.empty((n, ng)) for k in keys}
    inst['K_view'] = np.empty((n, ng, 2))
    inst['real_cosine'] = np.empty(n)
    inst['dispersion'] = {f: np.empty((n, 2)) for f in ('pred_cmd', 'real')}
    inst['region_ratio'] = {f: {s: [] for s in ('arm', 'background')} for f in ('pred_cmd', 'real')}
    own_i = int(path.stem.split('-')[-1])
    delta = np.abs(tokens(x['z_real'][idx['goal']]) - tokens(x['z_real'][idx['neutral']])).mean(axis=-1)
    order = np.argsort(delta, axis=1, kind='stable')
    masks = {'arm': order[:, -cfg.arm_tokens:], 'background': order[:, :-cfg.arm_tokens]}
    inst['arm_token_indices'] = masks['arm'].tolist()
    for field in ('pred_cmd', 'real', 'pred_exec'):
        z0 = tokens(x['z_' + field][idx['neutral']])
        sigma0 = z0.std(axis=1)
        if field != 'pred_exec':
            ref_disp = {s: [z0[v, m[v]].std(axis=0).mean() for v in range(2)] for s, m in masks.items()}
        for a in range(n):
            z = tokens(x['z_' + field][a])
            key = {'pred_cmd': 'raw', 'real': 'real', 'pred_exec': 'exec_raw'}[field]
            inst['K'][key][a] = [cost(z, g) for g in goals]
            if field != 'pred_exec':
                inst['dispersion'][field][a] = z.std(axis=1).mean(axis=1)
                if a in idx['big'] and round(inst['commanded_mm'][a]) == 30:
                    for s, m in masks.items():
                        for v in range(2):
                            assert ref_disp[s][v] > 0
                            inst['region_ratio'][field][s].append(float(z[v, m[v]].std(axis=0).mean() / ref_disp[s][v]))
            if field == 'real':
                inst['real_cosine'][a] = cosine_cost(z, goal_units[own_i])
                continue
            prefix = 'exec_' if field == 'pred_exec' else ''
            for variant in ('disp_channel', 'disp_scalar'):
                matched = match_dispersion(z, sigma0, scalar=variant == 'disp_scalar')
                inst['K'][prefix + variant][a] = [cost(matched, g) for g in goals]
                del matched
            if field == 'pred_cmd':
                inst['K']['cosine'][a] = [cosine_cost(z, g) for g in goal_units]
                inst['K_view'][a] = [[cost(z[v], g[v]) for v in range(2)] for g in goals]
    # E003 helpers operate on these aliases, not on a different implementation.
    inst['K_pred_cmd'], inst['K_real'] = inst['K']['raw'], inst['K']['real']
    del x
    return inst


def foreign_responses(insts, idxs, key):
    """K_i(a,j)-K_i(0,j), retaining only the seven foreign columns."""
    return [np.delete(inst['K'][key] - inst['K'][key][idx['neutral']], i, axis=1)
            for i, (inst, idx) in enumerate(zip(insts, idxs))]


def foreign_summary(insts, idxs, responses):
    """672 responses, split into 336 pairs at each commanded amplitude."""
    values = {10: [], 30: []}
    for inst, idx, F in zip(insts, idxs, responses):
        for a in idx['big']:
            values[round(inst['commanded_mm'][a])].extend(F[a].tolist())
    allv = values[10] + values[30]
    assert len(allv) == 672 and len(values[10]) == len(values[30]) == 336
    return dict(foreign_n=len(allv), foreign_negative=int(np.sum(np.array(allv) < 0)),
                foreign_median_1cm=float(np.median(values[10])), foreign_median_3cm=float(np.median(values[30])))


def gain_details(insts, idxs, costs, cfg, cosine=False):
    """Median_i odd_variant/odd_real; reject exactly zero or <10*repeat noise."""
    out = {}
    for cm in (1, 3):
        details = []
        for i, (inst, idx, C) in enumerate(zip(insts, idxs, costs)):
            R = inst['real_cosine'] if cosine else inst['K']['real'][:, i]
            plus = inst['names'].index(f'onray-plus-{cm / 100:g}')
            minus = inst['names'].index(f'onray-minus-{cm / 100:g}')
            odd, odd_real = float((C[plus] - C[minus]) / 2), float((R[plus] - R[minus]) / 2)
            noise = float(abs(R[idx['repeat']] - R[idx['neutral']]))
            gain = None if odd_real == 0 or abs(odd_real) < cfg.gain_noise_multiple * noise else divide(odd, odd_real)
            details.append(dict(instance=i, odd=odd, odd_real=odd_real, noise=noise, gain=gain))
        valid = [r['gain'] for r in details if r['gain'] is not None]
        out[f'gain_{cm}cm'] = float(np.median(valid)) if valid else None
        out[f'gain_n_{cm}cm'] = len(valid)
        out[f'gain_per_instance_{cm}cm'] = [r['gain'] for r in details]
        out[f'gain_details_{cm}cm'] = details
    return out


def metrics(insts, idxs, distances, costs, responses, cfg, raw_median, cosine=False):
    """Fixed E003 sign/ranking path plus the registered foreign and odd responses."""
    scored = e003.score(insts, idxs, distances, costs, cfg)
    census, rank = scored['sign_census_ge_1cm'], scored['rankings']['all']
    m = foreign_summary(insts, idxs, responses)
    m['removed_3cm'] = 1 - m['foreign_median_3cm'] / raw_median
    m.update(toward_falls=census['toward']['falls'], away_rises=census['away']['rises'], lateral_rises=census['lateral']['rises'])
    m.update({k: rank[k] for k in ('concordant', 'total', 'fraction', 'n_regret0', 'regret_mm', 'selected')})
    m['ci95'] = rank['bootstrap_ci95']
    gains = gain_details(insts, idxs, costs, cfg, cosine)
    m.update({k: v for k, v in gains.items() if not k.startswith('gain_details')})
    detail = dict(summary=m, sign_census=scored['sign_census_ge_1cm'], rankings=scored['rankings'],
                  gain_details={k: v for k, v in gains.items() if k.startswith('gain_details')}, per_instance=[])
    for i, (inst, idx, C, F, D) in enumerate(zip(insts, idxs, costs, responses, distances)):
        detail['per_instance'].append(dict(instance=i, costs=C.tolist(), dC=(C-C[idx['neutral']]).tolist(),
            foreign_goals=[j for j in range(len(insts)) if j != i], foreign_response=F.tolist(),
            sign_census=e003.sign_census(C-C[idx['neutral']], idx), ranking=e003.rank_instance(inst, idx, C, D)['all']))
    assert m['total'] == 1088
    return m, detail


def amplitude_folds(insts, idxs, distances, cfg):
    """Choose smallest maximizer on seven instances; evaluate untouched eighth."""
    grid = np.arange(round(cfg.lambda_max / cfg.lambda_step) + 1, dtype=np.float64) * cfg.lambda_step
    counts = np.empty((len(insts), len(grid)), dtype=int)
    totals = np.empty(len(insts), dtype=int)
    for i, (inst, idx, D) in enumerate(zip(insts, idxs, distances)):
        keep = [a for a in range(len(inst['names'])) if a not in (idx['repeat'], idx['goal'])]
        for k, lam in enumerate(grid):
            C = inst['K']['raw'][:, i] + lam * inst['norm_m']
            con = concordance(C[keep], D[keep])
            counts[i, k], totals[i] = con['concordant'], con['total']
    chosen, folds = [], []
    for i in range(len(insts)):
        train = counts.sum(axis=0) - counts[i]
        k = int(np.argmax(train))  # ascending grid, first exact tie
        chosen.append(float(grid[k]))
        folds.append(dict(held_out=i, chosen_lambda=float(grid[k]), training_concordant=int(train[k]),
                          training_total=int(totals.sum()-totals[i]), held_out_concordant=int(counts[i,k])))
    curve = [dict(lambda_value=float(lam), concordance=float(counts[:,k].sum()/totals.sum()),
                  concordant=int(counts[:,k].sum()), total=int(totals.sum())) for k, lam in enumerate(grid)]
    return dict(chosen_lambdas=chosen, any_at_upper_edge=bool(any(v == cfg.lambda_max for v in chosen)),
                folds=folds, in_sample_curve=curve)


def guards(summary, insts, idxs, responses):
    """Abort before emitting results unless the E001/E003 anchors are reproduced."""
    expected = {'raw': dict(concordant=515, total=1088, n_regret0=7, toward_falls=16, away_rises=1, lateral_rises=1,
                            foreign_n=672, foreign_negative=672),
                'real': dict(concordant=813, total=1088),
                'foreign': dict(concordant=907, total=1088, n_regret0=7, toward_falls=14, away_rises=16, lateral_rises=54)}
    checks = {}
    for variant, values in expected.items():
        for key, want in values.items():
            got = summary[variant][key]
            checks[f'{variant}_{key}'] = dict(measured=got, expected=want, passed=got == want)
    previous = json.loads((ROOT/'episodes/E003-goal-free-term/out/result.json').read_text())['pred_cmd']['foreign_response']
    vals = np.concatenate([F[idx['big']].ravel() for F, idx in zip(responses, idxs)])
    checks['raw_foreign_overall_median'] = dict(measured=float(np.median(vals)), expected=previous['median_F_pred'],
                                              passed=bool(np.isclose(np.median(vals), previous['median_F_pred'], rtol=0, atol=1e-14)))
    # Exact class/amplitude medians in E003; aggregate-amplitude medians were not stored there.
    for row in previous['by_class_amplitude']:
        vals = [F[a,j] for inst, idx, F in zip(insts, idxs, responses) for a in idx[row['cls']]
                if round(inst['commanded_mm'][a]) == row['commanded_mm'] for j in range(7)]
        got, want = float(np.median(vals)), row['median_F_pred']
        checks[f"foreign_{row['cls']}_{row['commanded_mm']}"] = dict(measured=got, expected=want,
                                                                   passed=bool(np.isclose(got,want,rtol=0,atol=1e-14)))
    old_gain = json.loads((ROOT/'episodes/E001-amplitude-sweep/out/oddeven.json').read_text())['onray']
    for cm in (1, 3):
        old = next(r['G_odd'] for r in old_gain if r['commanded_mm'] == cm*10)
        got = summary['raw'][f'gain_{cm}cm']
        checks[f'raw_gain_{cm}cm'] = dict(measured=got, expected=old['median'],
             passed=bool(np.isclose(got,old['median'],rtol=0,atol=1e-14) and summary['raw'][f'gain_n_{cm}cm'] == old['n'] == 8))
    result = dict(passed=all(r['passed'] for r in checks.values()), checks=checks)
    assert result['passed'], json.dumps(result, indent=2)
    return result


def predictions(summary, cfg):
    """Each registered clause is reported separately; no post-result thresholds."""
    d = summary['disp_channel']
    out = {}
    def add(key, measured, threshold, met):
        out[key] = dict(measured=measured, threshold=threshold, met=bool(met))
    add('P1_median_3cm', d['foreign_median_3cm'], f'> {cfg.p1_foreign_median_min}', d['foreign_median_3cm'] > cfg.p1_foreign_median_min)
    add('P1_negative', d['foreign_negative'], f'<= {cfg.p1_negative_fraction_max} * 672', d['foreign_negative'] <= cfg.p1_negative_fraction_max * 672)
    add('P2_toward', d['toward_falls'], f'== {cfg.p2_toward}', d['toward_falls'] == cfg.p2_toward)
    add('P2_away', d['away_rises'], f'>= {cfg.p2_away_min}', d['away_rises'] >= cfg.p2_away_min)
    add('P2_lateral', d['lateral_rises'], f'>= {cfg.p2_lateral_min}', d['lateral_rises'] >= cfg.p2_lateral_min)
    for cm in (1, 3):
        v = d[f'gain_{cm}cm']
        add(f'P2_gain_{cm}cm', v, str(cfg.p2_gain_range), v is not None and cfg.p2_gain_range[0] <= v <= cfg.p2_gain_range[1])
    add('P3_held_out', summary['amp_penalty']['fraction'], f'>= {cfg.p3_fraction_min}', summary['amp_penalty']['fraction'] >= cfg.p3_fraction_min)
    for a, b, inclusive in [('foreign','amp_penalty',True),('amp_penalty','disp_channel',False),('disp_channel','cosine',False),('cosine','raw',False)]:
        x, y = summary[a]['fraction'], summary[b]['fraction']
        add(f'P4_{a}_vs_{b}', [x,y], '>=' if inclusive else '>', x >= y if inclusive else x > y)
    for p in ('P1', 'P2', 'P3', 'P4'):
        out[p+'_met'] = all(v['met'] for k,v in out.items() if k.startswith(p+'_') and isinstance(v,dict))
    return out


def descriptive(insts, idxs, raw_responses, cfg):
    """View, target-alignment and arm/background summaries; no fitted thresholds."""
    meta = json.loads((ROOT/cfg.data_dir/'meta.json').read_text())
    maps = [v.get('view_map') for v in meta['instances'].values()]
    by_view = {}
    for v in range(2):
        F = [np.delete(inst['K_view'][:,:,v]-inst['K_view'][idx['neutral'],:,v], i, axis=1)
             for i,(inst,idx) in enumerate(zip(insts,idxs))]
        block = dict(foreign=foreign_summary(insts,idxs,F), own_even={}, dispersion={})
        for axis, prefixes in [('onray',['onray']),('offray',['offray0','offray1'])]:
            for cm in (1,3):
                vals = []
                for i,(inst,idx) in enumerate(zip(insts,idxs)):
                    C = inst['K_view'][:,i,v]
                    for prefix in prefixes:
                        p,m = [inst['names'].index(f'{prefix}-{sign}-{cm/100:g}') for sign in ('plus','minus')]
                        vals.append(float((C[p]+C[m])/2-C[idx['neutral']]))
                block['own_even'][f'{axis}_{cm}cm'] = dict(n=len(vals), median=float(np.median(vals)), values=vals)
        for field in ('pred_cmd','real'):
            block['dispersion'][field] = {}
            for group in ('neutral0','1cm','3cm','goal'):
                vals = []
                for inst,idx in zip(insts,idxs):
                    sel = [idx['neutral' if group == 'neutral0' else 'goal']] if group in ('neutral0','goal') else [a for a in idx['big'] if round(inst['commanded_mm'][a]) == (10 if group == '1cm' else 30)]
                    vals.extend(inst['dispersion'][field][sel,v].tolist())
                block['dispersion'][field][group] = dict(n=len(vals), median=float(np.median(vals)))
        by_view[f'view{v}'] = block
    pairs, own = [], []
    for i,(inst,idx,F) in enumerate(zip(insts,idxs,raw_responses)):
        three = [a for a in idx['big'] if round(inst['commanded_mm'][a]) == 30]
        lateral = [a for a in three if inst['classes'][a] == 'lateral']
        assert len(three) == 6 and len(lateral) == 4
        js = [j for j in range(len(insts)) if j != i]
        for col,j in enumerate(js):
            pairs.append(dict(instance=i, goal=j, starting_cost=float(inst['K']['raw'][idx['neutral'],j]),
                              response_all_3cm=float(F[three,col].mean()), response_lateral_3cm=float(F[lateral,col].mean())))
        C = inst['K']['raw'][:,i]
        own.append(dict(instance=i, starting_cost=float(C[idx['neutral']]), response_lateral_3cm=float((C[lateral]-C[idx['neutral']]).mean())))
    alignment = dict(n_foreign_pairs=len(pairs), foreign_points=pairs, own_goal_points=own)
    for group in ('all','lateral'):
        alignment[group+'_3cm'] = e003.correlations([r['starting_cost'] for r in pairs], [r[f'response_{group}_3cm'] for r in pairs])
    alignment['own_lateral_3cm'] = e003.correlations([r['starting_cost'] for r in own], [r['response_lateral_3cm'] for r in own])
    regions = {}
    for f in ('pred_cmd','real'):
        regions[f] = {}
        for s in ('arm','background'):
            vals = [v for inst in insts for v in inst['region_ratio'][f][s]]
            regions[f][s] = dict(n=len(vals), median=float(np.median(vals)), values=vals)
    return dict(view_labels=dict(labels=['view0','view1'], native_view_maps=maps,
        note='meta.json gives view_map [0,2] but no camera names; labels are view0/view1, not inferred agent/wrist names.'),
        by_view=by_view, alignment=alignment, arm_background=regions)


FIGURE = r'''\documentclass[border=5pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17,every axis/.append style={
 tick label style={font=\large},label style={font=\large},
 scaled ticks=false,axis line style={black!65},tick style={black!65},
 xtick={0,1,2,3,4,5,6,7},
 xticklabels={raw,real,foreign,channel,scalar,cosine,penalty,norm},
 xticklabel style={rotate=35,anchor=east},xmin=-0.5,xmax=7.5,
 ymajorgrids,grid style={black!10},legend style={font=\large,draw=none},
 unbounded coords=discard}}
\definecolor{one}{RGB}{0,114,178}
\definecolor{three}{RGB}{213,94,0}
\begin{document}
\begin{tikzpicture}
\begin{axis}[name=left,width=14cm,height=10cm,ylabel={median foreign response},
 legend pos=north west,yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=3}]
\addplot[black!60,thin,forget plot] coordinates {(-0.5,0) (7.5,0)};
\addplot[only marks,mark=*,mark size=3pt,one] table[x expr=\thisrow{x}-0.12,y=foreign_median_1cm]{variants.dat};
\addlegendentry{1 cm}
\addplot[only marks,mark=square*,mark size=3pt,three] table[x expr=\thisrow{x}+0.12,y=foreign_median_3cm]{variants.dat};
\addlegendentry{3 cm}
\end{axis}
\begin{axis}[at={(left.east)},anchor=west,xshift=2.3cm,width=14cm,height=10cm,
 ylabel={all-scope concordance},ymin=0.3,ymax=1,legend pos=north west]
\addplot[black!60,dashed] table[x=x,y=real_fraction]{variants.dat};
\addlegendentry{realized reference}
\addplot[only marks,mark=*,mark size=3pt,one,error bars/.cd,y dir=both,y explicit]
 table[x=x,y=fraction,y error minus=ci_minus,y error plus=ci_plus]{variants.dat};
\addlegendentry{95\% instance bootstrap}
\end{axis}
\end{tikzpicture}
\end{document}
'''


def write_figure(out):
    """Compile one two-panel pgfplots figure from variants.dat; rasterize at 200 dpi."""
    (out/'fig-contraction.tex').write_text(FIGURE)
    run = subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','fig-contraction.tex'], cwd=out, capture_output=True, text=True)
    if run.returncode:
        raise RuntimeError(run.stdout[-4000:])
    if shutil.which('pdftoppm'):
        subprocess.run(['pdftoppm','-png','-r','200','-singlefile','fig-contraction.pdf','fig-contraction'], cwd=out, check=True)
        renderer = 'pdftoppm'
    else:
        subprocess.run(['gs','-q','-dNOPAUSE','-dBATCH','-sDEVICE=png16m','-r200',
                        '-dTextAlphaBits=4','-dGraphicsAlphaBits=4',
                        '-sOutputFile=fig-contraction.png','fig-contraction.pdf'], cwd=out, check=True)
        renderer = 'ghostscript'
    for ext in ('aux','log'):
        (out/f'fig-contraction.{ext}').unlink(missing_ok=True)
    return renderer


def main():
    start = time.perf_counter()
    cfg = Config()
    # E003's two-pass loader retains goals only, never two complete instances.
    goals32, _ = e003.load_goals(cfg)
    goals = [tokens(g) for g in goals32]
    del goals32
    goal_units = [unit_tokens(g) for g in goals]
    insts = []
    for path in e003.instance_paths(cfg):
        insts.append(load_reduced(path, goals, goal_units, cfg))
    del goals, goal_units
    idxs = [e003.index_sets(inst,cfg) for inst in insts]
    distances = [np.linalg.norm(inst['ee']-inst['ee'][idx['goal']],axis=1)*1000 for inst,idx in zip(insts,idxs)]
    for inst,idx in zip(insts,idxs):
        assert inst['names'] == insts[0]['names']
        assert [len(idx[c]) for c in e003.CLASSES] == [2,2,8]
    raw_F = foreign_responses(insts,idxs,'raw')
    raw_median = foreign_summary(insts,idxs,raw_F)['foreign_median_3cm']
    own = {v: [inst['K'][v][:,i] for i,inst in enumerate(insts)] for v in ('raw','real','disp_channel','disp_scalar','cosine')}
    F = {v: foreign_responses(insts,idxs,v) for v in own}
    own['foreign'], _ = e003.corrected_costs(insts,idxs,'pred_cmd')
    F['foreign'] = [(r - (r.sum(axis=1,keepdims=True)-r)/6) for r in raw_F]
    amp = amplitude_folds(insts,idxs,distances,cfg)
    own['amp_penalty'] = [C+lam*inst['norm_m'] for C,lam,inst in zip(own['raw'],amp['chosen_lambdas'],insts)]
    F['amp_penalty'] = [r+lam*inst['norm_m'][:,None] for r,lam,inst in zip(raw_F,amp['chosen_lambdas'],insts)]
    own['norm_only'] = [inst['norm_m'] for inst in insts]
    F['norm_only'] = [np.repeat(inst['norm_m'][:,None],7,axis=1) for inst in insts]
    summary, variants = {}, {}
    for v in VARIANTS:
        summary[v],variants[v] = metrics(insts,idxs,distances,own[v],F[v],cfg,raw_median,cosine=v=='cosine')
    guard = guards(summary,insts,idxs,raw_F)
    desc = descriptive(insts,idxs,raw_F,cfg)
    shifted_raw_median = foreign_summary(insts,idxs,foreign_responses(insts,idxs,'exec_raw'))['foreign_median_3cm']
    desc['shifted'] = dict(field='z_pred_exec', raw_foreign_median_3cm=shifted_raw_median,
                           removed_3cm_reference='raw z_pred_exec foreign median at 3 cm',variants={})
    for v in ('disp_channel','disp_scalar'):
        C = [inst['K']['exec_'+v][:,i] for i,inst in enumerate(insts)]
        _, desc['shifted']['variants'][v] = metrics(insts,idxs,distances,C,foreign_responses(insts,idxs,'exec_'+v),cfg,shifted_raw_median)
    result: dict[str, Any] = dict(episode='E004-contraction', config=asdict(cfg),
        populations=dict(instances=8,candidates_per_instance=19,candidates_ge_1cm=96,foreign_pairs=672,
                         foreign_pairs_per_amplitude=336,toward=16,away=16,lateral=64,all_scope_pairs=1088),
        guards=guard,variants=variants,predictions=predictions(summary,cfg),descriptive=desc,amplitude_penalty=amp,
        instances=[dict(instance=i,names=inst['names'],commanded_mm=inst['commanded_mm'],D_mm=distances[i].tolist(),
                        arm_token_indices=inst['arm_token_indices']) for i,inst in enumerate(insts)],
        deviations=[],notes=['All results exploratory: same eight E001 instances, not a fresh sweep.',
        'Foreign/gain metrics for real are defined and computed rather than left null.',
        'Cosine and norm-only foreign responses have different units from L1; removed_3cm is reported literally, not a unit-invariant comparison.',
        'Scalar zero-dispersion fallback is factor 1 (same as channel); cosine zero-norm tokens are rejected.',
        'Shifted removed_3cm uses the shifted raw reference; other gain denominators remain realized own-goal L1.'])
    out = ROOT/cfg.out_dir
    out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(summary,indent=1,allow_nan=False)+'\n')
    rows = []
    for i,(v,m) in enumerate(summary.items()):
        row = dict(x=i,variant=v)
        # Compact JSON list cells remain one whitespace-separated token.
        row.update({k: json.dumps(value,separators=(',',':')) if isinstance(value,list) else value for k,value in m.items()})
        row.update(ci_low=m['ci95'][0],ci_high=m['ci95'][1],ci_minus=m['fraction']-m['ci95'][0],
                   ci_plus=m['ci95'][1]-m['fraction'],real_fraction=summary['real']['fraction'])
        rows.append(row)
    e003.write_dat(out/'variants.dat',list(rows[0]),rows)
    e003.write_dat(out/'lambda_curve.dat',['lambda_value','concordance','concordant','total'],amp['in_sample_curve'])
    result['figure_renderer'] = write_figure(out)
    if result['figure_renderer'] != 'pdftoppm':
        result['deviations'].append('pdftoppm is not installed/on PATH; rendered PNG with Ghostscript at the requested 200 dpi, without installing packages.')
    result['runtime_seconds'] = time.perf_counter()-start
    (out/'result.json').write_text(json.dumps(result,indent=1,allow_nan=False)+'\n')
    print(json.dumps(dict(guards_passed=guard['passed'],summary=summary,predictions=result['predictions'],
                         chosen_lambdas=amp['chosen_lambdas'],runtime_seconds=result['runtime_seconds']),indent=1))


if __name__ == '__main__':
    main()
