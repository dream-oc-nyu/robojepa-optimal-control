"""E005: fixed-view costs and fixed-radius shell rankings, CPU-only float64.

Run from the repository root: python3 episodes/E005-view-and-shell/run.py
One full instance at a time; the unchanged E001 evaluator defines all scores.
"""
from __future__ import annotations

import importlib.util
import json
import resource
import sys
import time
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
_spec = importlib.util.spec_from_file_location('e004_helpers', ROOT / 'episodes/E004-contraction/run.py')
assert _spec is not None and _spec.loader is not None
e004 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = e004
_spec.loader.exec_module(e004)
e003 = e004.e003
cost, concordance, selector, load_instance, divide = (e004.cost, e004.concordance, e004.selector, e004.load_instance, e004.divide)
VARIANTS = ('raw', 'view0', 'view1', 'disp_channel', 'foreign', 'real', 'real_view0', 'real_view1')
A_KEYS = ('foreign_n', 'foreign_negative', 'foreign_median_1cm', 'foreign_median_3cm', 'gain_1cm', 'gain_3cm', 'gain_n_1cm', 'gain_n_3cm')


@dataclass(frozen=True)
class Config:
    """Registered populations, noise exclusion, resampling and thresholds."""
    data_dir: str = 'runs/E001-amplitude-sweep/scoring-input'
    out_dir: str = 'episodes/E005-view-and-shell/out'
    n_instances: int = 8
    min_command_mm: float = 9.99
    moving_reference: str = 'onray-plus-0.001'
    bootstrap_draws: int = 2000
    bootstrap_seed: int = 0
    gain_noise_multiple: float = 10.0
    p1_gain_3cm: float = 0.30
    p1_toward: int = 16
    p2_away: int = 10
    p2_lateral: int = 24
    p2_fraction: float = 0.62
    p2_selection: int = 7
    p3_shell_fraction: float = 0.70
    p3_distinct_fraction: float = 0.90
    p3_selection: tuple = (8, 7)
    p4_difference: float = 0.05


def load_reduced(path, goals, cfg):
    """K_v(a,j) = mean |z_v(a)-g_v(j)|; keep only small cost arrays."""
    x = load_instance(path)
    names, kinds = x['names'].tolist(), x['kind'].tolist()
    inst = dict(names=names, kinds=kinds,
                classes=[e003.candidate_class(n, k) for n, k in zip(names, kinds)],
                commanded_mm=(np.linalg.norm(x['commanded_xyz'], axis=1)*1000).tolist(),
                ee=x['ee_world'].copy())
    idx = e003.index_sets(inst, cfg)
    n, ng = len(names), len(goals)
    K = {v: np.empty((n, ng)) for v in VARIANTS if v != 'foreign'}
    max_error = 0.0
    sigma0 = e004.tokens(x['z_pred_cmd'][idx['neutral']]).std(axis=1)
    for a in range(n):
        z = e004.tokens(x['z_pred_cmd'][a])
        r = e004.tokens(x['z_real'][a])
        matched = e004.match_dispersion(z, sigma0)
        for j, g in enumerate(goals):
            for key, features, viewkeys in [('raw', z, ('view0', 'view1')), ('real', r, ('real_view0', 'real_view1'))]:
                K[key][a,j] = cost(features, g)
                for v, vkey in enumerate(viewkeys):
                    K[vkey][a,j] = cost(features[v], g[v])
                err = abs(K[key][a,j] - .5*K[viewkeys[0]][a,j] - .5*K[viewkeys[1]][a,j])
                max_error = max(max_error, err)
            K['disp_channel'][a,j] = cost(matched, g)
    assert max_error <= 1e-12
    inst.update(K=K, K_pred_cmd=K['raw'], K_real=K['real'], raw_view_max_error=max_error)
    del x
    return inst


def shell_instance(inst, C, D, cm):
    """15 unordered pairs = 9 class-distinct + 6 lateral-lateral; ties count."""
    keep = [a for a, name in enumerate(inst['names']) if name.endswith(f'-{cm/100:g}')]
    assert len(keep) == 6
    full = concordance(C[keep], D[keep])
    split = {k: dict(concordant=0, total=0) for k in ('distinct', 'latlat')}
    for a, b in combinations(keep, 2):
        key = 'latlat' if inst['classes'][a] == inst['classes'][b] == 'lateral' else 'distinct'
        pair = concordance(C[[a,b]], D[[a,b]])
        for metric in ('concordant', 'total'):
            split[key][metric] += pair[metric]
    pick = min(keep, key=lambda a: C[a])
    toward = next(a for a in keep if inst['classes'][a] == 'toward')
    away = next(a for a in keep if inst['classes'][a] == 'away')
    lateral = [a for a in keep if inst['classes'][a] == 'lateral']
    assert split['distinct']['total'] == 9 and split['latlat']['total'] == 6
    assert full['concordant'] == sum(s['concordant'] for s in split.values())
    return dict(**full, **{f'{k}_{m}': v for k, counts in split.items() for m,v in counts.items()},
                selected=inst['names'][pick], toward_selected=pick == toward,
                toward_physically_closest=bool(D[toward] == min(D[keep])),
                strict_class_distance_order=bool(D[toward] < min(D[lateral]) and max(D[lateral]) < D[away]),
                names=[inst['names'][a] for a in keep], costs=C[keep].tolist(), D_mm=D[keep].tolist())


def shell_metrics(insts, costs, distances, sampled):
    """Pool shell counts; percentile intervals resample the eight whole instances."""
    summary, details = {}, {}
    for cm in (1, 3):
        key = f'shell{cm}'
        rows = [shell_instance(inst, C, D, cm) for inst, C, D in zip(insts, costs, distances)]
        for metric in ('concordant', 'distinct_concordant', 'latlat_concordant', 'toward_selected'):
            summary[f'{key}_{metric}'] = int(sum(r[metric] for r in rows))
        num = np.array([r['distinct_concordant'] for r in rows])[sampled].sum(axis=1)
        summary[f'{key}_distinct_ci95'] = np.quantile(num / 72, [.025, .975]).tolist()
        details[key] = rows
    return summary, details


def metrics(insts, idxs, distances, own, variant, cfg, sampled):
    """Fixed E003 score and E004 foreign/gain helpers, same-weight real denominator."""
    C = own[variant]
    scored = e003.score(insts, idxs, distances, C, cfg)
    census, rank = scored['sign_census_ge_1cm'], scored['rankings']['all']
    m = dict.fromkeys(A_KEYS)
    gains, F = {}, None
    if variant in ('raw', 'view0', 'view1'):
        F = e004.foreign_responses(insts, idxs, variant)
        m.update(e004.foreign_summary(insts, idxs, F))
        rkey = {'raw': 'real', 'view0': 'real_view0', 'view1': 'real_view1'}[variant]
        same_view = [dict(inst, K={**inst['K'], 'real': inst['K'][rkey]}) for inst in insts]
        gains = e004.gain_details(same_view, idxs, C, cfg)
        m.update({k: gains[k] for k in A_KEYS if k.startswith('gain_')})
    m.update(toward_falls=census['toward']['falls'], away_rises=census['away']['rises'], lateral_rises=census['lateral']['rises'])
    m.update({k: rank[k] for k in ('concordant', 'total', 'fraction', 'n_regret0', 'regret_mm')})
    m['ci95'] = rank['bootstrap_ci95']
    shell, rows = shell_metrics(insts, C, distances, sampled)
    m.update(shell)
    detail = dict(summary=m, sign_census=census, rankings=scored['rankings'], gains=gains, per_instance=[])
    for i, (inst, idx, c, d) in enumerate(zip(insts, idxs, C, distances)):
        detail['per_instance'].append(dict(instance=i, costs=c.tolist(), dC=(c-c[idx['neutral']]).tolist(),
            foreign_goals=[j for j in range(8) if j != i] if F is not None else None,
            foreign_response=F[i].tolist() if F is not None else None,
            ranking=e003.rank_instance(inst, idx, c, d)['all'],
            sign_census=e003.sign_census(c-c[idx['neutral']], idx),
            shells={s: r[i] for s,r in rows.items()}))
    assert m['total'] == 1088
    return m, detail


def prediction_checks(s, cfg):
    """Report every pre-registered inequality separately, without fitted cutoffs."""
    out = {}
    def add(key, measured, threshold, met):
        out[key] = dict(measured=measured, threshold=threshold, met=bool(met))
    v, r = s['view0'], s['raw']
    add('P1_gain_3cm', v['gain_3cm'], f'>= {cfg.p1_gain_3cm}', v['gain_3cm'] >= cfg.p1_gain_3cm)
    for cm in (1,3):
        a, b = v[f'gain_{cm}cm'], s['view1'][f'gain_{cm}cm']
        add(f'P1_gain_view0_gt_view1_{cm}cm', [a,b], 'view0 > view1', a > b)
    add('P1_toward', v['toward_falls'], f'== {cfg.p1_toward}', v['toward_falls'] == cfg.p1_toward)
    for key, threshold in [('away_rises',cfg.p2_away), ('lateral_rises',cfg.p2_lateral), ('fraction',cfg.p2_fraction), ('n_regret0',cfg.p2_selection)]:
        add('P2_'+key, v[key], f'>= {threshold}', v[key] >= threshold)
    for cm, selection in zip((1,3),cfg.p3_selection):
        key = f'shell{cm}'
        add(f'P3_{key}_fraction', r[key+'_concordant']/120, f'>= {cfg.p3_shell_fraction}', r[key+'_concordant']/120 >= cfg.p3_shell_fraction)
        add(f'P3_{key}_distinct', r[key+'_distinct_concordant']/72, f'>= {cfg.p3_distinct_fraction}', r[key+'_distinct_concordant']/72 >= cfg.p3_distinct_fraction)
        add(f'P3_{key}_toward_selected', r[key+'_toward_selected'], f'== {selection}', r[key+'_toward_selected'] == selection)
        for variant in ('disp_channel','foreign'):
            delta = abs(s[variant][key+'_distinct_concordant']-r[key+'_distinct_concordant'])/72
            add(f'P4_{key}_{variant}', delta, f'<= {cfg.p4_difference}', delta <= cfg.p4_difference)
    for p in ('P1','P2','P3','P4'):
        out[p+'_met'] = all(v['met'] for k,v in out.items() if k.startswith(p+'_') and isinstance(v,dict))
    return out


def guards(s, insts, path):
    """Abort publication unless historical anchors and the balanced path agree."""
    checks = {}
    def check(name, got, expected, atol=0):
        checks[name] = dict(measured=got, expected=expected, atol=atol,
                            passed=bool(np.isclose(got, expected, rtol=0, atol=atol)))
    for v, con, regret in [('raw',515,7), ('real',813,None), ('disp_channel',634,8), ('foreign',907,None)]:
        check(v+'_concordant', s[v]['concordant'], con)
        check(v+'_total', s[v]['total'], 1088)
        if regret is not None:
            check(v+'_n_regret0', s[v]['n_regret0'], regret)
    for cm, target in [(1,.221437),(3,.264436)]:
        check(f'raw_gain_{cm}cm',s['raw'][f'gain_{cm}cm'],target,1e-6)
        check(f'raw_gain_n_{cm}cm',s['raw'][f'gain_n_{cm}cm'],8)
    for v, n, med1, med3 in [('view0',655,-.00138197,-.00387324),('view1',672,-.01376155,-.02838163)]:
        for key, target, tol in [('foreign_n',672,0),('foreign_negative',n,0),('foreign_median_1cm',med1,1e-8),('foreign_median_3cm',med3,1e-8)]:
            check(v+'_'+key,s[v][key],target,tol)
    check('balanced_weight_concordant',path[5]['concordant'],515)
    check('balanced_weight_n_regret0',path[5]['n_regret0'],7)
    check('raw_view_max_abs_error',max(i['raw_view_max_error'] for i in insts),0,1e-12)
    result = dict(passed=all(c['passed'] for c in checks.values()), checks=checks)
    assert result['passed'], json.dumps(result,indent=2)
    return result


def odd_shares(insts, own):
    """0.5 odd_v / odd_raw, with exact-zero denominator reported as null."""
    result = {}
    for field, keys in [('predicted',('raw','view0','view1')), ('realized',('real','real_view0','real_view1'))]:
        result[field] = {}
        for cm in (1,3):
            vals = [[], []]
            for i, inst in enumerate(insts):
                p,m = [inst['names'].index(f'onray-{sign}-{cm/100:g}') for sign in ('plus','minus')]
                odds = [(own[k][i][p]-own[k][i][m])/2 for k in keys]
                for v in range(2):
                    vals[v].append(divide(.5*odds[v+1],odds[0]))
            result[field][f'{cm}cm'] = {f'view{v}': dict(values=a, median=float(np.median([x for x in a if x is not None])), n=sum(x is not None for x in a)) for v,a in enumerate(vals)}
    return result


FIGURE = r'''\documentclass[border=5pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17,every axis/.append style={tick label style={font=\large},label style={font=\large},scaled ticks=false,axis line style={black!65},tick style={black!65},ymajorgrids,grid style={black!10},legend style={font=\large,draw=none}}}
\definecolor{one}{RGB}{0,114,178}
\definecolor{three}{RGB}{213,94,0}
\begin{document}
\begin{tikzpicture}
\begin{axis}[name=left,width=15cm,height=10cm,ylabel={class-distinct shell concordance},ymin=0.4,ymax=1.05,xmin=-0.5,xmax=7.5,xtick={0,1,2,3,4,5,6,7},xticklabels={raw,view 0,view 1,channel,foreign,real,real v0,real v1},xticklabel style={rotate=35,anchor=east},legend pos=south east]
\addplot[black!65,dotted,forget plot] coordinates {(-.5,.5) (7.5,.5)};
\addplot[only marks,mark=*,mark size=3pt,one,error bars/.cd,y dir=both,y explicit]
 table[x expr=\thisrow{x}-0.11,y=plot1,y error minus=ci_minus,y error plus=ci_plus]{shells.dat};
\addlegendentry{1 cm}
\addplot[only marks,mark=square*,mark size=3pt,three,error bars/.cd,y dir=both,y explicit]
 table[x expr=\thisrow{x}+0.11,y=plot3,y error minus=ci_minus,y error plus=ci_plus]{shells.dat};
\addlegendentry{3 cm}
\end{axis}
\begin{axis}[at={(left.east)},anchor=west,xshift=2.6cm,width=14cm,height=10cm,xlabel={view-0 weight $w_0$},ylabel={all-scope concordance},xmin=-.04,xmax=1.04,ymin=.4,ymax=.8,xtick={0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1},legend pos=north west]
\addplot[one,thick,mark=*,mark size=2.8pt,point meta=explicit symbolic,nodes near coords,nodes near coords style={font=\large,anchor=south,yshift=3pt}]
 table[x=w0,y=fraction,meta=n_regret0]{weight_path.dat};
\addlegendentry{labels: zero-regret / 8}
\end{axis}
\end{tikzpicture}
\end{document}
'''


def write_figure(out):
    """Reuse E004's compiler/rasterizer, then give its products E005 names."""
    e004.FIGURE = FIGURE.replace('ymajorgrids,', 'unbounded coords=discard,ymajorgrids,')
    renderer = e004.write_figure(out)
    for ext in ('tex','pdf','png'):
        (out/f'fig-contraction.{ext}').replace(out/f'fig-view-shell.{ext}')
    return renderer


def main():
    start, cfg = time.perf_counter(), Config()
    goals32, _ = e003.load_goals(cfg)
    goals = [e004.tokens(g) for g in goals32]
    del goals32
    insts = []
    for path in e003.instance_paths(cfg):
        insts.append(load_reduced(path, goals, cfg))
        print(f'Reduced {path.name}', flush=True)
    del goals
    idxs = [e003.index_sets(inst,cfg) for inst in insts]
    distances = [np.linalg.norm(inst['ee']-inst['ee'][idx['goal']],axis=1)*1000 for inst,idx in zip(insts,idxs)]
    assert all(inst['names'] == insts[0]['names'] for inst in insts)
    assert all([len(idx[c]) for c in e003.CLASSES] == [2,2,8] for idx in idxs)
    own = {v: [inst['K'][v][:,i] for i,inst in enumerate(insts)] for v in VARIANTS if v != 'foreign'}
    own['foreign'], _ = e003.corrected_costs(insts,idxs,'pred_cmd')
    sampled = np.random.default_rng(cfg.bootstrap_seed).integers(0,8,(cfg.bootstrap_draws,8))
    summary, variants = {}, {}
    for v in VARIANTS:
        summary[v], variants[v] = metrics(insts,idxs,distances,own,v,cfg,sampled)
    weight_path = []
    for k in range(11):
        w = k/10
        C = [w*a+(1-w)*b for a,b in zip(own['view0'],own['view1'])]
        rank = e003.score(insts,idxs,distances,C,cfg)['rankings']['all']
        shell, _ = shell_metrics(insts,C,distances,sampled)
        weight_path.append(dict(w0=w,concordant=rank['concordant'],total=rank['total'],fraction=rank['fraction'],n_regret0=rank['n_regret0'],
                                shell1_distinct_concordant=shell['shell1_distinct_concordant'],shell3_distinct_concordant=shell['shell3_distinct_concordant']))
    guard = guards(summary,insts,weight_path)
    physical = [r['shells'][f'shell{cm}'] for r in variants['raw']['per_instance'] for cm in (1,3)]
    desc = dict(odd_shares=odd_shares(insts,own),weight_path=weight_path,
                shell_physics=dict(n=len(physical),toward_physically_closest=sum(r['toward_physically_closest'] for r in physical),
                                   strict_toward_lateral_away=sum(r['strict_class_distance_order'] for r in physical)))
    result = dict(episode='E005-view-and-shell',config=asdict(cfg),
        populations=dict(instances=8,candidates_per_instance=19,candidates_ge_1cm=96,foreign_pairs=672,foreign_pairs_per_amplitude=336,
                         toward=16,away=16,lateral=64,all_scope_pairs=1088,shell_pairs=120,shell_distinct_pairs=72,shell_latlat_pairs=48),
        guards=guard,variants=variants,predictions=prediction_checks(summary,cfg),descriptive=desc,
        instances=[dict(instance=i,names=inst['names'],commanded_mm=inst['commanded_mm'],D_mm=distances[i].tolist()) for i,inst in enumerate(insts)],
        bootstrap_instance_indices=sampled.tolist(), deviations=[],
        notes=['Exploratory reanalysis of the same eight E001 instances, not a fresh sweep.',
               'Block A is defined only for raw/view0/view1; the other five variants use null.',
               'Physical class ordering means toward strictly closer than every lateral, each strictly closer than away.',
               'Shell selection thresholds are literal equalities (8 at 1 cm, 7 at 3 cm).',
               'Shells use name suffixes exactly; all comparisons retain exact floating-point ties.'])
    out = ROOT/cfg.out_dir
    out.mkdir(parents=True,exist_ok=True)
    rows = []
    for x,v in enumerate(VARIANTS):
        for cm in (1,3):
            m, key = summary[v], f'shell{cm}'
            frac = m[key+'_distinct_concordant']/72
            lo,hi = m[key+'_distinct_ci95']
            rows.append(dict(x=x,variant=v,shell_cm=cm,concordant=m[key+'_concordant'],total=120,
                distinct_concordant=m[key+'_distinct_concordant'],distinct_total=72,latlat_concordant=m[key+'_latlat_concordant'],latlat_total=48,
                distinct_fraction=frac,ci_low=lo,ci_high=hi,ci_minus=frac-lo,ci_plus=hi-frac,
                toward_selected=m[key+'_toward_selected'],plot1=frac if cm == 1 else 'nan',plot3=frac if cm == 3 else 'nan'))
    e003.write_dat(out/'shells.dat',list(rows[0]),rows)
    e003.write_dat(out/'weight_path.dat',list(weight_path[0]),weight_path)
    result['figure_renderer'] = write_figure(out)
    if result['figure_renderer'] != 'pdftoppm':
        result['deviations'].append('pdftoppm unavailable; PNG rendered with installed Ghostscript at 200 dpi, no installation.')
    result['runtime_seconds'] = time.perf_counter()-start
    result['peak_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    (out/'summary.json').write_text(json.dumps(summary,indent=1,allow_nan=False)+'\n')
    (out/'result.json').write_text(json.dumps(result,indent=1,allow_nan=False)+'\n')
    print(json.dumps(dict(guards_passed=guard['passed'],runtime_seconds=result['runtime_seconds'],peak_rss_mib=result['peak_rss_mib'],predictions=result['predictions']),indent=1))


if __name__ == '__main__':
    main()
