"""E003: is the motion term of the predicted cost goal-free, and can it be subtracted?

Run from the repository root:  python3 episodes/E003-goal-free-term/run.py

Notation. Instance i (8 of them), candidate a (19 per instance), reference
0 = `neutral0`, own goal g_i = z_real_i(goal), foreign goals g_j, j != i.

    C(z, g)       = mean |z - g|                       (readings.cost, float64)
    K_f[i, a, j]  = C(z_f,i(a), g_j)                   f in {pred_cmd, pred_exec, real}
    dC_f(a)       = K_f[i, a, i] - K_f[i, 0, i]        own-goal cost change
    F_f(a, j)     = K_f[i, a, j] - K_f[i, 0, j]        foreign response, j != i
    b_i(a)        = mean_{j != i} F_pred(a, j)         goal-free baseline (b_i(0) = 0)
    C_corr(a)     = K_pred[i, a, i] - b_i(a)           corrected cost
    D(a)          = |ee_i(a) - ee_i(goal)|             physical distance (mm)

Every quantity below is a function of the cost tensor K_f (8, 19, 8), of the
end-effector positions, and of a few per-candidate feature statistics. Ranking
and selection use the fixed E001 evaluator (`readings.selector`) unchanged; the
pooling and cluster bootstrap repeat `analyze.pool` (2000 draws, seed 0).

The raw arrays are read only, one instance at a time (two passes: goals, then
costs). Outputs go to episodes/E003-goal-free-term/out/.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'episodes' / 'E001-amplitude-sweep'))
from readings import cost, divide, load_instance, selector  # noqa: E402  (fixed evaluator)

FIELDS = ('pred_cmd', 'pred_exec', 'real')
SCOPES = ('all', 'onray', 'offray')
CLASSES = ('toward', 'away', 'lateral')
CLASS_ID = {'neutral0': 0, 'toward': 1, 'away': 2, 'lateral': 3, 'neutral1': 4, 'goal': 5}


@dataclass(frozen=True)
class Config:
    """Paths, populations, bootstrap settings and the pre-registered thresholds."""
    data_dir: str = 'runs/E001-amplitude-sweep/scoring-input'
    out_dir: str = 'episodes/E003-goal-free-term/out'
    n_instances: int = 8
    min_command_mm: float = 9.99          # "commands >= 1 cm"
    moving_reference: str = 'onray-plus-0.001'
    bootstrap_draws: int = 2000
    bootstrap_seed: int = 0
    # pre-registered thresholds (episode.md, Prediction 1-3)
    p1_fpred_neg_min: float = 0.85
    p1_freal_neg_range: tuple = (0.35, 0.65)
    p2_away_rise_min: int = 12
    p2_lateral_rise_min: int = 40
    p2_toward_fall_required: int = 16
    p3_concordance_min: float = 0.60
    p3_regret0_required: int = 8


# ----------------------------------------------------------------------------- loading

def instance_paths(cfg):
    paths = [ROOT / cfg.data_dir / f'instance-{i:02d}.npz' for i in range(cfg.n_instances)]
    assert all(p.exists() for p in paths), 'missing instance files'
    return paths


def candidate_class(name, kind):
    """toward = onray-plus, away = onray-minus, lateral = offray; else the name."""
    if kind == 'onray':
        return 'toward' if name.startswith('onray-plus') else 'away'
    return 'lateral' if kind == 'offray' else name


def load_goals(cfg):
    """Pass 1: g_j = z_real_j[goal] (float32, 4 MB each) and ee_j(goal)."""
    goals, ee_goal = [], []
    for path in instance_paths(cfg):
        x = load_instance(path)
        g = int(np.flatnonzero(x['kind'] == 'goal')[0])
        goals.append(x['z_real'][g].copy())
        ee_goal.append(x['ee_world'][g].copy())
        del x
    return goals, np.stack(ee_goal)


def token_statistics(z):
    """Per-candidate feature statistics of z (2, 15, 20, C), in float64.

    token_std  = mean_c std_t z[t, c]   over the T = 2*15*20 tokens (population std)
    mean_abs   = mean_{t, c} |z[t, c]|
    token_l1   = mean_t sum_c |z[t, c]| = C * mean_abs
    """
    t = np.asarray(z, np.float64).reshape(-1, z.shape[-1])
    return dict(token_std=float(t.std(axis=0).mean()), mean_abs=float(np.abs(t).mean()),
                token_l1=float(np.abs(t).sum(axis=1).mean()))


def read_instance(path, goals):
    """Pass 2: K_f[a, j] for the three fields, plus per-candidate descriptives.

    err_f(a) = mean |z_pred_f(a) - z_real(a)| (prediction error, via readings.cost).
    """
    x = load_instance(path)
    names = [str(n) for n in x['names']]
    kinds = [str(k) for k in x['kind']]
    out = dict(names=names, kinds=kinds, classes=[candidate_class(n, k) for n, k in zip(names, kinds)],
               commanded_mm=[float(np.linalg.norm(c) * 1000) for c in x['commanded_xyz']],
               ee=x['ee_world'].copy())
    for f in FIELDS:
        z = x['z_' + f]
        out['K_' + f] = np.array([[cost(z[a], g) for g in goals] for a in range(len(names))])
        stats = [token_statistics(z[a]) for a in range(len(names))]
        for key in ('token_std', 'mean_abs', 'token_l1'):
            out[f'{key}_{f}'] = np.array([s[key] for s in stats])
    for f in ('pred_cmd', 'pred_exec'):
        out['err_' + f] = np.array([cost(x['z_' + f][a], x['z_real'][a]) for a in range(len(names))])
    del x
    return out


# ----------------------------------------------------------------------------- core maths

def index_sets(inst, cfg):
    """Row indices: neutral0, repeat, goal, and the >= 1 cm candidates by class."""
    names, classes = inst['names'], inst['classes']
    idx = dict(neutral=names.index('neutral0'), repeat=names.index('neutral1'), goal=names.index('goal'),
               moving_ref=names.index(cfg.moving_reference))
    big = [a for a, c in enumerate(classes) if c in CLASSES and inst['commanded_mm'][a] >= cfg.min_command_mm]
    idx['big'] = big
    for c in CLASSES:
        idx[c] = [a for a in big if classes[a] == c]
    return idx


def own_cost(inst, i, field):
    """K_f[i, :, i]: cost of every candidate against the own goal."""
    return inst['K_' + field][:, i]


def foreign_response(inst, i, field, neutral):
    """F_f(a, j) for all j (N, 8); the own-goal column j = i is set to nan."""
    K = inst['K_' + field]
    F = K - K[neutral][None, :]
    F[:, i] = np.nan
    return F


def baseline_mean(F):
    """b(a) = mean_{j != i} F(a, j)."""
    return np.nanmean(F, axis=1)


def sign_census(dC, idx):
    """Counts over the >= 1 cm candidates of one instance: falls (<0), rises (>0), zeros."""
    return {c: dict(n=len(idx[c]), falls=int(sum(dC[a] < 0 for a in idx[c])),
                    rises=int(sum(dC[a] > 0 for a in idx[c])),
                    zeros=int(sum(dC[a] == 0 for a in idx[c]))) for c in CLASSES}


def add_census(total, part):
    for c in CLASSES:
        e = total.setdefault(c, dict(n=0, falls=0, rises=0, zeros=0))
        for k in e:
            e[k] += part[c][k]
    return total


def rank_instance(inst, idx, C, D_mm):
    """Fixed E001 evaluator on one instance: rows = all but neutral1 and goal."""
    rows = [dict(name=inst['names'][a], kind=inst['kinds'][a], D_mm=float(D_mm[a]), C_x=float(C[a]))
            for a in range(len(inst['names'])) if a not in (idx['repeat'], idx['goal'])]
    neutral_D = float(D_mm[idx['neutral']])
    return {s: selector(rows if s == 'all' else [r for r in rows if r['kind'] == s], 'x', neutral_D)
            for s in SCOPES}


def pool_rankings(per_instance, cfg):
    """Pooled pair counts and cluster bootstrap over instances, as analyze.pool.

    fraction = sum_i concordant_i / sum_i total_i; the 95% interval is the
    percentile interval of that ratio over 2000 resamples of whole instances.
    """
    n = len(per_instance)
    sampled = np.random.default_rng(cfg.bootstrap_seed).integers(0, n, (cfg.bootstrap_draws, n))
    out = {}
    for s in SCOPES:
        rows = [r[s] for r in per_instance]
        con = {k: int(sum(r['concordance'][k] for r in rows)) for k in
               ('concordant', 'discordant', 'total', 'cost_ties', 'distance_ties', 'both_ties')}
        num = np.array([r['concordance']['concordant'] for r in rows])[sampled].sum(axis=1)
        den = np.array([r['concordance']['total'] for r in rows])[sampled].sum(axis=1)
        regrets = [float(r['regret_mm']) for r in rows]
        out[s] = dict(**con, fraction=divide(con['concordant'], con['total']),
                      bootstrap_ci95=np.quantile(num / den, [.025, .975]).tolist(),
                      per_instance_fraction=[r['concordance']['fraction'] for r in rows],
                      selected=[r['selected'] for r in rows], regret_mm=regrets,
                      n_regret0=int(sum(r == 0 for r in regrets)), n_instances=n)
    return out


def score(insts, idxs, D, costs, cfg):
    """Census of dC = C(a) - C(neutral0) and pooled rankings for per-instance cost vectors."""
    census, ranks = {}, []
    for inst, idx, d, C in zip(insts, idxs, D, costs):
        add_census(census, sign_census(C - C[idx['neutral']], idx))
        ranks.append(rank_instance(inst, idx, C, d))
    return dict(sign_census_ge_1cm=census, rankings=pool_rankings(ranks, cfg))


def corrected_costs(insts, idxs, field, pick=None):
    """C_corr = K_f[i, :, i] - b_i with b_i the mean over foreign goals, or (pick = k)
    the single foreign goal j = (i + k) mod 8."""
    out, base = [], []
    for i, (inst, idx) in enumerate(zip(insts, idxs)):
        F = foreign_response(inst, i, field, idx['neutral'])
        b = baseline_mean(F) if pick is None else F[:, (i + pick) % len(insts)]
        base.append(b)
        out.append(own_cost(inst, i, field) - b)
    return out, base


# ----------------------------------------------------------------------------- summaries

def spread(v):
    v = np.asarray(list(v), float)
    return dict(n=int(v.size), median=float(np.median(v)), mean=float(v.mean()),
                min=float(v.min()), max=float(v.max()))


def neg(v):
    v = np.asarray(v, float)
    return dict(n=int(v.size), negative=int((v < 0).sum()), positive=int((v > 0).sum()),
                zero=int((v == 0).sum()), fraction_negative=float((v < 0).mean()))


def ranks_average(v):
    """Average ranks (ties share the mean rank)."""
    v = np.asarray(v, float)
    order = np.argsort(v, kind='stable')
    r = np.empty(len(v))
    r[order] = np.arange(len(v), dtype=float)
    _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, r)
    return (sums / counts)[inv]


def correlations(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return dict(pearson=float(np.corrcoef(a, b)[0, 1]),
                spearman=float(np.corrcoef(ranks_average(a), ranks_average(b))[0, 1]))


def foreign_table(insts, idxs, ee_goal, field):
    """One record per (instance, candidate, foreign goal): F_f, F_real, dDf_mm.

    dDf(a, j) = |ee_i(a) - ee_j(goal)| - |ee_i(neutral0) - ee_j(goal)|   (mm)
    """
    rows = []
    for i, (inst, idx) in enumerate(zip(insts, idxs)):
        Fp = foreign_response(inst, i, field, idx['neutral'])
        Fr = foreign_response(inst, i, 'real', idx['neutral'])
        dist = np.linalg.norm(inst['ee'][:, None, :] - ee_goal[None, :, :], axis=2) * 1000
        dDf = dist - dist[idx['neutral']][None, :]
        for a, name in enumerate(inst['names']):
            for j in range(len(insts)):
                if j != i:
                    rows.append(dict(instance=i, a=a, name=name, cls=inst['classes'][a],
                                     commanded_mm=inst['commanded_mm'][a], j=j, big=a in idx['big'],
                                     F_pred=float(Fp[a, j]), F_real=float(Fr[a, j]), dDf_mm=float(dDf[a, j]),
                                     dC_pred=float(inst['K_' + field][a, i] - inst['K_' + field][idx['neutral'], i]),
                                     dC_real=float(inst['K_real'][a, i] - inst['K_real'][idx['neutral'], i])))
    return rows


def foreign_census(rows, n_instances):
    """Sign census of F over the (candidate >= 1 cm, foreign goal) pairs, with splits."""
    big = [r for r in rows if r['big']]
    out = dict(n_pairs=len(big), F_pred=neg([r['F_pred'] for r in big]), F_real=neg([r['F_real'] for r in big]),
               median_F_pred=float(np.median([r['F_pred'] for r in big])),
               median_F_real=float(np.median([r['F_real'] for r in big])))
    cand = {(r['instance'], r['a']): r for r in big}.values()  # one record per candidate for own-goal dC
    out['own_goal_scale'] = dict(n_candidates=len(cand),
                                 median_dC_pred=float(np.median([r['dC_pred'] for r in cand])),
                                 median_dC_real=float(np.median([r['dC_real'] for r in cand])),
                                 median_abs_dC_pred=float(np.median([abs(r['dC_pred']) for r in cand])),
                                 median_abs_dC_real=float(np.median([abs(r['dC_real']) for r in cand])))
    split = []
    for c in CLASSES:
        for mm in (10, 30):
            sel = [r for r in big if r['cls'] == c and round(r['commanded_mm']) == mm]
            cs = {(r['instance'], r['a']): r for r in sel}.values()
            split.append(dict(cls=c, commanded_mm=mm, n_pairs=len(sel), n_candidates=len(cs),
                              F_pred_negative=int(sum(r['F_pred'] < 0 for r in sel)),
                              F_real_negative=int(sum(r['F_real'] < 0 for r in sel)),
                              median_F_pred=float(np.median([r['F_pred'] for r in sel])),
                              median_F_real=float(np.median([r['F_real'] for r in sel])),
                              median_own_dC_pred=float(np.median([r['dC_pred'] for r in cs])),
                              median_own_dC_real=float(np.median([r['dC_real'] for r in cs]))))
    out['by_class_amplitude'] = split
    out['by_class'] = [dict(cls=c, n_pairs=sum(s['n_pairs'] for s in split if s['cls'] == c),
                            F_pred_negative=sum(s['F_pred_negative'] for s in split if s['cls'] == c),
                            F_real_negative=sum(s['F_real_negative'] for s in split if s['cls'] == c))
                       for c in CLASSES]
    out['by_amplitude'] = [dict(commanded_mm=mm, n_pairs=sum(s['n_pairs'] for s in split if s['commanded_mm'] == mm),
                                F_pred_negative=sum(s['F_pred_negative'] for s in split if s['commanded_mm'] == mm),
                                F_real_negative=sum(s['F_real_negative'] for s in split if s['commanded_mm'] == mm))
                           for mm in (10, 30)]
    for key, group in (('per_instance', 'instance'), ('per_foreign_goal', 'j')):
        out[key] = []
        for i in range(n_instances):
            sel = [r for r in big if r[group] == i]
            out[key].append({group: i, 'n_pairs': len(sel),
                             'F_pred_negative': int(sum(r['F_pred'] < 0 for r in sel)),
                             'F_real_negative': int(sum(r['F_real'] < 0 for r in sel)),
                             'F_pred_fraction': float(np.mean([r['F_pred'] < 0 for r in sel])),
                             'F_real_fraction': float(np.mean([r['F_real'] < 0 for r in sel]))})
    return out


def physical_approach(rows):
    """(4e) sign agreement and correlation of F with dDf over the >= 1 cm pairs."""
    big = [r for r in rows if r['big']]
    d = np.array([r['dDf_mm'] for r in big])
    out = dict(n_pairs=len(big), dDf_negative=int((d < 0).sum()), dDf_zero=int((d == 0).sum()))
    for f in ('F_pred', 'F_real'):
        v = np.array([r[f] for r in big])
        agree = int((np.sign(v) == np.sign(d)).sum())
        out[f] = dict(sign_agree=agree, fraction=agree / len(big),
                      negative_given_dDf_negative=[int(((v < 0) & (d < 0)).sum()), int((d < 0).sum())],
                      negative_given_dDf_positive=[int(((v < 0) & (d > 0)).sum()), int((d > 0).sum())],
                      **correlations(v, d))
    return out


def descriptive_by_group(insts, keys):
    """Summaries of per-candidate quantities by (class, commanded mm)."""
    groups = {}
    for inst in insts:
        for a, c in enumerate(inst['classes']):
            g = groups.setdefault((CLASS_ID[c], c, round(inst['commanded_mm'][a], 3)), {k: [] for k in keys})
            for k in keys:
                g[k].append(float(inst[k][a]))
    return [dict(cls=c, commanded_mm=mm, **{k: spread(v[k]) for k in keys})
            for (_, c, mm), v in sorted(groups.items())]


def moving_reference_census(insts, idxs):
    """(4c) own-goal census of C_f(a) - C_f(onray-plus-0.001) for the >= 1 cm candidates."""
    out = {}
    for f in ('pred_cmd', 'pred_exec', 'real'):
        census = {}
        for i, (inst, idx) in enumerate(zip(insts, idxs)):
            C = own_cost(inst, i, f)
            add_census(census, sign_census(C - C[idx['moving_ref']], idx))
        out[f] = census
    return out


def range_of(values):
    v = list(values)
    return dict(values=v, min=min(v), median=float(np.median(v)), max=max(v))


# ----------------------------------------------------------------------------- guard

def guard(raw, cfg):
    """E001 reference numbers reproduced through the scoring path used for C_corr."""
    def falls(f):
        return {c: [raw[f]['sign_census_ge_1cm'][c]['falls'], raw[f]['sign_census_ge_1cm'][c]['n']] for c in CLASSES}
    got = dict(pred_cmd_falls=falls('pred_cmd'), real_falls=falls('real'),
               pred_cmd_all=[raw['pred_cmd']['rankings']['all']['concordant'], raw['pred_cmd']['rankings']['all']['total']],
               pred_cmd_all_ci95=[round(v, 6) for v in raw['pred_cmd']['rankings']['all']['bootstrap_ci95']],
               real_all=[raw['real']['rankings']['all']['concordant'], raw['real']['rankings']['all']['total']],
               pred_cmd_regret0=raw['pred_cmd']['rankings']['all']['n_regret0'],
               real_regret0=raw['real']['rankings']['all']['n_regret0'])
    want = dict(pred_cmd_falls=dict(toward=[16, 16], away=[15, 16], lateral=[63, 64]),
                real_falls=dict(toward=[16, 16], away=[0, 16], lateral=[28, 64]),
                pred_cmd_all=[515, 1088], pred_cmd_all_ci95=[0.454963, 0.497243],
                real_all=[813, 1088], pred_cmd_regret0=7, real_regret0=8)
    ok = got == want
    return dict(passed=ok, measured=got, expected=want,
                pred_cmd_all_fraction=raw['pred_cmd']['rankings']['all']['fraction'],
                real_all_fraction=raw['real']['rankings']['all']['fraction'],
                pred_cmd_falls_total=[sum(v[0] for v in got['pred_cmd_falls'].values()),
                                      sum(v[1] for v in got['pred_cmd_falls'].values())])


# ----------------------------------------------------------------------------- outputs

def token(v):
    if isinstance(v, float):
        return 'nan' if np.isnan(v) else format(v, '.17g')
    return str(v)


def write_dat(path, columns, rows):
    path.write_text(' '.join(columns) + '\n' + ''.join(' '.join(token(r[c]) for c in columns) + '\n' for r in rows))


XPOS = {('toward', 10): 1.0, ('toward', 30): 2.0, ('away', 10): 3.0, ('away', 30): 4.0,
        ('lateral', 10): 5.0, ('lateral', 30): 6.0}


def corrected_rows(insts, idxs, D, base, base_real):
    """One record per instance x candidate for corrected.dat."""
    rows = []
    for i, (inst, idx) in enumerate(zip(insts, idxs)):
        dp = own_cost(inst, i, 'pred_cmd') - own_cost(inst, i, 'pred_cmd')[idx['neutral']]
        dr = own_cost(inst, i, 'real') - own_cost(inst, i, 'real')[idx['neutral']]
        for a, name in enumerate(inst['names']):
            c, mm = inst['classes'][a], inst['commanded_mm'][a]
            rows.append(dict(instance=i, name=name, **{'class': c}, cls_id=CLASS_ID[c], commanded_mm=mm,
                             D_mm=float(D[i][a]), dC_pred=float(dp[a]), b=float(base[i][a]),
                             dC_corr=float(dp[a] - base[i][a]), dC_real=float(dr[a]),
                             b_real=float(base_real[i][a]), dC_real_corr=float(dr[a] - base_real[i][a]),
                             xpos=XPOS.get((c, round(mm)), float('nan')) if a in idx['big'] else float('nan')))
    return rows


FIGURE = r"""\documentclass[border=4pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\usepackage{pgfplots}
\pgfplotsset{compat=1.17}
\definecolor{real}{RGB}{0,114,178}
\definecolor{pred}{RGB}{213,94,0}
\definecolor{corr}{RGB}{0,158,115}
\definecolor{ctoward}{RGB}{0,0,0}
\definecolor{caway}{RGB}{204,121,167}
\definecolor{clateral}{RGB}{86,180,233}
\pgfplotsset{
  every axis/.append style={
    tick label style={font=\large}, label style={font=\large},
    yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=2},
    xticklabel style={/pgf/number format/fixed,/pgf/number format/precision=2},
    scaled ticks=false, axis line style={black!70}, tick style={black!70},
    legend style={font=\large, draw=none, fill=white, fill opacity=0.85, text opacity=1},
    legend cell align=left, unbounded coords=discard, filter discard warning=false}}
\begin{document}
\begin{tikzpicture}
\begin{axis}[name=left, width=11.5cm, height=10.5cm,
  xmin=@LO@, xmax=@HI@, ymin=@YLO@, ymax=@YHI@,
  xlabel={realized response to a foreign goal, $F_{\mathrm{real}}$ (mean $|\Delta z|$)},
  ylabel={predicted response to a foreign goal, $F_{\mathrm{pred}}$ (mean $|\Delta z|$)},
  grid=major, grid style={black!8}, legend pos=north west]
\draw[black!55, line width=0.6pt] (axis cs:@LO@,0) -- (axis cs:@HI@,0);
\draw[black!55, line width=0.6pt] (axis cs:0,@YLO@) -- (axis cs:0,@YHI@);
\addplot[forget plot, black!35, dashed, domain=@LO@:@HI@, samples=2] {x};
\addplot[only marks, mark=*, mark size=2.2pt, clateral, opacity=0.55, draw opacity=0,
  restrict expr to domain={\thisrow{grp}}{3:3}] table[x=F_real, y=F_pred]{foreign_response.dat};
\addlegendentry{lateral}
\addplot[only marks, mark=*, mark size=2.2pt, caway, opacity=0.7, draw opacity=0,
  restrict expr to domain={\thisrow{grp}}{2:2}] table[x=F_real, y=F_pred]{foreign_response.dat};
\addlegendentry{away from own goal}
\addplot[only marks, mark=*, mark size=2.2pt, ctoward, opacity=0.6, draw opacity=0,
  restrict expr to domain={\thisrow{grp}}{1:1}] table[x=F_real, y=F_pred]{foreign_response.dat};
\addlegendentry{toward own goal}
\node[font=\large, anchor=north east, align=right] at (rel axis cs:0.985,0.985)
  {@NPAIRS@ pairs, $\geq$ 1 cm\\$F_{\mathrm{pred}}<0$: @NPRED@\\$F_{\mathrm{real}}<0$: @NREAL@};
\end{axis}
\begin{axis}[name=right, at={(left.east)}, anchor=west, xshift=3.0cm, width=14.5cm, height=10.5cm,
  xmin=0.4, xmax=6.6, xtick={1,2,3,4,5,6},
  xticklabels={10,30,10,30,10,30},
  xlabel={commanded displacement (mm)},
  ylabel={own-goal cost change vs.\ no motion, $\Delta C$ (mean $|\Delta z|$)},
  ymajorgrids, grid style={black!8}, legend pos=north west, clip=false,
  extra x ticks={1.5,3.5,5.5}, extra x tick labels={toward goal, away from goal, lateral},
  extra x tick style={tick style={draw=none}, tick label style={font=\large, yshift=4pt, anchor=south},
    ticklabel pos=upper, major tick length=0pt}]
\draw[black!55, line width=0.6pt] (axis cs:0.4,0) -- (axis cs:6.6,0);
\draw[black!25] (rel axis cs:0.3387,0) -- (rel axis cs:0.3387,1);
\draw[black!25] (rel axis cs:0.6613,0) -- (rel axis cs:0.6613,1);
\addplot[forget plot, only marks, mark=square*, mark size=2.2pt, pred, opacity=0.45, draw opacity=0]
  table[x expr=\thisrow{xpos}-0.25, y=dC_pred]{corrected.dat};
\addplot[forget plot, only marks, mark=triangle*, mark size=2.8pt, corr, opacity=0.45, draw opacity=0]
  table[x expr=\thisrow{xpos}, y=dC_corr]{corrected.dat};
\addplot[forget plot, only marks, mark=*, mark size=2.2pt, real, opacity=0.45, draw opacity=0]
  table[x expr=\thisrow{xpos}+0.25, y=dC_real]{corrected.dat};
\addplot[only marks, mark=square*, mark size=4pt, pred, mark options={draw=black, line width=0.6pt}]
  table[x expr=\thisrow{xpos}-0.25, y=dC_pred]{fig_goal_free_medians.dat};
\addlegendentry{predicted, raw}
\addplot[only marks, mark=triangle*, mark size=5pt, corr, mark options={draw=black, line width=0.6pt}]
  table[x expr=\thisrow{xpos}, y=dC_corr]{fig_goal_free_medians.dat};
\addlegendentry{predicted, baseline subtracted}
\addplot[only marks, mark=*, mark size=4pt, real, mark options={draw=black, line width=0.6pt}]
  table[x expr=\thisrow{xpos}+0.25, y=dC_real]{fig_goal_free_medians.dat};
\addlegendentry{realized}
\end{axis}
\end{tikzpicture}
\end{document}
"""


def write_figure(out, frows, crows, census):
    """Two panels read from the .dat tables; medians (large outlined marks) from a small table."""
    med = []
    for (c, mm), x in sorted(XPOS.items(), key=lambda kv: kv[1]):
        sel = [r for r in crows if r['xpos'] == x]
        med.append(dict(xpos=x, **{'class': c}, commanded_mm=float(mm), n=len(sel),
                        **{k: float(np.median([r[k] for r in sel])) for k in ('dC_pred', 'dC_corr', 'dC_real')}))
    write_dat(out / 'fig_goal_free_medians.dat', ['xpos', 'class', 'commanded_mm', 'n', 'dC_pred', 'dC_corr', 'dC_real'], med)
    big = [r for r in frows if r['big']]
    v = np.array([[r['F_pred'], r['F_real']] for r in big])
    pad = 0.06 * (v.max() - v.min())
    lo, hi = np.floor((v[:, 1].min() - pad) * 200) / 200, np.ceil((v[:, 1].max() + pad) * 200) / 200
    ylo, yhi = np.floor((v[:, 0].min() - pad) * 200) / 200, np.ceil((max(v[:, 0].max(), 0.0) + 6 * pad) * 200) / 200
    tex = (FIGURE.replace('@LO@', f'{lo:.3f}').replace('@HI@', f'{hi:.3f}')
           .replace('@YLO@', f'{ylo:.3f}').replace('@YHI@', f'{yhi:.3f}')
           .replace('@NPAIRS@', str(census['n_pairs']))
           .replace('@NPRED@', f"{census['F_pred']['negative']}/{census['n_pairs']}")
           .replace('@NREAL@', f"{census['F_real']['negative']}/{census['n_pairs']}"))
    (out / 'fig-goal-free.tex').write_text(tex)
    run = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', 'fig-goal-free.tex'],
                         cwd=out, capture_output=True, text=True)
    if run.returncode != 0:
        print(run.stdout[-3000:])
        return 'pdflatex failed'
    for ext in ('aux', 'log'):
        (out / f'fig-goal-free.{ext}').unlink(missing_ok=True)
    if shutil.which('pdftoppm'):
        subprocess.run(['pdftoppm', '-png', '-r', '200', '-singlefile', 'fig-goal-free.pdf', 'fig-goal-free'], cwd=out, check=True)
        return 'pdftoppm'
    if shutil.which('gs'):
        subprocess.run(['gs', '-q', '-dNOPAUSE', '-dBATCH', '-sDEVICE=png16m', '-r200', '-dTextAlphaBits=4',
                        '-dGraphicsAlphaBits=4', '-sOutputFile=fig-goal-free.png', 'fig-goal-free.pdf'], cwd=out, check=True)
        return 'ghostscript'
    try:
        import fitz  # type: ignore
        fitz.open(out / 'fig-goal-free.pdf')[0].get_pixmap(dpi=200).save(out / 'fig-goal-free.png')
        return 'pymupdf'
    except ImportError:
        return 'no png converter available'


# ----------------------------------------------------------------------------- main

def predicted_block(insts, idxs, D, ee_goal, field, cfg):
    """Quantities (1)-(3) for one predicted field: foreign census, C_corr census and rankings."""
    frows = foreign_table(insts, idxs, ee_goal, field)
    costs, base = corrected_costs(insts, idxs, field)
    return frows, base, dict(foreign_response=foreign_census(frows, len(insts)),
                             corrected=score(insts, idxs, D, costs, cfg))


def predictions(block, cfg):
    """Pre-registered thresholds (episode.md) against the measured values."""
    fr, sc = block['foreign_response'], block['corrected']
    cen, allr = sc['sign_census_ge_1cm'], sc['rankings']['all']
    n = fr['n_pairs']
    lo, hi = cfg.p1_freal_neg_range
    p = {
        'P1_F_pred_negative': dict(threshold=f'>= {cfg.p1_fpred_neg_min:.0%} of {n}', count=fr['F_pred']['negative'], n=n,
                                   measured=fr['F_pred']['fraction_negative'],
                                   met=bool(fr['F_pred']['fraction_negative'] >= cfg.p1_fpred_neg_min)),
        'P1_F_real_negative': dict(threshold=f'{lo:.0%}-{hi:.0%} of {n}', count=fr['F_real']['negative'], n=n,
                                   measured=fr['F_real']['fraction_negative'],
                                   met=bool(lo <= fr['F_real']['fraction_negative'] <= hi)),
        'P2_away_rises': dict(threshold=f'>= {cfg.p2_away_rise_min} of 16', measured=cen['away']['rises'], n=cen['away']['n'],
                              met=bool(cen['away']['rises'] >= cfg.p2_away_rise_min)),
        'P2_lateral_rises': dict(threshold=f'>= {cfg.p2_lateral_rise_min} of 64', measured=cen['lateral']['rises'],
                                 n=cen['lateral']['n'], met=bool(cen['lateral']['rises'] >= cfg.p2_lateral_rise_min)),
        'P2_toward_falls': dict(threshold=f'{cfg.p2_toward_fall_required} of 16', measured=cen['toward']['falls'],
                                n=cen['toward']['n'], met=bool(cen['toward']['falls'] >= cfg.p2_toward_fall_required)),
        'P3_concordance_all': dict(threshold=f'>= {cfg.p3_concordance_min:.0%}', count=allr['concordant'], n=allr['total'],
                                   measured=allr['fraction'], bootstrap_ci95=allr['bootstrap_ci95'],
                                   met=bool(allr['fraction'] >= cfg.p3_concordance_min)),
        'P3_regret0': dict(threshold=f'{cfg.p3_regret0_required} of 8', measured=allr['n_regret0'], n=allr['n_instances'],
                           met=bool(allr['n_regret0'] >= cfg.p3_regret0_required)),
    }
    p['P1_met'] = p['P1_F_pred_negative']['met'] and p['P1_F_real_negative']['met']
    p['P2_met'] = p['P2_away_rises']['met'] and p['P2_lateral_rises']['met'] and p['P2_toward_falls']['met']
    p['P3_met'] = p['P3_concordance_all']['met'] and p['P3_regret0']['met']
    return p


def main():
    cfg = Config()
    out = ROOT / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    goals, ee_goal = load_goals(cfg)
    insts = [read_instance(p, goals) for p in instance_paths(cfg)]
    del goals
    idxs = [index_sets(inst, cfg) for inst in insts]
    D = [np.linalg.norm(inst['ee'] - inst['ee'][idx['goal']], axis=1) * 1000 for inst, idx in zip(insts, idxs)]
    for inst, idx in zip(insts, idxs):
        assert inst['names'] == insts[0]['names'], 'candidate order differs between instances'
        assert [len(idx[c]) for c in CLASSES] == [2, 2, 8], 'unexpected >= 1 cm population'

    # raw costs through the scoring path, then the guard
    raw = {f: score(insts, idxs, D, [own_cost(inst, i, f) for i, inst in enumerate(insts)], cfg) for f in FIELDS}
    g = guard(raw, cfg)
    if not g['passed']:
        print(json.dumps(g, indent=1))
        raise SystemExit('GUARD FAILED: E001 reference numbers not reproduced; nothing written.')

    # (1)-(3) for the commanded-input prediction, and the executed-input shift
    frows, base, main_block = predicted_block(insts, idxs, D, ee_goal, 'pred_cmd', cfg)
    frows_exec, _, exec_block = predicted_block(insts, idxs, D, ee_goal, 'pred_exec', cfg)

    # (4d) the same correction on the realized cost
    costs_real, base_real = corrected_costs(insts, idxs, 'real')
    real_corr = score(insts, idxs, D, costs_real, cfg)

    # (4f) single-foreign-goal baselines, j = (i + k) mod 8
    single = {}
    for f in ('pred_cmd', 'pred_exec'):
        variants = []
        for k in range(1, len(insts)):
            s = score(insts, idxs, D, corrected_costs(insts, idxs, f, pick=k)[0], cfg)
            c, r = s['sign_census_ge_1cm'], s['rankings']['all']
            variants.append(dict(k=k, away_rises=c['away']['rises'], lateral_rises=c['lateral']['rises'],
                                 toward_falls=c['toward']['falls'], concordant=r['concordant'], total=r['total'],
                                 fraction=r['fraction'], n_regret0=r['n_regret0'], selected=r['selected'],
                                 max_regret_mm=max(r['regret_mm'])))
        single[f] = dict(enumeration='baseline b_i(a) = F(a, j) with j = (i + k) mod 8, k = 1..7', variants=variants,
                         **{key: range_of(v[key] for v in variants) for key in
                            ('away_rises', 'lateral_rises', 'toward_falls', 'concordant', 'fraction', 'n_regret0')})

    b_big = [abs(base[i][a]) for i, idx in enumerate(idxs) for a in idx['big']]
    result = dict(
        episode='E003-goal-free-term', config=asdict(cfg),
        populations=dict(instances=len(insts), candidates_per_instance=len(insts[0]['names']),
                         candidates_ge_1cm=sum(len(idx['big']) for idx in idxs),
                         foreign_pairs_ge_1cm=sum(r['big'] for r in frows),
                         classes_ge_1cm={c: sum(len(idx[c]) for idx in idxs) for c in CLASSES},
                         all_scope_pairs='17 candidates per instance (all but neutral1 and goal): 136 pairs x 8 = 1088'),
        guard=g,
        raw={f: raw[f] for f in FIELDS},
        pred_cmd=main_block,
        predictions=predictions(main_block, cfg),
        baseline_scale=dict(median_abs_b_ge_1cm=float(np.median(b_big)), n=len(b_big)),
        descriptive=dict(
            a_prediction_error=dict(definition='err_f(a) = mean |z_pred_f(a) - z_real(a)| over all entries (float64); '
                                               'summary over the candidates of the 8 instances in each (class, commanded mm) group',
                                    by_group=descriptive_by_group(insts, ['err_pred_cmd', 'err_pred_exec'])),
            b_dispersion=dict(definition='token_std = mean over the 1664 channels of the population std over the 600 tokens (2x15x20); '
                                         'mean_abs = mean |z| over all entries; token_l1 = mean over tokens of sum_c |z| (= 1664 * mean_abs)',
                              by_group=descriptive_by_group(insts, [f'{k}_{f}' for f in ('pred_cmd', 'real', 'pred_exec')
                                                                    for k in ('token_std', 'mean_abs', 'token_l1')])),
            c_moving_reference=dict(definition=f'dC(a) = C_f(a) - C_f({cfg.moving_reference}), own goal, commands >= 1 cm, same field f for both terms',
                                    census=moving_reference_census(insts, idxs)),
            d_realized_corrected=dict(definition='C_real_corr(a) = C(z_real_i(a), g_i) - mean_j F_real(a, j)', **real_corr),
            e_physical_approach=dict(definition='dDf(a, j) = |ee_i(a) - ee_j(goal)| - |ee_i(neutral0) - ee_j(goal)| (mm); pairs with commands >= 1 cm',
                                     pred_cmd=physical_approach(frows),
                                     pred_exec=physical_approach(frows_exec)['F_pred']),
            f_shifted=dict(single_foreign_goal=single, pred_exec=exec_block,
                           pred_exec_predictions=predictions(exec_block, cfg))))
    (out / 'result.json').write_text(json.dumps(result, indent=1, allow_nan=False) + '\n')

    for r in frows:
        r['class'], r['cls_id'] = r['cls'], CLASS_ID[r['cls']]
        r['grp'] = CLASS_ID[r['cls']] if r['big'] else 0
    write_dat(out / 'foreign_response.dat',
              ['instance', 'name', 'class', 'commanded_mm', 'j', 'F_pred', 'F_real', 'dDf_mm', 'cls_id', 'grp'], frows)
    crows = corrected_rows(insts, idxs, D, base, base_real)
    write_dat(out / 'corrected.dat', ['instance', 'name', 'class', 'cls_id', 'commanded_mm', 'D_mm', 'dC_pred', 'b',
                                      'dC_corr', 'dC_real', 'b_real', 'dC_real_corr', 'xpos'], crows)
    png = write_figure(out, frows, crows, main_block['foreign_response'])
    print(json.dumps(dict(guard=g, predictions=result['predictions'], figure_png=png), indent=1))


if __name__ == '__main__':
    main()
