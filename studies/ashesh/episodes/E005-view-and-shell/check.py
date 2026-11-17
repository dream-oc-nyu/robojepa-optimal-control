"""E005 independent check: per-view costs, shells, weight path. CPU numpy only.

Run from the repo root: python3 episodes/E005-view-and-shell/check.py
Reads runs/E001-amplitude-sweep/scoring-input/instance-0{0..7}.npz (read only),
writes episodes/E005-view-and-shell/out/check.json.
"""
import gc
import json
import os
import sys
import time
from itertools import combinations

import numpy as np

sys.path.insert(0, 'episodes/E001-amplitude-sweep')
from readings import cost, concordance, selector, load_instance, divide  # noqa: E402

T0 = time.time()
DATA = 'runs/E001-amplitude-sweep/scoring-input/instance-%02d.npz'
OUT = 'episodes/E005-view-and-shell/out/check.json'
N = 8
VARIANTS = ['raw', 'view0', 'view1', 'real', 'real_view0', 'real_view1', 'foreign', 'disp_channel']
REAL_OF = {'raw': 'real', 'view0': 'real_view0', 'view1': 'real_view1'}
NAMES = None


def view_cost(z, g, v):
    """C_v(z, g) = mean |z[v] - g[v]| (float64), via the fixed evaluator's mean-abs."""
    return cost(z[v], g[v])


def disp_match(zp, i0):
    """z~(a) = mu(a) + (z(a) - mu(a)) * sigma(0)/sigma(a), per view and channel, over 300 tokens."""
    z0 = zp[i0].astype(np.float64)
    s0 = z0.std(axis=(1, 2), keepdims=True)  # population std, shape (2,1,1,C)
    out = {}
    for a in range(len(zp)):
        z = zp[a].astype(np.float64)
        mu = z.mean(axis=(1, 2), keepdims=True)
        s = z.std(axis=(1, 2), keepdims=True)
        f = np.ones_like(s)
        np.divide(s0, s, out=f, where=s != 0)
        out[a] = mu + (z - mu) * f
    return out


# ---- pass 1: goals of every instance (4 MB each)
goals = []
for i in range(N):
    x = load_instance(DATA % i)
    if NAMES is None:
        NAMES = [str(s) for s in x['names']]
    assert [str(s) for s in x['names']] == NAMES
    goals.append(x['z_real'][NAMES.index('goal')].copy())
    del x
    gc.collect()

I0, I1, IG = NAMES.index('neutral0'), NAMES.index('neutral1'), NAMES.index('goal')
ALL = [k for k in range(len(NAMES)) if k not in (I1, IG)]  # 17, file order
assert len(ALL) == 17


def klass(name):
    return 'toward' if name.startswith('onray-plus') else 'away' if name.startswith('onray-minus') else 'lateral'


# ---- pass 2: per-instance costs
inst = []
for i in range(N):
    x = load_instance(DATA % i)
    zp, zr, g = x['z_pred_cmd'], x['z_real'], goals[i]
    assert np.array_equal(g, zr[IG])
    D = np.linalg.norm(x['ee_world'] - x['ee_world'][IG], axis=1) * 1000.0
    cmd_mm = np.linalg.norm(x['commanded_xyz'], axis=1) * 1000.0
    kind = [str(k) for k in x['kind']]
    # predicted costs against every goal j: (19, 8) for raw / view0 / view1 (neutral1, goal rows unused)
    P = {v: np.full((len(NAMES), N), np.nan) for v in ('raw', 'view0', 'view1')}
    for a in ALL:
        for j in range(N):
            P['raw'][a, j] = cost(zp[a], goals[j])
            P['view0'][a, j] = view_cost(zp[a], goals[j], 0)
            P['view1'][a, j] = view_cost(zp[a], goals[j], 1)
    C = {v: P[v][:, i].copy() for v in P}
    # realized costs, own goal, all 19 (neutral1 needed for the noise rule)
    C['real'] = np.array([cost(zr[a], g) for a in range(len(NAMES))])
    C['real_view0'] = np.array([view_cost(zr[a], g, 0) for a in range(len(NAMES))])
    C['real_view1'] = np.array([view_cost(zr[a], g, 1) for a in range(len(NAMES))])
    # foreign-corrected
    others = [j for j in range(N) if j != i]
    Fraw = P['raw'][:, others] - P['raw'][I0, others][None, :]
    C['foreign'] = P['raw'][:, i] - Fraw.mean(axis=1)
    # dispersion-matched
    zt = disp_match(zp, I0)
    C['disp_channel'] = np.array([cost(zt[a], g) for a in range(len(NAMES))])
    del zt
    inst.append(dict(C=C, P=P, D=D, cmd_mm=cmd_mm, kind=kind, others=others))
    del x, zp, zr
    gc.collect()
    print('instance', i, 'done', round(time.time() - T0, 1), 's', flush=True)

BIG = [k for k in ALL if inst[0]['cmd_mm'][k] >= 9.99]  # goal row is not a candidate
for r in inst:
    assert [k for k in ALL if r['cmd_mm'][k] >= 9.99] == BIG
assert len(BIG) == 12 and all(NAMES[k].endswith(('-0.01', '-0.03')) for k in BIG)
SHELL = {'shell1': [k for k in range(len(NAMES)) if NAMES[k].endswith('-0.01')],
         'shell3': [k for k in range(len(NAMES)) if NAMES[k].endswith('-0.03')]}
assert all(len(s) == 6 for s in SHELL.values())
BOOT = np.random.default_rng(0).integers(0, 8, (2000, 8))


def all_scope(costs_by_inst):
    """costs_by_inst[i]: (19,) cost array. Returns concordant, total, fraction, ci95, n_regret0, regret, selected."""
    con, tot, reg, sel = [], [], [], []
    for r, c in zip(inst, costs_by_inst):
        rows = [dict(name=NAMES[a], kind=r['kind'][a], D_mm=float(r['D'][a]), C_x=float(c[a])) for a in ALL]
        s = selector(rows, 'x', float(r['D'][I0]))
        con.append(s['concordance']['concordant'])
        tot.append(s['concordance']['total'])
        reg.append(float(s['regret_mm']))
        sel.append(s['selected'])
    con, tot = np.array(con), np.array(tot)
    ratio = con[BOOT].sum(1) / tot[BOOT].sum(1)
    return dict(concordant=int(con.sum()), total=int(tot.sum()), fraction=divide(con.sum(), tot.sum()),
                ci95=[float(q) for q in np.percentile(ratio, [2.5, 97.5])],
                n_regret0=int(sum(v == 0 for v in reg)), regret_mm=reg), sel


def shells(costs_by_inst):
    out = {}
    for S, idx in SHELL.items():
        tot = dis = lat = tsel = 0
        for r, c in zip(inst, costs_by_inst):
            cs, ds = [float(c[a]) for a in idx], [float(r['D'][a]) for a in idx]
            cl = [klass(NAMES[a]) for a in idx]
            assert sorted(cl) == ['away'] + ['lateral'] * 4 + ['toward']
            nd = nl = npairs_d = npairs_l = 0
            for p, q in combinations(range(6), 2):
                sa, sb = np.sign(cs[p] - cs[q]), np.sign(ds[p] - ds[q])
                ok = int(sa == sb and sa != 0 and sb != 0)
                if cl[p] == 'lateral' and cl[q] == 'lateral':
                    nl += ok
                    npairs_l += 1
                else:
                    nd += ok
                    npairs_d += 1
            assert (npairs_d, npairs_l) == (9, 6)
            assert nd + nl == concordance(cs, ds)['concordant']
            tot += nd + nl
            dis += nd
            lat += nl
            tsel += int(cl[int(np.argmin(cs))] == 'toward')  # np.argmin: first on ties
        out[S + '_concordant'] = tot
        out[S + '_distinct_concordant'] = dis
        out[S + '_latlat_concordant'] = lat
        out[S + '_toward_selected'] = tsel
    return out


res, selected = {}, {}
for v in VARIANTS:
    cb = [r['C'][v] for r in inst]
    m = {}
    # --- A
    akeys = ['foreign_n', 'foreign_negative', 'foreign_median_1cm', 'foreign_median_3cm',
             'gain_1cm', 'gain_3cm', 'gain_n_1cm', 'gain_n_3cm']
    if v in REAL_OF:
        F1, F3 = [], []
        for r in inst:
            for a in BIG:
                f = [float(r['P'][v][a, j] - r['P'][v][I0, j]) for j in r['others']]
                (F1 if NAMES[a].endswith('-0.01') else F3).extend(f)
        assert len(F1) == len(F3) == 336
        m['foreign_n'] = len(F1) + len(F3)
        m['foreign_negative'] = int(sum(f < 0 for f in F1 + F3))
        m['foreign_median_1cm'] = float(np.median(F1))
        m['foreign_median_3cm'] = float(np.median(F3))
        for A, tag in (('0.01', '1cm'), ('0.03', '3cm')):
            ip, im = NAMES.index('onray-plus-' + A), NAMES.index('onray-minus-' + A)
            G = []
            for r in inst:
                cp, cr = r['C'][v], r['C'][REAL_OF[v]]
                odd_p = ((cp[ip] - cp[I0]) - (cp[im] - cp[I0])) / 2
                odd_r = ((cr[ip] - cr[I0]) - (cr[im] - cr[I0])) / 2
                noise = abs(cr[I1] - cr[I0])
                gval = None if (odd_r == 0 or abs(odd_r) < 10 * noise) else divide(odd_p, odd_r)
                if gval is not None:
                    G.append(gval)
            m['gain_' + tag] = float(np.median(G)) if G else None
            m['gain_n_' + tag] = len(G)
        m = {k: m[k] for k in akeys}
    else:
        m = {k: None for k in akeys}
    # --- B
    tf = ar = lr = 0
    for r in inst:
        c = r['C'][v]
        for a in BIG:
            dC = c[a] - c[I0]
            k = klass(NAMES[a])
            tf += int(k == 'toward' and dC < 0)
            ar += int(k == 'away' and dC > 0)
            lr += int(k == 'lateral' and dC > 0)
    m.update(toward_falls=tf, away_rises=ar, lateral_rises=lr)
    b, selected[v] = all_scope(cb)
    m.update(b)
    # --- C
    m.update(shells(cb))
    res[v] = m

# --- D
weight_path = []
for k in range(11):
    w0 = k / 10
    cb = [w0 * r['C']['view0'] + (1 - w0) * r['C']['view1'] for r in inst]
    b, sel = all_scope(cb)
    s = shells(cb)
    weight_path.append(dict(w0=w0, concordant=b['concordant'], n_regret0=b['n_regret0'],
                            shell1_distinct_concordant=s['shell1_distinct_concordant'],
                            shell3_distinct_concordant=s['shell3_distinct_concordant']))

# --- anchors
anchors = []


def anchor(label, got, want, tol=0):
    ok = (got == want) if tol == 0 else (got is not None and abs(got - want) <= tol)
    anchors.append(dict(label=label, got=got, want=want, ok=bool(ok)))


R = res
anchor('raw concordant', R['raw']['concordant'], 515)
anchor('raw total', R['raw']['total'], 1088)
anchor('raw n_regret0', R['raw']['n_regret0'], 7)
anchor('raw toward_falls', R['raw']['toward_falls'], 16)
anchor('raw away_rises', R['raw']['away_rises'], 1)
anchor('raw lateral_rises', R['raw']['lateral_rises'], 1)
anchor('raw gain_1cm', R['raw']['gain_1cm'], 0.221437, 1e-6)
anchor('raw gain_3cm', R['raw']['gain_3cm'], 0.264436, 1e-6)
anchor('raw gain_n_1cm', R['raw']['gain_n_1cm'], 8)
anchor('raw gain_n_3cm', R['raw']['gain_n_3cm'], 8)
anchor('real concordant', R['real']['concordant'], 813)
anchor('real n_regret0', R['real']['n_regret0'], 8)
anchor('foreign concordant', R['foreign']['concordant'], 907)
anchor('foreign toward_falls', R['foreign']['toward_falls'], 14)
anchor('foreign away_rises', R['foreign']['away_rises'], 16)
anchor('foreign lateral_rises', R['foreign']['lateral_rises'], 54)
anchor('disp_channel concordant', R['disp_channel']['concordant'], 634)
anchor('disp_channel n_regret0', R['disp_channel']['n_regret0'], 8)
anchor('disp_channel toward_falls', R['disp_channel']['toward_falls'], 16)
anchor('disp_channel away_rises', R['disp_channel']['away_rises'], 9)
anchor('disp_channel lateral_rises', R['disp_channel']['lateral_rises'], 10)
anchor('view0 foreign_n', R['view0']['foreign_n'], 672)
anchor('view0 foreign_negative', R['view0']['foreign_negative'], 655)
anchor('view0 foreign_median_1cm', R['view0']['foreign_median_1cm'], -0.00138197, 1e-8)
anchor('view0 foreign_median_3cm', R['view0']['foreign_median_3cm'], -0.00387324, 1e-8)
anchor('view1 foreign_negative', R['view1']['foreign_negative'], 672)
anchor('view1 foreign_median_1cm', R['view1']['foreign_median_1cm'], -0.01376155, 1e-8)
anchor('view1 foreign_median_3cm', R['view1']['foreign_median_3cm'], -0.02838163, 1e-8)

runtime = time.time() - T0
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(dict(variants=res, weight_path=weight_path, selected=selected, anchors=anchors,
                   anchors_ok=all(a['ok'] for a in anchors), runtime_s=runtime,
                   **res), f, indent=1)

keys = list(res['raw'].keys())
for k in keys:
    print(k)
    for v in VARIANTS:
        print('   %-13s %r' % (v, res[v][k]))
print('selected')
for v in VARIANTS:
    print('   %-13s %r' % (v, selected[v]))
print('weight_path')
for w in weight_path:
    print('  ', w)
bad = [a for a in anchors if not a['ok']]
print('anchors: %d/%d ok' % (len(anchors) - len(bad), len(anchors)))
for a in bad:
    print('  ANCHOR FAIL', a)
print('runtime_s %.1f' % runtime)
assert not bad, 'anchor mismatch'
