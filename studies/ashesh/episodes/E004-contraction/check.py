"""E004 independent recomputation. Run: python3 episodes/E004-contraction/check.py (repo root).

Per instance i a table K_v[a, j] = cost of variant v for candidate a against goal g_j is built for
the latent variants; every metric is then arithmetic on those 19 x 8 tables.
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'episodes', 'E001-amplitude-sweep'))
from readings import cost, concordance, selector, load_instance, divide  # noqa: E402,F401

T0 = time.time()
N = 8
PATHS = [os.path.join(ROOT, 'runs', 'E001-amplitude-sweep', 'scoring-input', f'instance-0{i}.npz') for i in range(N)]
LATENT = ('raw', 'real', 'disp_channel', 'disp_scalar', 'cosine', 'real_cosine')
VARIANTS = ('raw', 'real', 'foreign', 'disp_channel', 'disp_scalar', 'cosine', 'amp_penalty', 'norm_only')


def cos_cost(z, g):
    """1 - mean over 600 tokens of cos(z_token, g_token); z, g (2, 15, 20, 1664)."""
    z = np.asarray(z, np.float64).reshape(-1, z.shape[-1])
    g = np.asarray(g, np.float64).reshape(-1, g.shape[-1])
    nz, ng = np.sqrt((z * z).sum(1)), np.sqrt((g * g).sum(1))
    assert (nz > 0).all() and (ng > 0).all()
    return float(1.0 - ((z * g).sum(1) / (nz * ng)).mean())


def stats(z):
    """mean and population std over the 300 tokens of each view, per channel -> (2, 1, 1, C)."""
    return z.mean(axis=(1, 2), keepdims=True), z.std(axis=(1, 2), keepdims=True)


# ---------------- pass 1: goals ----------------
goals = []
for p in PATHS:
    x = load_instance(p)
    gi = int(np.flatnonzero(x['names'] == 'goal')[0])
    goals.append(x['z_real'][gi].copy())
    del x

# ---------------- pass 2: cost tables ----------------
K = {v: np.zeros((N, 19, N)) for v in LATENT}
names = None
norm_m, D_mm, kinds = np.zeros((N, 19)), np.zeros((N, 19)), None
for i, p in enumerate(PATHS):
    x = load_instance(p)
    nm = [str(s) for s in x['names']]
    if names is None:
        names, kinds = nm, [str(k) for k in x['kind']]
    assert nm == names and [str(k) for k in x['kind']] == kinds
    n0, g_idx = names.index('neutral0'), names.index('goal')
    assert np.array_equal(x['z_real'][g_idx], goals[i])
    norm_m[i] = np.linalg.norm(x['commanded_xyz'], axis=1)
    D_mm[i] = np.linalg.norm(x['ee_world'] - x['ee_world'][g_idx], axis=1) * 1000
    mu0, sd0 = stats(x['z_pred_cmd'][n0].astype(np.float64))
    for a in range(19):
        zp = x['z_pred_cmd'][a]
        zr = x['z_real'][a]
        z64 = zp.astype(np.float64)
        mu, sd = stats(z64)
        fac = np.ones_like(sd)
        nzero = sd != 0
        fac[nzero] = sd0[nzero] / sd[nzero]
        z_ch = mu + (z64 - mu) * fac
        den = sd.mean(axis=-1, keepdims=True)
        assert (den > 0).all()
        z_sc = mu + (z64 - mu) * (sd0.mean(axis=-1, keepdims=True) / den)
        for j in range(N):
            g = goals[j]
            K['raw'][i, a, j] = cost(zp, g)
            K['real'][i, a, j] = cost(zr, g)
            K['disp_channel'][i, a, j] = cost(z_ch, g)
            K['disp_scalar'][i, a, j] = cost(z_sc, g)
            K['cosine'][i, a, j] = cos_cost(zp, g)
            K['real_cosine'][i, a, j] = cos_cost(zr, g)
    del x
    print(f'instance {i} tables done, {time.time() - T0:.0f}s', flush=True)

# ---------------- populations ----------------
n0, n1 = names.index('neutral0'), names.index('neutral1')
mm = np.round(norm_m * 1000, 6)
moving = [a for a in range(19) if kinds[a] in ('onray', 'offray')]
pop = [a for a in moving if (norm_m[:, a] * 1000 >= 9.99).all()]
assert all(((norm_m[:, a] * 1000 >= 9.99) == (norm_m[0, a] * 1000 >= 9.99)).all() for a in moving)
toward = [a for a in pop if names[a].startswith('onray-plus')]
away = [a for a in pop if names[a].startswith('onray-minus')]
lateral = [a for a in pop if kinds[a] == 'offray']
assert (len(pop), len(toward), len(away), len(lateral)) == (12, 2, 2, 8)
c1 = [a for a in pop if np.allclose(norm_m[:, a], 0.01, atol=2e-5)]
c3 = [a for a in pop if np.allclose(norm_m[:, a], 0.03, atol=2e-5)]
assert len(c1) == 6 and len(c3) == 6 and sorted(c1 + c3) == pop
rank_idx = [a for a in range(19) if names[a] not in ('neutral1', 'goal')]
assert len(rank_idx) == 17
RNG_IDX = np.random.default_rng(0).integers(0, 8, (2000, 8))


def rank_one(i, own):
    rows = [dict(name=names[a], kind=kinds[a], D_mm=float(D_mm[i, a]), C_x=float(own[a])) for a in rank_idx]
    return selector(rows, 'x', float(D_mm[i, n0]))


def boot(conc, tot):
    conc, tot = np.asarray(conc, float), np.asarray(tot, float)
    f = conc[RNG_IDX].sum(1) / tot[RNG_IDX].sum(1)
    return [float(v) for v in np.percentile(f, [2.5, 97.5])]


# ---------------- amplitude penalty: lambda by leave-one-instance-out ----------------
LAMS = [k / 100 for k in range(301)]
own_raw = np.stack([K['raw'][i, :, i] for i in range(N)])  # (8, 19)
conc_grid = np.zeros((301, N), int)
tot_grid = np.zeros((301, N), int)
for li, lam in enumerate(LAMS):
    for i in range(N):
        c = rank_one(i, own_raw[i] + lam * norm_m[i])['concordance']
        conc_grid[li, i], tot_grid[li, i] = c['concordant'], c['total']
lam_fold = []
for i in range(N):
    tr = [j for j in range(N) if j != i]
    frac = conc_grid[:, tr].sum(1) / tot_grid[:, tr].sum(1)
    lam_fold.append(LAMS[int(np.argmax(frac))])  # argmax returns the first = smallest lambda on ties
frac_all = conc_grid.sum(1) / tot_grid.sum(1)
best_in = int(np.argmax(frac_all))
lam_fold_arr = np.array(lam_fold)

# ---------------- own-goal costs (8, 19) and foreign responses (8, 19, 8; nan at j == i) ----------------
own, F = {}, {}
for v in LATENT:
    own[v] = np.stack([K[v][i, :, i] for i in range(N)])
    F[v] = K[v] - K[v][:, n0:n0 + 1, :]
    for i in range(N):
        F[v][i, :, i] = np.nan
own['foreign'] = own['raw'] - np.nanmean(F['raw'], axis=2)
F['foreign'] = np.full((N, 19, N), np.nan)
for i in range(N):
    for j in range(N):
        if j == i:
            continue
        others = [k for k in range(N) if k not in (i, j)]
        F['foreign'][i, :, j] = F['raw'][i, :, j] - F['raw'][i][:, others].mean(axis=1)
own['amp_penalty'] = own['raw'] + lam_fold_arr[:, None] * norm_m
F['amp_penalty'] = F['raw'] + (lam_fold_arr[:, None] * norm_m)[:, :, None]
own['norm_only'] = norm_m.copy()
F['norm_only'] = np.where(np.isnan(F['raw']), np.nan, norm_m[:, :, None] + 0 * K['raw'])


def foreign_vals(v, cands):
    vals = [F[v][i, a, j] for i in range(N) for a in cands for j in range(N) if j != i]
    assert np.isfinite(vals).all()
    return np.array(vals)


def gains(v, amp):
    ref = own['real_cosine'] if v == 'cosine' else own['real']
    p, m = names.index(f'onray-plus-{amp}'), names.index(f'onray-minus-{amp}')
    out = []
    for i in range(N):
        odd_real = ((ref[i, p] - ref[i, n0]) - (ref[i, m] - ref[i, n0])) / 2
        odd_v = ((own[v][i, p] - own[v][i, n0]) - (own[v][i, m] - own[v][i, n0])) / 2
        noise = abs(ref[i, n1] - ref[i, n0])
        if odd_real == 0 or abs(odd_real) < 10 * noise:
            continue
        out.append(odd_v / odd_real)
    return (float(np.median(out)) if out else None), len(out)


raw_med3 = float(np.median(foreign_vals('raw', c3)))
res = {}
for v in VARIANTS:
    fa = foreign_vals(v, pop)
    med1, med3 = float(np.median(foreign_vals(v, c1))), float(np.median(foreign_vals(v, c3)))
    dC = own[v] - own[v][:, n0:n0 + 1]
    g1, k1 = gains(v, '0.01')
    g3, k3 = gains(v, '0.03')
    sel = [rank_one(i, own[v][i]) for i in range(N)]
    conc = [s['concordance']['concordant'] for s in sel]
    tot = [s['concordance']['total'] for s in sel]
    res[v] = dict(
        foreign_n=int(fa.size), foreign_negative=int((fa < 0).sum()),
        foreign_median_1cm=med1, foreign_median_3cm=med3, removed_3cm=None if raw_med3 == 0 else float(1 - med3 / raw_med3),
        toward_falls=int((dC[:, toward] < 0).sum()), away_rises=int((dC[:, away] > 0).sum()),
        lateral_rises=int((dC[:, lateral] > 0).sum()),
        gain_1cm=g1, gain_3cm=g3, gain_n_1cm=k1, gain_n_3cm=k3,
        concordant=int(sum(conc)), total=int(sum(tot)), fraction=divide(sum(conc), sum(tot)),
        ci95=boot(conc, tot), n_regret0=int(sum(s['regret_mm'] == 0 for s in sel)),
        regret_mm=[float(s['regret_mm']) for s in sel])
res['amp_penalty'].update(lambdas=lam_fold, insample_lambda=LAMS[best_in], insample_fraction=float(frac_all[best_in]))

# ---------------- anchors ----------------
anchors = []


def anchor(label, got, want, tol=0):
    ok = abs(got - want) <= tol
    anchors.append(ok)
    print(f'  [{"ok" if ok else "FAIL"}] {label}: got {got}, expected {want}' + (f' +/- {tol}' if tol else ''))


print('anchors:')
r = res['raw']
anchor('raw concordant', r['concordant'], 515); anchor('raw total', r['total'], 1088)
anchor('raw n_regret0', r['n_regret0'], 7); anchor('raw toward_falls', r['toward_falls'], 16)
anchor('raw away_rises', r['away_rises'], 1); anchor('raw lateral_rises', r['lateral_rises'], 1)
anchor('raw foreign_n', r['foreign_n'], 672); anchor('raw foreign_negative', r['foreign_negative'], 672)
anchor('raw foreign_median_1cm', r['foreign_median_1cm'], -0.008, 0.0005)
anchor('raw foreign_median_3cm', r['foreign_median_3cm'], -0.0165, 0.0005)
anchor('raw gain_1cm', r['gain_1cm'], 0.2214, 0.0005); anchor('raw gain_3cm', r['gain_3cm'], 0.2644, 0.0005)
anchor('raw gain_n_1cm', r['gain_n_1cm'], 8); anchor('raw gain_n_3cm', r['gain_n_3cm'], 8)
r = res['real']
anchor('real concordant', r['concordant'], 813); anchor('real n_regret0', r['n_regret0'], 8)
anchor('real toward_falls', r['toward_falls'], 16); anchor('real away_rises', r['away_rises'], 16)
dCr = own['real'] - own['real'][:, n0:n0 + 1]
anchor('real lateral falls', int((dCr[:, lateral] < 0).sum()), 28)
r = res['foreign']
anchor('foreign concordant', r['concordant'], 907); anchor('foreign away_rises', r['away_rises'], 16)
anchor('foreign lateral_rises', r['lateral_rises'], 54); anchor('foreign toward_falls', r['toward_falls'], 14)
anchor('foreign n_regret0', r['n_regret0'], 7)
print('ALL ANCHORS REPRODUCED' if all(anchors) else 'ANCHOR FAILURE')

os.makedirs(os.path.join(HERE, 'out'), exist_ok=True)
out_path = os.path.join(HERE, 'out', 'check.json')
with open(out_path, 'w') as f:
    json.dump(res, f, indent=1)
for v in VARIANTS:
    print(v, json.dumps(res[v]))
print('lambdas per fold:', lam_fold, '| in-sample best:', LAMS[best_in], repr(float(frac_all[best_in])),
      f'({int(conc_grid[best_in].sum())}/{int(tot_grid[best_in].sum())})')
print('wrote', out_path, f'runtime {time.time() - T0:.1f}s')
assert all(anchors)
