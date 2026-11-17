#!/usr/bin/env python3
"""Independent recomputation of the goal offsets in out/prior.json.

Does not import prior.py.  For every instance the raw units are re-read and

    S(g) = sum_k ( ||g - p_k|| - D_k )^2,   p_0 = 0, p_{k+1} = p_k + executed_k,

is minimised by a derivative-free coarse-to-fine grid search over the cube
[-1.05 D_0, 1.05 D_0]^3 (g lies on the sphere ||g|| = D_0), keeping the TOP_K best
separated cells at every level so that a mirror solution is followed as well.

Asserted:
  * unique instances: grid minimiser within TOL_G_M of prior.json's g;
  * ambiguous instances, mirror pair: grid minimiser within TOL_G_M of one of the two listed
    solutions; continuum (collinear visited points): grid minimiser is itself a solution
    (RMS below AMBIG_RMS_M), since a circle of solutions has no finite list to match;
  * every residual ||g - p_k|| - D_k at prior.json's g, recomputed here, below TOL_RES_M.
Reported only: the linearised sphere-difference solution and the grid's own ambiguity verdict.

    python3 episodes/E006-action-interface/check_prior.py [--runs ...] [--out ...]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
from pathlib import Path

import numpy as np

TOL_G_M = 0.5e-3
TOL_RES_M = 1e-3
TOP_K = 6
GRID_N = 11
FINAL_SPACING_M = 1e-7
AMBIG_SEP_M, AMBIG_RMS_M = 2e-3, 0.5e-3


def geometry(runs: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """instance -> (P, D) pooled over all its units, straight from the raw files."""
    acc: dict[int, list] = {}
    for path in sorted(runs.glob("*/inst*-seed*/unit.json")):
        u = json.loads(path.read_text())
        p, P, D = np.zeros(3), [np.zeros(3)], [u["initial_distance_m"]]
        for t in u["ticks"]:
            p = p + np.asarray(t["executed"], float)
            P.append(p.copy())
            D.append(t["distance_m"])
        acc.setdefault(u["instance"], []).append((np.array(P), np.array(D)))
    return {i: (np.vstack([a for a, _ in v]), np.concatenate([b for _, b in v])) for i, v in acc.items()}


def sumsq(G: np.ndarray, P: np.ndarray, D: np.ndarray) -> np.ndarray:
    """S(g) for candidates G (M, 3)."""
    return np.sum((np.linalg.norm(G[:, None, :] - P[None], axis=2) - D[None]) ** 2, axis=1)


def grid(center: np.ndarray, half: float, n: int) -> tuple[np.ndarray, float]:
    ax = np.linspace(-half, half, n)
    G = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3) + center
    return G, ax[1] - ax[0]


def top_separated(G: np.ndarray, S: np.ndarray, k: int, sep: float) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for i in np.argsort(S):
        if all(np.linalg.norm(G[i] - q) > sep for q in out):
            out.append(G[i])
            if len(out) == k:
                break
    return out


def grid_search(P: np.ndarray, D: np.ndarray) -> list[tuple[np.ndarray, float]]:
    """Candidates (g, rms) sorted by rms; each of TOP_K coarse cells is refined to FINAL_SPACING_M."""
    G, sp = grid(np.zeros(3), 1.05 * float(np.max(D[np.linalg.norm(P, axis=1) == 0])), 41)
    seeds = top_separated(G, sumsq(G, P, D), TOP_K, 2.5 * sp)
    out = []
    for g in seeds:
        s = sp
        while s > FINAL_SPACING_M:
            half = 2.0 * s
            G, s_new = grid(g, half, GRID_N)
            g_new = G[np.argmin(sumsq(G, P, D))]
            # shrink only when the minimiser is interior; on the window boundary, recentre
            # at the same spacing so that a cell can travel to its minimum
            interior = bool(np.all(np.abs(g_new - g) < half - 0.5 * s_new))
            g = g_new
            s = s_new if interior else s
        out.append((g, float(np.sqrt(sumsq(g[None], P, D)[0] / len(D)))))
    return sorted(out, key=lambda x: x[1])


def linearised(P: np.ndarray, D: np.ndarray) -> np.ndarray:
    """2 p_k . g = ||p_k||^2 + D_0^2 - D_k^2 (difference with the equation at p = 0)."""
    D0 = D[np.linalg.norm(P, axis=1) == 0].mean()
    return np.linalg.lstsq(2 * P, np.sum(P * P, axis=1) + D0 ** 2 - D ** 2, rcond=None)[0]


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs/E002-planner-loop"))
    ap.add_argument("--out", type=Path, default=here / "out")
    args = ap.parse_args()

    prior = json.loads((args.out / "prior.json").read_text())
    geo = geometry(args.runs)
    assert sorted(geo) == sorted(f["instance"] for f in prior["instances"]), "instance sets differ"

    rows, failures = [], []
    for f in prior["instances"]:
        inst = f["instance"]
        P, D = geo[inst]
        assert len(D) == f["n_equations"], f"inst {inst}: equation count differs"
        cands = grid_search(P, D)
        g_grid, rms_grid = cands[0]
        listed = [np.array(s["g_m"]) for s in f["solutions"]]
        d_primary = float(np.linalg.norm(g_grid - listed[0]))
        d_any = float(min(np.linalg.norm(g_grid - s) for s in listed))
        res = np.linalg.norm(listed[0] - P, axis=1) - D
        max_res = float(np.max(np.abs(res)))
        good = [g for g, r in cands if r < AMBIG_RMS_M]
        grid_ambiguous = any(np.linalg.norm(a - b) > AMBIG_SEP_M for a in good for b in good)
        g_lin = linearised(P, D)

        if f.get("ambiguity_kind") == "continuum":
            ok_g = rms_grid < AMBIG_RMS_M
        else:
            ok_g = (d_primary if f["status"] == "unique" else d_any) <= TOL_G_M
        ok_res = max_res < TOL_RES_M
        if not ok_g:
            failures.append(f"inst {inst}: grid g differs from prior.json by {1e3 * (d_primary if f['status'] == 'unique' else d_any):.3f} mm")
        if not ok_res:
            failures.append(f"inst {inst}: max residual {1e3 * max_res:.3f} mm >= 1 mm")
        rows.append({
            "instance": inst, "status_prior": f["status"], "n_equations": int(len(D)),
            "g_prior_m": listed[0].tolist(), "g_grid_m": g_grid.tolist(), "g_linearised_m": g_lin.tolist(),
            "grid_minus_prior_m": d_primary, "grid_minus_nearest_listed_m": d_any,
            "linearised_minus_prior_m": float(np.linalg.norm(g_lin - listed[0])),
            "rms_grid_m": rms_grid, "max_abs_residual_at_prior_m": max_res,
            "grid_ambiguous": bool(grid_ambiguous), "ambiguity_verdict_agrees": bool(grid_ambiguous == (f["status"] == "ambiguous")),
            "g_agrees": bool(ok_g), "residuals_below_1mm": bool(ok_res),
        })
        print(f"inst {inst} [{f['status']}]: grid g=({', '.join(f'{100 * x:+.4f}' for x in g_grid)}) cm  "
              f"|grid-prior|={1e3 * d_primary:.5f} mm  |lin-prior|={1e3 * rows[-1]['linearised_minus_prior_m']:.5f} mm  "
              f"max residual={max_res:.2e} m  grid_ambiguous={grid_ambiguous}  "
              f"{'OK' if ok_g and ok_res else 'FAIL'}")

    out = {"method": "coarse-to-fine grid search of the sum of squared sphere residuals, top-6 cells refined",
           "tolerance_g_m": TOL_G_M, "tolerance_residual_m": TOL_RES_M,
           "n_instances": len(rows), "n_pass": sum(r["g_agrees"] and r["residuals_below_1mm"] for r in rows),
           "passed": not failures, "failures": failures, "instances": rows}
    (args.out / "check.json").write_text(json.dumps(out, indent=1, allow_nan=False))
    print(f"check: {out['n_pass']}/{out['n_instances']} instances pass; wrote {args.out}/check.json")
    assert not failures, "; ".join(failures)


if __name__ == "__main__":
    main()
