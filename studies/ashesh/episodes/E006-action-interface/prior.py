#!/usr/bin/env python3
"""E006 prior: goal offset by trilateration, and commanded component against remaining offset.

Reads exported planner-loop units (read-only) and writes three tables.

Geometry
--------
Within one unit, p_0 = 0 and p_{k+1} = p_k + e_k (e_k = executed translation of tick k),
D_0 = initial_distance_m and D_{k+1} = distance_m logged after tick k.  All units of an
instance share one goal offset g in R^3 (frame of `executed`) with

    || g - p_k || = D_k            for every k of every unit.

g is the least-squares solution of these sphere equations.  The remaining offset before
tick k is r_k = g - p_k.  The planner's metric command is a_k; the model was shown
u = clamp(a / CLAMP_M, -1, 1) per axis, so a component is `saturated` when |a_i| > CLAMP_M.

Usage (from the repo root)
--------------------------
    python3 episodes/E006-action-interface/prior.py \
        [--runs runs/E002-planner-loop] [--out episodes/E006-action-interface/out]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

AXES = ("x", "y", "z")


@dataclass(frozen=True)
class Config:
    """Thresholds, all in metres unless stated."""

    CLAMP_M: float = 0.05            # per-axis saturation of the model-side action
    MOVED_M: float = 1e-3            # ||e_k|| above this counts as motion
    AMBIG_SEP_M: float = 2e-3        # two fits further apart than this are distinct
    AMBIG_RMS_M: float = 0.5e-3      # ... if both have RMS residual below this
    BIN_FAR_M: float = 0.03          # |r_i| >= 3 cm
    BIN_NEAR_M: float = 0.01         # |r_i| <  1 cm
    RATIO_MIN_R_M: float = 0.01      # a_i / r_i only where |r_i| >= 1 cm
    DISTINCT_POINT_M: float = 1e-3   # visited points closer than this add no geometry
    LM_ITERS: int = 300
    BASELINE_GROUP: str = "cem_release"
    EXPECT_SAT_TICKS: int = 28       # written expectation: >= 28 of 40 at 8 units
    EXPECT_SAT_DENOM: int = 40


# --------------------------------------------------------------------------- loading


def load_units(runs: Path) -> list[dict]:
    """Every runs/<group>/inst<I>-seed<S>/unit.json, with `group` taken from the path."""
    units = []
    for path in sorted(runs.glob("*/inst*-seed*/unit.json")):
        with open(path, "r") as fh:
            u = json.load(fh)
        u["group"] = path.parent.parent.name
        u["path"] = str(path)
        units.append(u)
    units.sort(key=lambda u: (u["instance"], u["group"], u["seed"]))
    return units


def unit_geometry(u: dict) -> tuple[np.ndarray, np.ndarray]:
    """Visited points P (K+1, 3) with p_0 = 0, p_{k+1} = p_k + e_k, and distances D (K+1,)."""
    e = np.array([t["executed"] for t in u["ticks"]], dtype=float)
    P = np.vstack([np.zeros(3), np.cumsum(e, axis=0)])
    D = np.array([u["initial_distance_m"]] + [t["distance_m"] for t in u["ticks"]], dtype=float)
    return P, D


# --------------------------------------------------------------------------- trilateration


def residuals(g: np.ndarray, P: np.ndarray, D: np.ndarray) -> np.ndarray:
    """f_k(g) = ||g - p_k|| - D_k."""
    return np.linalg.norm(g - P, axis=1) - D


def jacobian(g: np.ndarray, P: np.ndarray) -> np.ndarray:
    """J_k = (g - p_k)^T / ||g - p_k||  (unit vector from p_k to g)."""
    d = g - P
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)


def linearised(P: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Sphere differences against the origin equation ||g|| = D_0:

        ||g - p_k||^2 - ||g||^2 = D_k^2 - D_0^2   =>   2 p_k . g = ||p_k||^2 + D_0^2 - D_k^2.

    Minimum-norm least squares; rank-deficient when the p_k are coplanar with the origin.
    """
    D0sq = float(np.mean(D[np.linalg.norm(P, axis=1) < 1e-12] ** 2))
    A = 2.0 * P
    b = np.sum(P * P, axis=1) + D0sq - D ** 2
    g, *_ = np.linalg.lstsq(A, b, rcond=None)
    return g


def best_fit_plane(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centroid c, unit normal n (direction of least spread) and singular values of P - c."""
    c = P.mean(axis=0)
    _, s, vt = np.linalg.svd(P - c, full_matrices=False)
    return c, vt[-1], s


def mirror(g: np.ndarray, c: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Reflection of g through the plane {x : n . (x - c) = 0}."""
    return g - 2.0 * np.dot(g - c, n) * n


def levenberg_marquardt(g0: np.ndarray, P: np.ndarray, D: np.ndarray, iters: int) -> np.ndarray:
    """min_g sum_k f_k(g)^2 with steps (J^T J + lam I) dg = -J^T f, lam adapted on acceptance."""
    g = np.array(g0, dtype=float)
    f = residuals(g, P, D)
    cost = float(f @ f)
    lam = 1e-3
    for _ in range(iters):
        J = jacobian(g, P)
        H, grad = J.T @ J, J.T @ f
        accepted = False
        for _ in range(40):
            step = np.linalg.solve(H + lam * np.eye(3), -grad)
            f_new = residuals(g + step, P, D)
            cost_new = float(f_new @ f_new)
            if cost_new <= cost:
                g, f, cost, lam, accepted = g + step, f_new, cost_new, max(lam * 0.3, 1e-15), True
                break
            lam *= 10.0
        if not accepted or np.linalg.norm(step) < 1e-15:
            break
    return g


def starts(P: np.ndarray, D: np.ndarray) -> list[np.ndarray]:
    """Linearised solution, its mirror images, and 26 directions on the sphere ||g|| = D_0."""
    g_lin = linearised(P, D)
    c, n, _ = best_fit_plane(P)
    out = [g_lin, mirror(g_lin, c, n), mirror(g_lin, np.zeros(3), n), -g_lin]
    for i in range(3):
        m = g_lin.copy()
        m[i] = -m[i]
        out.append(m)
    for d in itertools.product((-1.0, 0.0, 1.0), repeat=3):
        d = np.array(d)
        if d.any():
            out.append(D[0] * d / np.linalg.norm(d))
    return out


def distinct_points(P: np.ndarray, tol: float) -> np.ndarray:
    """Greedy de-duplication of visited points at tolerance `tol`."""
    keep: list[np.ndarray] = []
    for p in P:
        if all(np.linalg.norm(p - q) > tol for q in keep):
            keep.append(p)
    return np.array(keep)


def describe_fit(g: np.ndarray, P: np.ndarray, D: np.ndarray) -> dict:
    f = residuals(g, P, D)
    return {
        "g_m": g.tolist(),
        "norm_g_m": float(np.linalg.norm(g)),
        "max_abs_residual_m": float(np.max(np.abs(f))),
        "rms_residual_m": float(np.sqrt(np.mean(f ** 2))),
        "residuals_m": f.tolist(),
        "jacobian_singular_values": np.linalg.svd(jacobian(g, P), compute_uv=False).tolist(),
    }


def fit_instance(units: list[dict], cfg: Config) -> dict:
    """Pooled multi-start fit of g; clusters of acceptable solutions decide identifiability.

    status = `unique`    one cluster with RMS < AMBIG_RMS_M
             `ambiguous` two or more such clusters more than AMBIG_SEP_M apart; two clusters are
                         the mirror pair of coplanar visited points, more than two sample the
                         circle left by collinear points (`ambiguity_kind`)
             `poor_fit`  no start reaches RMS < AMBIG_RMS_M (the sphere model does not hold)
    """
    geo = [unit_geometry(u) for u in units]
    P = np.vstack([p for p, _ in geo])
    D = np.concatenate([d for _, d in geo])
    c, n, _ = best_fit_plane(P)

    sols = [levenberg_marquardt(s, P, D, cfg.LM_ITERS) for s in starts(P, D)]
    n_starts = len(sols)
    sols += [levenberg_marquardt(mirror(s, c, n), P, D, cfg.LM_ITERS) for s in list(sols)]
    rms = [float(np.sqrt(np.mean(residuals(s, P, D) ** 2))) for s in sols]
    order = np.argsort(rms)

    clusters: list[np.ndarray] = []
    for i in order:
        if rms[i] < cfg.AMBIG_RMS_M and all(np.linalg.norm(sols[i] - q) > cfg.AMBIG_SEP_M for q in clusters):
            clusters.append(sols[i])
    status = "unique" if len(clusters) == 1 else ("ambiguous" if clusters else "poor_fit")
    if not clusters:
        clusters = [sols[order[0]]]

    Q = distinct_points(P, cfg.DISTINCT_POINT_M)
    cq, nq, sq = best_fit_plane(Q) if len(Q) > 1 else (Q[0], np.zeros(3), np.zeros(1))
    d0 = [u["initial_distance_m"] for u in units]
    g_lin = linearised(P, D)
    return {
        "instance": units[0]["instance"],
        "status": status,
        "ambiguity_kind": (None if status != "ambiguous" else "mirror_pair" if len(clusters) == 2 else "continuum"),
        "units": [{"group": u["group"], "seed": u["seed"], "calibration": bool(u["calibration"])} for u in units],
        "n_equations": int(len(D)),
        "n_distinct_points": int(len(Q)),
        "distinct_points_singular_values_m": np.asarray(sq).tolist(),
        "distinct_points_max_out_of_plane_m": float(np.max(np.abs((Q - cq) @ nq))) if len(Q) > 1 else 0.0,
        "initial_distance_spread_m": float(max(d0) - min(d0)),
        "n_starts": int(n_starts),
        "n_runs_including_mirror_restarts": int(len(sols)),
        "linearised_g_m": g_lin.tolist(),
        "linearised_minus_fit_m": float(np.linalg.norm(g_lin - clusters[0])),
        "solutions": [describe_fit(s, P, D) for s in clusters],
        **describe_fit(clusters[0], P, D),
    }


# --------------------------------------------------------------------------- per-tick rows


def cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else None


def tick_rows(u: dict, g: np.ndarray, cfg: Config, ambiguous: bool) -> tuple[list[dict], list[dict]]:
    """Per tick k with r_k = g - p_k:

        full_command_distance = || r_k - a_k ||                (compare with logged D_{k+1})
        clamp_cf_distance     = || r_k - clamp(a_k, +-CLAMP) || (counterfactual, one step)
        executed_distance     = || r_k - e_k ||                (the fitted equation k+1)
    """
    P, D = unit_geometry(u)
    key = {"instance": u["instance"], "group": u["group"], "seed": u["seed"],
           "calibration": bool(u["calibration"]), "g_ambiguous": ambiguous}
    ticks, axes = [], []
    for k, t in enumerate(u["ticks"]):
        r, a, e = g - P[k], np.array(t["commanded"], float), np.array(t["executed"], float)
        moved = bool(np.linalg.norm(e) > cfg.MOVED_M)
        conv = bool(t["blocking_converged"])
        sat = np.abs(a) > cfg.CLAMP_M
        a_clamp = np.clip(a, -cfg.CLAMP_M, cfg.CLAMP_M)
        full = float(np.linalg.norm(r - a))
        ticks.append({
            **key, "tick": k,
            "r_m": r.tolist(), "a_m": a.tolist(), "e_m": e.tolist(),
            "cos_a_r": cosine(a, r), "norm_a_m": float(np.linalg.norm(a)), "norm_r_m": float(np.linalg.norm(r)),
            "norm_e_m": float(np.linalg.norm(e)), "norm_a_minus_e_m": float(np.linalg.norm(a - e)),
            "logged_distance_before_m": float(D[k]), "logged_distance_after_m": float(D[k + 1]),
            "full_command_distance_m": full,
            "full_command_minus_logged_m": full - float(D[k + 1]),
            "executed_distance_m": float(np.linalg.norm(r - e)),
            "clamp_cf_distance_m": float(np.linalg.norm(r - a_clamp)),
            "clamp_cf_kind": "tick0_counterfactual" if k == 0 else "per_state_counterfactual_on_baseline_trajectory",
            "n_saturated_axes": int(sat.sum()), "saturated_axes": [AXES[i] for i in range(3) if sat[i]],
            "blocking_converged": conv, "moved": moved,
            "controller_sim_steps": t.get("controller_sim_steps"), "rho": t.get("rho"),
        })
        for i in range(3):
            axes.append({
                **key, "tick": k, "axis": AXES[i],
                "r_m": float(r[i]), "a_m": float(a[i]), "e_m": float(e[i]),
                "saturated": bool(sat[i]), "sign_match": bool(np.sign(a[i]) == np.sign(r[i])),
                "blocking_converged": conv, "moved": moved,
            })
    return ticks, axes


# --------------------------------------------------------------------------- aggregates


def rate(rows: list[dict]) -> dict:
    n = len(rows)
    m = sum(r["sign_match"] for r in rows)
    return {"match": m, "n": n, "rate": (m / n if n else None)}


def aggregate(ticks: list[dict], axes: list[dict], cfg: Config) -> dict:
    """Aggregates (1)-(4) over one scope.  (1) uses every tick; (2)-(4) drop instances whose
    g is not unique."""
    ok_axes = [r for r in axes if not r["g_ambiguous"]]
    ok_ticks = [t for t in ticks if not t["g_ambiguous"]]

    far = [r for r in ok_axes if abs(r["r_m"]) >= cfg.BIN_FAR_M]
    mid = [r for r in ok_axes if cfg.BIN_NEAR_M <= abs(r["r_m"]) < cfg.BIN_FAR_M]
    near = [r for r in ok_axes if abs(r["r_m"]) < cfg.BIN_NEAR_M]

    sel = [r for r in ok_axes if not r["saturated"] and abs(r["r_m"]) >= cfg.RATIO_MIN_R_M]
    a = np.array([r["a_m"] for r in sel])
    rr = np.array([r["r_m"] for r in sel])
    if len(sel):
        q1, med, q3 = np.percentile(a / rr, [25, 50, 75])
        ratio = {"n": len(sel), "median": float(med), "q1": float(q1), "q3": float(q3), "iqr": float(q3 - q1),
                 "slope_through_origin": float(a @ rr / (rr @ rr)), "n_sign_match": int(sum(r["sign_match"] for r in sel))}
    else:
        ratio = {"n": 0, "median": None, "q1": None, "q3": None, "iqr": None,
                 "slope_through_origin": None, "n_sign_match": 0}

    cons = [t for t in ok_ticks if t["moved"] and t["blocking_converged"]]
    dev = [abs(t["full_command_minus_logged_m"]) for t in cons]
    return {
        "n_units": len({(t["instance"], t["group"], t["seed"]) for t in ticks}),
        "n_instances": len({t["instance"] for t in ticks}),
        "n_ticks": len(ticks),
        "n_ticks_g_unique": len(ok_ticks),
        "saturated_ticks": {"with_saturated_axis": sum(t["n_saturated_axes"] > 0 for t in ticks), "n_ticks": len(ticks),
                            "saturated_components": sum(t["n_saturated_axes"] for t in ticks), "n_components": 3 * len(ticks)},
        "sign_match": {"r_ge_3cm": rate(far), "r_1_to_3cm": rate(mid), "r_lt_1cm": rate(near)},
        "unsaturated_ratio_a_over_r": ratio,
        "full_command_vs_logged": {"n_ticks_moved_and_converged": len(cons),
                                   "max_abs_diff_m": max(dev) if dev else None,
                                   "median_abs_diff_m": float(np.median(dev)) if dev else None},
        "tick0": [{"instance": t["instance"], "group": t["group"], "seed": t["seed"],
                   "D0_m": t["logged_distance_before_m"], "D1_logged_m": t["logged_distance_after_m"],
                   "clamp_cf_distance_m": t["clamp_cf_distance_m"], "cos_a0_g": t["cos_a_r"],
                   "saturated_axes": t["saturated_axes"]}
                  for t in ok_ticks if t["tick"] == 0],
    }


def same_seed_pairs(units: list[dict], cfg: Config) -> list[dict]:
    """(5) For two units of one instance and seed: max_i |a_i - a'_i| per tick, and at tick 0
    the signed per-axis difference with the saturation flags of both commands."""
    out = []
    by = {}
    for u in units:
        by.setdefault((u["instance"], u["seed"]), []).append(u)
    for (inst, seed), us in sorted(by.items()):
        for ua, ub in itertools.combinations(us, 2):
            A = np.array([t["commanded"] for t in ua["ticks"]], float)
            B = np.array([t["commanded"] for t in ub["ticks"]], float)
            k = min(len(A), len(B))
            diff = A[:k] - B[:k]
            out.append({
                "instance": inst, "seed": seed, "unit_a": ua["group"], "unit_b": ub["group"],
                "per_tick_max_abs_commanded_diff_mm": (1e3 * np.max(np.abs(diff), axis=1)).tolist(),
                "tick0_diff_a_minus_b_mm": (1e3 * diff[0]).tolist(),
                "tick0_saturated_a": (np.abs(A[0]) > cfg.CLAMP_M).tolist(),
                "tick0_saturated_b": (np.abs(B[0]) > cfg.CLAMP_M).tolist(),
            })
    return out


# --------------------------------------------------------------------------- output


def write_tables(out: Path, axes: list[dict], ticks: list[dict]) -> None:
    """pgfplots tables in centimetres; rows of instances without a unique g are left out."""
    b = lambda v: int(bool(v))
    with open(out / "axis_response.dat", "w") as fh:
        fh.write("instance group seed tick axis r_cm a_cm e_cm saturated sign_match moved\n")
        for r in axes:
            if not r["g_ambiguous"]:
                fh.write(f"{r['instance']} {r['group']} {r['seed']} {r['tick']} {r['axis']} "
                         f"{100 * r['r_m']:.5f} {100 * r['a_m']:.5f} {100 * r['e_m']:.5f} "
                         f"{b(r['saturated'])} {b(r['sign_match'])} {b(r['moved'])}\n")
    with open(out / "tick0.dat", "w") as fh:
        fh.write("instance group seed D0_cm D1_cm clampcf_cm cos\n")
        for t in ticks:
            if t["tick"] == 0 and not t["g_ambiguous"]:
                fh.write(f"{t['instance']} {t['group']} {t['seed']} {100 * t['logged_distance_before_m']:.5f} "
                         f"{100 * t['logged_distance_after_m']:.5f} {100 * t['clamp_cf_distance_m']:.5f} "
                         f"{num(t['cos_a_r'], '.5f')}\n")


def num(v, spec: str) -> str:
    """Format a float that may be undefined (cosine of a zero vector) as `nan`."""
    return format(v, spec) if v is not None else "nan"


def cm(v) -> str:
    return "(" + ", ".join(f"{100 * x:+.3f}" for x in v) + ")"


def print_summary(res: dict, cfg: Config) -> None:
    print(f"E006 prior | runs={res['runs']} | units={len(res['units'])} | instances={len(res['instances'])}")
    print("\n[fit] goal offset g per instance (cm), frame of `executed`")
    for f in res["instances"]:
        us = ",".join(f"{u['group']}/s{u['seed']}" for u in f["units"])
        print(f"  inst {f['instance']}: {f['status'].upper()}  g={cm(f['g_m'])}  |g|={100 * f['norm_g_m']:.4f} cm  "
              f"eqs={f['n_equations']} distinct_pts={f['n_distinct_points']}  units=[{us}]")
        print(f"      residual max={f['max_abs_residual_m']:.2e} m rms={f['rms_residual_m']:.2e} m  "
              f"sv(J)=({', '.join(f'{s:.3f}' for s in f['jacobian_singular_values'])})  "
              f"points out-of-plane max={1e3 * f['distinct_points_max_out_of_plane_m']:.2f} mm  "
              f"|g_lin-g|={1e3 * f['linearised_minus_fit_m']:.4f} mm  D0 spread={f['initial_distance_spread_m']:.1e} m")
        if f["status"] != "unique":
            if f["ambiguity_kind"]:
                print(f"      {f['ambiguity_kind']}: {len(f['solutions'])} distinct solutions found (first 4 printed, all in prior.json)")
            for j, s in enumerate(f["solutions"][:4]):
                print(f"      solution {j}: g={cm(s['g_m'])} cm  rms={s['rms_residual_m']:.2e} m  max={s['max_abs_residual_m']:.2e} m")
    excl = [f["instance"] for f in res["instances"] if f["status"] != "unique"]
    print(f"  excluded from (2)-(4) and from the .dat tables (g not unique): {excl if excl else 'none'}")

    print("\n[ticks] r = remaining offset before the tick, a = command (cm); * = saturated axis; "
          "cf = ||r - clamp(a)|| (tick 0: counterfactual; later ticks: per-state, on the baseline's trajectory)")
    print("  inst group        s k |  r_x     r_y     r_z   |  a_x      a_y      a_z     | cos    |r|    D_next ||r-a||  cf     mv cv")
    for t in res["ticks"]:
        a = "".join(f"{100 * x:+8.2f}{'*' if AXES[i] in t['saturated_axes'] else ' '}" for i, x in enumerate(t["a_m"]))
        r = "".join(f"{100 * x:+8.2f}" for x in t["r_m"])
        print(f"  {t['instance']:<4d} {t['group']:<12s} {t['seed']} {t['tick']} |{r} |{a}| {num(t['cos_a_r'], '+.3f')} "
              f"{100 * t['norm_r_m']:6.2f} {100 * t['logged_distance_after_m']:6.2f} {100 * t['full_command_distance_m']:6.2f} "
              f"{100 * t['clamp_cf_distance_m']:6.2f}  {int(t['moved'])}  {int(t['blocking_converged'])}"
              f"{'  [g ambiguous]' if t['g_ambiguous'] else ''}")

    for name, agg in res["aggregates"].items():
        print(f"\n[aggregate: {name}] units={agg['n_units']} instances={agg['n_instances']} ticks={agg['n_ticks']} "
              f"(ticks with unique g: {agg['n_ticks_g_unique']})")
        s = agg["saturated_ticks"]
        print(f"  (1) ticks with >=1 saturated component: {s['with_saturated_axis']}/{s['n_ticks']}   "
              f"saturated components: {s['saturated_components']}/{s['n_components']}   "
              f"(written expectation: >= {cfg.EXPECT_SAT_TICKS} of {cfg.EXPECT_SAT_DENOM} at 8 units)")
        sm = agg["sign_match"]
        fmt = lambda d: f"{d['match']}/{d['n']}" + (f" ({d['rate']:.2f})" if d["n"] else "")
        print(f"  (2) sign(a_i)=sign(r_i): |r_i|>=3cm {fmt(sm['r_ge_3cm'])}   1<=|r_i|<3cm {fmt(sm['r_1_to_3cm'])}   "
              f"|r_i|<1cm {fmt(sm['r_lt_1cm'])}")
        q = agg["unsaturated_ratio_a_over_r"]
        if q["n"]:
            print(f"  (3) unsaturated, |r_i|>=1cm: n={q['n']}  a_i/r_i median={q['median']:.3f} "
                  f"IQR=[{q['q1']:.3f}, {q['q3']:.3f}] (width {q['iqr']:.3f})  slope through origin={q['slope_through_origin']:.3f}  "
                  f"sign match {q['n_sign_match']}/{q['n']}")
        else:
            print("  (3) unsaturated, |r_i|>=1cm: n=0")
        c = agg["full_command_vs_logged"]
        if c["n_ticks_moved_and_converged"]:
            print(f"      consistency | ||r-a|| - D_next | on moved+converged ticks: n={c['n_ticks_moved_and_converged']} "
                  f"max={1e3 * c['max_abs_diff_m']:.2f} mm median={1e3 * c['median_abs_diff_m']:.2f} mm")
        print("  (4) tick 0:  inst group        s   D0_cm  D1_cm  clampcf_cm  cos(a0,g)  saturated")
        for t in agg["tick0"]:
            print(f"               {t['instance']:<4d} {t['group']:<12s} {t['seed']}  {100 * t['D0_m']:6.2f} {100 * t['D1_logged_m']:6.2f} "
                  f"{100 * t['clamp_cf_distance_m']:9.2f}   {num(t['cos_a0_g'], '+.3f')}     {','.join(t['saturated_axes']) or '-'}")

    print(f"\n[aggregate: same-seed pairs] (5) n_pairs={len(res['same_seed_pairs'])}")
    for p in res["same_seed_pairs"]:
        print(f"  inst {p['instance']} seed {p['seed']}: {p['unit_a']} - {p['unit_b']}")
        print("      per-tick max |commanded diff| (mm): " + ", ".join(f"{x:.3f}" for x in p["per_tick_max_abs_commanded_diff_mm"]))
        sat = ["".join(AXES[i] for i in range(3) if p[k][i]) or "-" for k in ("tick0_saturated_a", "tick0_saturated_b")]
        print("      tick 0 per-axis diff (mm): " + ", ".join(f"{AXES[i]}={x:+.3f}" for i, x in enumerate(p["tick0_diff_a_minus_b_mm"]))
              + f"   saturated axes: a={sat[0]} b={sat[1]}")


def main() -> None:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=Path("runs/E002-planner-loop"))
    ap.add_argument("--out", type=Path, default=here / "out")
    args = ap.parse_args()
    cfg = Config()

    units = load_units(args.runs)
    if not units:
        sys.exit(f"no unit.json under {args.runs}")
    by_inst: dict[int, list[dict]] = {}
    for u in units:
        by_inst.setdefault(u["instance"], []).append(u)

    fits, ticks, axes = [], [], []
    for inst in sorted(by_inst):
        fit = fit_instance(by_inst[inst], cfg)
        fits.append(fit)
        for u in by_inst[inst]:
            t, a = tick_rows(u, np.array(fit["g_m"]), cfg, ambiguous=fit["status"] != "unique")
            ticks += t
            axes += a

    base = lambda r: r["group"] == cfg.BASELINE_GROUP and not r["calibration"]
    res = {
        "runs": str(args.runs),
        "config": asdict(cfg),
        "units": [{"path": u["path"], "group": u["group"], "planner": u["planner"], "instance": u["instance"],
                   "seed": u["seed"], "calibration": bool(u["calibration"]), "n_ticks": len(u["ticks"]),
                   "initial_distance_m": u["initial_distance_m"]} for u in units],
        "instances": fits,
        "excluded_instances": [f["instance"] for f in fits if f["status"] != "unique"],
        "aggregates": {
            f"{cfg.BASELINE_GROUP}_non_calibration": aggregate([t for t in ticks if base(t)], [a for a in axes if base(a)], cfg),
            "all_units": aggregate(ticks, axes, cfg),
        },
        "same_seed_pairs": same_seed_pairs(units, cfg),
        "ticks": ticks,
        "axis_rows": axes,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "prior.json", "w") as fh:
        json.dump(res, fh, indent=1, allow_nan=False)
    write_tables(args.out, axes, ticks)
    print_summary(res, cfg)
    print(f"\nwrote {args.out}/prior.json, axis_response.dat ({sum(not r['g_ambiguous'] for r in axes)} rows), tick0.dat")


if __name__ == "__main__":
    main()
