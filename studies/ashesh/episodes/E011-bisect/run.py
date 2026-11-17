#!/usr/bin/env python3
"""E011 part A: census of saturated sign flips on exported reach units (CPU, numpy).

Geometry (per instance, pooled over arms)
-----------------------------------------
Within one unit p_0 = 0, p_{k+1} = p_k + e_k and D_0 = initial_distance_m, D_{k+1} = distance_m
after tick k.  One goal offset g in R^3 per instance solves || g - p_k || = D_k for every k of
every unit of that instance (least squares, `fit_instance` of E006 prior.py).  The remaining
offset before tick k is r_k = g - p_k.

Definitions (a = commanded, e = executed, axis i, tick t)
---------------------------------------------------------
    saturated(t, i)   |a_i(t)| >= SAT_M - SAT_TOL_M
    moved(t)          ||e(t)|| >= MOVED_M
    frozen(t)         ||a(t)|| >= FROZEN_CMD_M and ||e(t)|| < MOVED_M
    flip(t, i)        t >= 1, saturated(t-1, i), saturated(t, i),
                      sgn a_i(t-1) != sgn a_i(t), moved(t-1)
    true bracket      sgn e_i(t-1) == sgn r_i(t-1) and |e_i(t-1)| > |r_i(t-1)|
    overshoot(t-1, i) t-1 in 0..3, saturated(t-1, i), moved(t-1), sgn e_i == sgn r_i,
                      |e_i(t-1)| - |r_i(t-1)| >= OVERSHOOT_M
    logged next       | r_i(t) - e_i(t) |
    bisected next     | r_i(t) - sgn a_i(t) * |e_i(t-1)| / 2 |

Predictions (judged on instances with a unique g)
-------------------------------------------------
    A1  true brackets / flips                      >= A1_MIN
    A2  flips with |r_i(t)| >= A2_FAR_M            == 0
    A3  overshoots followed by a flip / overshoots >= A3_MIN
    A4  (bisected < logged) / flips >= A4_MIN  and  max bisected <= A4_MAX_M

Usage (from the repo root)
--------------------------
    python3 episodes/E011-bisect/run.py [--runs runs/E002-planner-loop]
        [--guardoff-runs runs/E009-guard-off] [--out episodes/E011-bisect/out]
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR_PATH = HERE.parent / "E006-action-interface" / "prior.py"
AXES = ("x", "y", "z")


@dataclass(frozen=True)
class Config:
    """Thresholds in metres unless stated."""

    SAT_M: float = 0.05              # per-axis saturation of the model-side action
    SAT_TOL_M: float = 1e-5          # saturated when |a_i| >= SAT_M - SAT_TOL_M
    MOVED_M: float = 1e-3            # tick moved when ||e|| >= MOVED_M
    FROZEN_CMD_M: float = 0.01       # frozen tick: ||a|| >= this and not moved
    A1_MIN: float = 0.85
    A2_FAR_M: float = 0.10
    OVERSHOOT_M: float = 0.02
    A3_MIN: float = 0.70
    A4_MIN: float = 0.80
    A4_MAX_M: float = 0.05
    ARM_RESIDUAL_FLAG_M: float = 1e-3  # an arm above this does not share the goal
    RUN_ARMS: tuple = ("cem_release", "cem_clampexec", "cem_cap10")
    GUARDOFF_ARMS: tuple = ("cem_release_guardoff",)
    CALIBRATION_DIR: str = "_calibration"
    EXPECT_UNITS: int = 26
    EXPECT_TICKS: int = 130
    EXPECT_TRANSITIONS: int = 104
    EXPECT_AXIS_TRANSITIONS: int = 312


def load_prior():
    """E006 prior.py loaded from its path (the directory name has a hyphen)."""
    spec = importlib.util.spec_from_file_location("e006_prior", PRIOR_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e006_prior"] = mod  # dataclass resolution needs the module registered
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- loading


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_arm(root: Path, arm: str, calibration: bool) -> list[dict]:
    units = []
    for path in sorted((root / arm).glob("inst*-seed*/unit.json")):
        u = json.loads(path.read_text())
        u["group"] = arm
        u["calibration"] = calibration   # census membership, set from the directory
        u["path"] = str(path)
        u["sha256"] = sha256(path)
        units.append(u)
    return units


def load_all(runs: Path, guardoff: Path, cfg: Config) -> tuple[list[dict], list[dict]]:
    """Census units (four arms) and geometry-only calibration units."""
    census = []
    for arm in cfg.RUN_ARMS:
        census += load_arm(runs, arm, False)
    for arm in cfg.GUARDOFF_ARMS:
        census += load_arm(guardoff, arm, False)
    return census, load_arm(runs, cfg.CALIBRATION_DIR, True)


def arrays(u: dict) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([t["commanded"] for t in u["ticks"]], dtype=float)
    e = np.array([t["executed"] for t in u["ticks"]], dtype=float)
    return a, e


# --------------------------------------------------------------------------- geometry


def fit_geometry(inst: int, census: list[dict], calib: list[dict], prior, cfg: Config) -> dict:
    """Pooled fit of g over every unit of the instance; residuals split by arm."""
    pool = [u for u in census + calib if u["instance"] == inst]
    fit = prior.fit_instance(pool, prior.Config())
    g = np.array(fit["g_m"])
    per_arm: dict[str, float] = {}
    for u in pool:
        P, D = prior.unit_geometry(u)
        m = float(np.max(np.abs(prior.residuals(g, P, D))))
        per_arm[u["group"]] = max(per_arm.get(u["group"], 0.0), m)
    return {
        "instance": inst,
        "status": fit["status"],
        "ambiguity_kind": fit["ambiguity_kind"],
        "g_cm": (100 * g).tolist(),
        "rms_residual_mm": 1e3 * fit["rms_residual_m"],
        "max_abs_residual_mm": 1e3 * fit["max_abs_residual_m"],
        "n_equations": fit["n_equations"],
        "n_distinct_points": fit["n_distinct_points"],
        "distinct_points_max_out_of_plane_mm": 1e3 * fit["distinct_points_max_out_of_plane_m"],
        "pooled_units": [u["group"] for u in pool],
        "max_abs_residual_mm_per_arm": {k: 1e3 * v for k, v in sorted(per_arm.items())},
        "arms_over_residual_flag": sorted(k for k, v in per_arm.items() if v > cfg.ARM_RESIDUAL_FLAG_M),
        "solutions_g_cm": [(100 * np.array(s["g_m"])).tolist() for s in fit["solutions"]],
        "solutions_rms_residual_mm": [1e3 * s["rms_residual_m"] for s in fit["solutions"]],
    }


# --------------------------------------------------------------------------- census


def unit_census(u: dict, g: np.ndarray | None, cfg: Config) -> dict:
    """Flips, overshoots, frozen ticks and counts of one unit.  g = None: the instance has no
    unique goal offset; flips are still listed (they need no g) with r-dependent fields None."""
    a, e = arrays(u)
    n = len(a)
    sat = np.abs(a) >= cfg.SAT_M - cfg.SAT_TOL_M
    en = np.linalg.norm(e, axis=1)
    moved = en >= cfg.MOVED_M
    frozen = (np.linalg.norm(a, axis=1) >= cfg.FROZEN_CMD_M) & ~moved
    r = None
    if g is not None:
        P = np.vstack([np.zeros(3), np.cumsum(e, axis=0)])
        r = g - P[:n]
    key = {"arm": u["group"], "instance": u["instance"], "seed": u["seed"]}
    cm = lambda v: None if v is None else float(100 * v)

    flips, overshoots = [], []
    for t in range(1, n):
        for i in range(3):
            is_flip = bool(sat[t - 1, i] and sat[t, i] and np.sign(a[t - 1, i]) != np.sign(a[t, i])
                           and moved[t - 1])
            if is_flip:
                row = {**key, "tick": t, "axis": AXES[i], "judged": r is not None,
                       "commanded_prev_cm": cm(a[t - 1, i]), "executed_prev_cm": cm(e[t - 1, i]),
                       "r_prev_cm": None, "commanded_cm": cm(a[t, i]), "executed_cm": cm(e[t, i]),
                       "r_cm": None, "true_bracket": None, "tick_t_moved": bool(moved[t]),
                       "logged_next_abs_r_cm": None, "bisected_next_abs_r_cm": None}
                if r is not None:
                    rp, ep = r[t - 1, i], e[t - 1, i]
                    row["r_prev_cm"], row["r_cm"] = cm(rp), cm(r[t, i])
                    row["true_bracket"] = bool(np.sign(ep) == np.sign(rp) and abs(ep) > abs(rp))
                    row["logged_next_abs_r_cm"] = cm(abs(r[t, i] - e[t, i]))
                    row["bisected_next_abs_r_cm"] = cm(abs(r[t, i] - np.sign(a[t, i]) * abs(ep) / 2))
                flips.append(row)
            if r is not None and sat[t - 1, i] and moved[t - 1]:
                rp, ep = r[t - 1, i], e[t - 1, i]
                if np.sign(ep) == np.sign(rp) and abs(ep) - abs(rp) >= cfg.OVERSHOOT_M:
                    kind = ("flip at t" if is_flip else
                            "unsaturated at t" if not sat[t, i] else "saturated same sign at t")
                    overshoots.append({**key, "tick_prev": t - 1, "axis": AXES[i],
                                       "commanded_prev_cm": cm(a[t - 1, i]), "executed_prev_cm": cm(ep),
                                       "r_prev_cm": cm(rp), "overshoot_cm": cm(abs(ep) - abs(rp)),
                                       "commanded_cm": cm(a[t, i]), "r_cm": cm(r[t, i]),
                                       "followed_by_flip": is_flip, "classification": kind})
    return {
        **key, "n_ticks": n, "n_transitions": n - 1, "n_axis_transitions": 3 * (n - 1),
        "n_saturated_components": int(sat.sum()), "n_components": int(sat.size),
        "flips": flips, "overshoots": overshoots,
        "first_flip_tick": min((f["tick"] for f in flips), default=None),
        "frozen_ticks": [{**key, "tick": int(t), "norm_commanded_cm": cm(np.linalg.norm(a[t])),
                          "norm_executed_cm": cm(en[t])} for t in np.flatnonzero(frozen)],
    }


def frac(num: int, den: int) -> float | None:
    return num / den if den else None


def a4_block(flips: list[dict], cfg: Config) -> dict:
    n = len(flips)
    better = sum(f["bisected_next_abs_r_cm"] < f["logged_next_abs_r_cm"] for f in flips)
    mx = max((f["bisected_next_abs_r_cm"] for f in flips), default=None)
    rate = frac(better, n)
    return {"value": rate, "numerator": better, "denominator": n, "threshold": cfg.A4_MIN,
            "max_bisected_cm": mx, "max_bisected_threshold_cm": 100 * cfg.A4_MAX_M,
            "n_bisected_above_threshold": sum(f["bisected_next_abs_r_cm"] > 100 * cfg.A4_MAX_M for f in flips),
            "fraction_clause_holds": None if rate is None else rate >= cfg.A4_MIN,
            "max_clause_holds": None if mx is None else mx <= 100 * cfg.A4_MAX_M,
            "holds": None if rate is None else bool(rate >= cfg.A4_MIN and mx <= 100 * cfg.A4_MAX_M)}


def judge(flips: list[dict], overshoots: list[dict], cfg: Config) -> dict:
    """A1-A4 on flips and overshoots of instances with a unique g.  `holds` is None when a
    ratio has an empty denominator."""
    n = len(flips)
    tb = sum(f["true_bracket"] for f in flips)
    far = [f for f in flips if abs(f["r_cm"]) >= 100 * cfg.A2_FAR_M]
    no = len(overshoots)
    cov = sum(o["followed_by_flip"] for o in overshoots)
    kinds = {k: sum(o["classification"] == k for o in overshoots)
             for k in ("flip at t", "unsaturated at t", "saturated same sign at t")}
    a1, a3 = frac(tb, n), frac(cov, no)
    return {
        "A1": {"value": a1, "numerator": tb, "denominator": n, "threshold": cfg.A1_MIN,
               "holds": None if a1 is None else a1 >= cfg.A1_MIN},
        "A2": {"value": len(far), "numerator": len(far), "denominator": n, "threshold": 0,
               "far_m": cfg.A2_FAR_M, "holds": len(far) == 0,
               "rows": [{k: f[k] for k in ("arm", "instance", "tick", "axis", "r_cm", "true_bracket")} for f in far]},
        "A3": {"value": a3, "numerator": cov, "denominator": no, "threshold": cfg.A3_MIN,
               "holds": None if a3 is None else a3 >= cfg.A3_MIN,
               "classification_counts": kinds,
               "second_clause_rest_unsaturated_holds": kinds["saturated same sign at t"] == 0},
        "A4": {**a4_block(flips, cfg),
               "on_flips_whose_tick_t_moved": a4_block([f for f in flips if f["tick_t_moved"]], cfg)},
    }


def excluded_sensitivity(census: list[dict], geometry: list[dict], cfg: Config) -> list[dict]:
    """Not judged.  For an instance without a unique g: the counts each candidate g would add."""
    out = []
    for f in geometry:
        if f["status"] == "unique":
            continue
        for g_cm, rms in zip(f["solutions_g_cm"], f["solutions_rms_residual_mm"]):
            rows = [unit_census(u, np.array(g_cm) / 100, cfg) for u in census if u["instance"] == f["instance"]]
            fl = [x for r in rows for x in r["flips"]]
            ov = [x for r in rows for x in r["overshoots"]]
            out.append({"instance": f["instance"], "candidate_g_cm": g_cm, "rms_residual_mm": rms,
                        "n_flips": len(fl), "n_true_brackets": sum(x["true_bracket"] for x in fl),
                        "n_overshoots": len(ov), "n_overshoots_followed_by_flip": sum(o["followed_by_flip"] for o in ov),
                        "overshoot_classifications": [o["classification"] for o in ov]})
    return out


def count_by(rows: list[dict], key: str, keys) -> dict:
    return {str(k): sum(r[key] == k for r in rows) for k in keys}


# --------------------------------------------------------------------------- output


def write_flips_dat(path: Path, flips: list[dict], arms: list[str]) -> None:
    """pgfplots table (cm); rows of flips without a unique g carry nan and judged = 0."""
    cols = ("commanded_prev_cm", "executed_prev_cm", "r_prev_cm", "commanded_cm", "r_cm")
    num = lambda v: "nan" if v is None else f"{v:.5f}"
    b = lambda v: "nan" if v is None else str(int(v))
    with open(path, "w") as fh:
        fh.write("# arm codes: " + ", ".join(f"{i} = {a}" for i, a in enumerate(arms)) + "; axis codes: 0 = x, 1 = y, 2 = z\n")
        fh.write("arm instance tick axis " + " ".join(cols)
                 + " true_bracket tick_t_moved logged_next_abs_r_cm bisected_next_abs_r_cm judged\n")
        for f in flips:
            fh.write(f"{arms.index(f['arm'])} {f['instance']} {f['tick']} {AXES.index(f['axis'])} "
                     + " ".join(num(f[c]) for c in cols)
                     + f" {b(f['true_bracket'])} {b(f['tick_t_moved'])} {num(f['logged_next_abs_r_cm'])} "
                       f"{num(f['bisected_next_abs_r_cm'])} {b(f['judged'])}\n")


def write_figure(out: Path, flips: list[dict]) -> dict:
    """Scatter of |r_i| after tick t (cm): x = logged, y = bisected, with y = x.  Reads flips.dat."""
    judged = [f for f in flips if f["judged"]]
    for name in ("fig-flips.pdf", "fig-flips.tex"):
        (out / name).unlink(missing_ok=True)  # never report an earlier build as this one's
    if not judged:
        return {"compiled": False, "reason": "no judged flip to draw"}
    top = max(max(f["logged_next_abs_r_cm"], f["bisected_next_abs_r_cm"]) for f in judged)
    lim = max(2.0, 2.0 * np.ceil(1.05 * top / 2.0))
    classes = [(1, "blue,mark=*,mark size=3.2pt,fill opacity=0.55,draw opacity=0.9", "True bracket"),
               (0, "red!80!black,mark=x,mark size=5.5pt,line width=1.6pt", "False flip")]
    tex = [r"\documentclass[tikz,border=8pt]{standalone}", r"\usepackage{pgfplots}",
           r"\pgfplotsset{compat=1.18}", r"\begin{document}", r"\begin{tikzpicture}",
           r"\begin{axis}[width=13cm,height=13cm,axis equal image,",
           rf"xmin=0,xmax={lim:g},ymin=0,ymax={lim:g},",
           r"xlabel={Remaining offset after tick $t$, logged step (cm)},",
           r"ylabel={Remaining offset after tick $t$, half step (cm)},",
           r"label style={font=\Large},tick label style={font=\large},",
           r"legend style={font=\large,at={(0.03,0.97)},anchor=north west},legend cell align=left,",
           r"grid=major,major grid style={gray!20}]",
           rf"\addplot[black,dashed,thick,domain=0:{lim:g},samples=2,forget plot] {{x}};"]
    for flag, style, label in classes:
        if any(f["true_bracket"] == bool(flag) for f in judged):
            tex.append(r"\addplot[only marks," + style + r",restrict expr to domain={\thisrow{judged}}{1:1},"
                       r"restrict expr to domain={\thisrow{true_bracket}}{" + f"{flag}:{flag}" + r"}] "
                       r"table[x=logged_next_abs_r_cm,y=bisected_next_abs_r_cm] {flips.dat};")
            tex.append(r"\addlegendentry{" + label + "}")
    tex += [r"\end{axis}", r"\end{tikzpicture}", r"\end{document}"]
    (out / "fig-flips.tex").write_text("\n".join(tex) + "\n")

    compiler = shutil.which("pdflatex")
    if not compiler:
        return {"compiled": False, "reason": "pdflatex not installed"}
    proc = subprocess.run([compiler, "-interaction=nonstopmode", "-halt-on-error", "fig-flips.tex"],
                          cwd=out, capture_output=True, text=True)
    pdf = out / "fig-flips.pdf"
    ok = proc.returncode == 0 and pdf.is_file()
    if ok:
        for ext in (".aux", ".log"):
            (out / f"fig-flips{ext}").unlink(missing_ok=True)
        return {"compiled": True, "returncode": 0, "pdf_bytes": pdf.stat().st_size,
                "axis_limit_cm": float(lim), "n_points": len(judged)}
    tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-25:])
    return {"compiled": False, "returncode": proc.returncode, "reason": "pdflatex failed", "log_tail": tail}


def fmt(v, spec="+.2f") -> str:
    return "n/a" if v is None else format(v, spec)


def flip_line(f: dict) -> str:
    return (f"    {f['arm']:<21s} inst {f['instance']} t={f['tick']} {f['axis']}  "
            f"a(t-1)={fmt(f['commanded_prev_cm'])} e(t-1)={fmt(f['executed_prev_cm'])} r(t-1)={fmt(f['r_prev_cm'])}  "
            f"a(t)={fmt(f['commanded_cm'])} r(t)={fmt(f['r_cm'])}  moved(t)={int(f['tick_t_moved'])}  "
            f"logged={fmt(f['logged_next_abs_r_cm'], '.2f')} bisected={fmt(f['bisected_next_abs_r_cm'], '.2f')} cm")


def print_summary(res: dict) -> None:
    p, pop = res["predictions"], res["population"]
    print(f"E011 part A | units={pop['n_units']} ticks={pop['n_ticks']} transitions={pop['n_transitions']} "
          f"axis-transitions={pop['n_axis_transitions']} | saturated components "
          f"{pop['n_saturated_components']}/{pop['n_components']}")
    print("[geometry] goal offset per instance (cm)")
    for f in res["geometry"]:
        g = ", ".join(f"{x:+.3f}" for x in f["g_cm"])
        arms = " ".join(f"{k}={v:.3f}" for k, v in f["max_abs_residual_mm_per_arm"].items())
        print(f"  inst {f['instance']}: {f['status']:<9s} g=({g}) rms={f['rms_residual_mm']:.4f} mm eqs={f['n_equations']}  "
              f"max|res| mm: {arms}" + (f"  FLAG {f['arms_over_residual_flag']}" if f["arms_over_residual_flag"] else ""))
    print("[excluded from A1-A4] " + ("; ".join(f"inst {x['instance']}: {x['reason']}" for x in res["excluded_instances"]) or "none"))
    for x in res["excluded_instance_sensitivity"]:
        g = ", ".join(f"{v:+.3f}" for v in x["candidate_g_cm"])
        print(f"  not judged: inst {x['instance']} candidate g=({g}) -> flips {x['n_flips']}, overshoots {x['n_overshoots']} "
              f"(followed by a flip: {x['n_overshoots_followed_by_flip']}) {x['overshoot_classifications']}")
    print(f"[flips] all={pop['n_flips_all']} judged={pop['n_flips_judged']} "
          f"(axis-transitions on judged instances: {pop['n_axis_transitions_judged']})")
    yn = lambda v: "n/a" if v is None else str(bool(v)).lower()
    print(f"  A1 precision      {p['A1']['numerator']}/{p['A1']['denominator']} = {fmt(p['A1']['value'], '.3f')}"
          f"  (>= {p['A1']['threshold']})  holds={yn(p['A1']['holds'])}")
    print(f"  A2 far-axis flips {p['A2']['numerator']} of {p['A2']['denominator']} flips with |r(t)| >= 10 cm"
          f"  (== 0)  holds={yn(p['A2']['holds'])}")
    c = p["A3"]["classification_counts"]
    print(f"  A3 coverage       {p['A3']['numerator']}/{p['A3']['denominator']} = {fmt(p['A3']['value'], '.3f')}"
          f"  (>= {p['A3']['threshold']})  holds={yn(p['A3']['holds'])}  | not followed: unsaturated at t={c['unsaturated at t']}, "
          f"saturated same sign at t={c['saturated same sign at t']}; second clause holds={yn(p['A3']['second_clause_rest_unsaturated_holds'])}")
    for label, a in (("A4 all flips (judged)", p["A4"]), ("A4 tick t moved     ", p["A4"]["on_flips_whose_tick_t_moved"])):
        print(f"  {label} bisected<logged {a['numerator']}/{a['denominator']} = {fmt(a['value'], '.3f')} (>= {a['threshold']}), "
              f"max bisected {fmt(a['max_bisected_cm'], '.3f')} cm (<= {a['max_bisected_threshold_cm']:g}; "
              f"{a['n_bisected_above_threshold']} above)  holds={yn(a['holds'])}")
    s = res["secondary"]
    print("[secondary] flips per arm: " + ", ".join(f"{k}={v}" for k, v in s["flips_per_arm"].items()))
    print("            flips per instance: " + ", ".join(f"{k}={v}" for k, v in s["flips_per_instance"].items()))
    print("            units with a flip per arm: " + ", ".join(f"{k}={v['with_flip']}/{v['units']}" for k, v in s["units_with_flip_per_arm"].items()))
    print(f"            frozen ticks: {len(s['frozen_ticks'])}  " + " ".join(f"{z['arm']}/{z['instance']}/t{z['tick']}" for z in s["frozen_ticks"]))
    false = [f for f in res["flips"] if f["judged"] and not f["true_bracket"]]
    print(f"[false flips] {len(false)}")
    for f in false:
        print(flip_line(f))
    unj = [f for f in res["flips"] if not f["judged"]]
    if unj:
        print(f"[flips on excluded instances, not judged] {len(unj)}")
        for f in unj:
            print(flip_line(f))
    print(f"[figure] {res['figure']}")


# --------------------------------------------------------------------------- main


def run(runs: Path, guardoff: Path, out: Path, cfg: Config, check_population: bool = True,
        figure: bool = True) -> dict:
    prior = load_prior()
    census, calib = load_all(runs, guardoff, cfg)
    if not census:
        sys.exit(f"no census unit under {runs} / {guardoff}")
    for u in census:
        if len(u["ticks"]) < 2 or [t["tick"] for t in u["ticks"]] != list(range(len(u["ticks"]))):
            sys.exit(f"ticks not 0..n-1 in {u['path']}")
    instances = sorted({u["instance"] for u in census})
    arms = [a for a in cfg.RUN_ARMS + cfg.GUARDOFF_ARMS]

    geometry = [fit_geometry(i, census, calib, prior, cfg) for i in instances]
    geo = {f["instance"]: f for f in geometry}
    excluded = []
    for f in geometry:
        if f["status"] != "unique":
            reason = (f"goal offset not unique: status '{f['status']}'"
                      + (f" ({f['ambiguity_kind']}, {len(f['solutions_g_cm'])} solutions)" if f["ambiguity_kind"] else ""))
            excluded.append({"instance": f["instance"], "reason": reason})
    judged_inst = {f["instance"] for f in geometry if f["status"] == "unique"}

    rows = [unit_census(u, np.array(geo[u["instance"]]["g_cm"]) / 100 if u["instance"] in judged_inst else None, cfg)
            for u in census]
    flips = [f for r in rows for f in r["flips"]]
    overshoots = [o for r in rows for o in r["overshoots"]]
    judged_flips = [f for f in flips if f["judged"]]

    pop = {
        "n_units": len(rows), "n_ticks": sum(r["n_ticks"] for r in rows),
        "n_transitions": sum(r["n_transitions"] for r in rows),
        "n_axis_transitions": sum(r["n_axis_transitions"] for r in rows),
        "n_axis_transitions_judged": sum(r["n_axis_transitions"] for r in rows if r["instance"] in judged_inst),
        "n_units_judged": sum(r["instance"] in judged_inst for r in rows),
        "n_saturated_components": sum(r["n_saturated_components"] for r in rows),
        "n_components": sum(r["n_components"] for r in rows),
        "n_flips_all": len(flips), "n_flips_judged": len(judged_flips),
        "n_overshoots": len(overshoots),
        "units_per_arm": count_by(rows, "arm", arms),
        "n_calibration_units_geometry_only": len(calib),
    }
    if check_population:
        got = (pop["n_units"], pop["n_ticks"], pop["n_transitions"], pop["n_axis_transitions"])
        want = (cfg.EXPECT_UNITS, cfg.EXPECT_TICKS, cfg.EXPECT_TRANSITIONS, cfg.EXPECT_AXIS_TRANSITIONS)
        assert got == want, f"population {got} != expected {want}"

    res = {
        "config": asdict(cfg),
        "inputs": {"runs": str(runs), "guardoff_runs": str(guardoff), "prior": str(PRIOR_PATH),
                   "prior_sha256": sha256(PRIOR_PATH),
                   "units": [{"path": u["path"], "sha256": u["sha256"], "arm": u["group"], "instance": u["instance"],
                              "seed": u["seed"], "census": not u["calibration"]} for u in census + calib]},
        "population": pop,
        "population_checked": check_population,
        "geometry": geometry,
        "excluded_instances": excluded,
        "predictions": judge(judged_flips, overshoots, cfg),
        "secondary": {
            "scope": "all census units; flips need no goal offset, so excluded instances are counted here",
            "flips_per_arm": count_by(flips, "arm", arms),
            "flips_per_instance": count_by(flips, "instance", instances),
            "flips_per_tick": count_by(flips, "tick", range(1, 5)),
            "flips_per_axis": count_by(flips, "axis", AXES),
            "first_flip_tick_per_unit": [{"arm": r["arm"], "instance": r["instance"], "first_flip_tick": r["first_flip_tick"],
                                          "n_flips": len(r["flips"])} for r in rows],
            "units_with_flip_per_arm": {a: {"with_flip": sum(r["arm"] == a and bool(r["flips"]) for r in rows),
                                            "units": sum(r["arm"] == a for r in rows)} for a in arms},
            "saturated_components_per_arm": {a: {"saturated": sum(r["n_saturated_components"] for r in rows if r["arm"] == a),
                                                 "components": sum(r["n_components"] for r in rows if r["arm"] == a)} for a in arms},
            "frozen_ticks": [z for r in rows for z in r["frozen_ticks"]],
        },
        "flips": flips,
        "overshoots": overshoots,
        "excluded_instance_sensitivity": excluded_sensitivity(census, geometry, cfg),
    }
    out.mkdir(parents=True, exist_ok=True)
    write_flips_dat(out / "flips.dat", flips, arms)
    res["figure"] = write_figure(out, flips) if figure else {"compiled": False, "reason": "disabled (--no-figure)"}
    (out / "census.json").write_text(json.dumps(res, indent=1, allow_nan=False) + "\n")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=Path("runs/E002-planner-loop"))
    ap.add_argument("--guardoff-runs", type=Path, default=Path("runs/E009-guard-off"))
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--no-population-check", action="store_true",
                    help="skip the 26 / 130 / 104 / 312 assertion (synthetic units)")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()
    res = run(args.runs, args.guardoff_runs, args.out, Config(),
              check_population=not args.no_population_check, figure=not args.no_figure)
    print_summary(res)
    print(f"wrote {args.out}/census.json, flips.dat ({len(res['flips'])} rows)"
          + (", fig-flips.tex, fig-flips.pdf" if res["figure"]["compiled"] else ""))
    if not res["figure"]["compiled"] and res["figure"].get("reason") == "pdflatex failed":
        sys.exit(f"figure compilation failed:\n{res['figure']['log_tail']}")


if __name__ == "__main__":
    main()
