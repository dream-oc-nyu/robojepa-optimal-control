#!/usr/bin/env python3
"""Synthetic checks of the E011 census.  Every unit lives under a temporary directory and is
passed through --runs / --guardoff-runs / --out; nothing is written into the real out directory.

    python3 -B episodes/E011-bisect/test_census.py
"""
import sys

sys.dont_write_bytecode = True

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parent / "run.py"
CM = 0.01

# Unsaturated 3-D motion (cm): makes the pooled visited points non-coplanar, so g is unique.
GEOMETRY_STEPS = [(1, 2, 0), (0, -1, 3), (2, 1, 1), (-1, 2, -2), (1, -3, 2)]


def make_unit(root: Path, arm: str, instance: int, g_cm, commanded_cm, executed_cm=None) -> None:
    """unit.json with D_k = ||g - p_k|| computed from the true goal offset g."""
    a = np.array(commanded_cm, dtype=float) * CM
    e = a.copy() if executed_cm is None else np.array(executed_cm, dtype=float) * CM
    g = np.array(g_cm, dtype=float) * CM
    P = np.vstack([np.zeros(3), np.cumsum(e, axis=0)])
    D = np.linalg.norm(g - P, axis=1)
    unit = {"planner": arm, "instance": instance, "seed": 0, "complete": True,
            "initial_distance_m": float(D[0]),
            "ticks": [{"tick": k, "commanded": a[k].tolist(), "executed": e[k].tolist(),
                       "distance_m": float(D[k + 1]), "rho": 0.0} for k in range(len(a))]}
    d = root / arm / f"inst{instance}-seed0"
    d.mkdir(parents=True)
    (d / "unit.json").write_text(json.dumps(unit))


def x_only(xs):
    return [(x, 0, 0) for x in xs]


class CensusCase(unittest.TestCase):
    def census(self, build, figure=False) -> dict:
        """build(runs_dir) writes units; returns census.json of a CLI run in a temp dir."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runs, guard, out = tmp / "runs", tmp / "guardoff", tmp / "out"
            runs.mkdir(), guard.mkdir()
            build(runs)
            cmd = [sys.executable, "-B", str(RUN), "--runs", str(runs), "--guardoff-runs", str(guard),
                   "--out", str(out), "--no-population-check"] + ([] if figure else ["--no-figure"])
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            res = json.loads((out / "census.json").read_text())
            res["_files"] = sorted(p.name for p in out.iterdir())
            res["_dat_lines"] = (out / "flips.dat").read_text().splitlines()
        self.assertEqual([x["instance"] for x in res["excluded_instances"]], [])
        return res

    def pair(self, g_cm, commanded_cm, executed_cm=None, figure=False) -> dict:
        """One unit under test (cem_cap10) pooled with a 3-D geometry unit (cem_release)."""
        def build(runs):
            make_unit(runs, "cem_cap10", 200, g_cm, commanded_cm, executed_cm)
            make_unit(runs, "cem_release", 200, g_cm, GEOMETRY_STEPS)
        return self.census(build, figure)

    def test_a_period_two_orbit(self):
        g = (3, 0.5, -0.7)
        res = self.pair(g, x_only([10, -10, 10, -10, 10]), figure=True)
        np.testing.assert_allclose(res["geometry"][0]["g_cm"], g, atol=1e-6)
        self.assertEqual(res["geometry"][0]["status"], "unique")
        self.assertEqual(res["population"]["n_axis_transitions"], 24)
        flips = res["flips"]
        self.assertEqual([(f["arm"], f["tick"], f["axis"]) for f in flips],
                         [("cem_cap10", t, "x") for t in (1, 2, 3, 4)])
        self.assertTrue(all(f["true_bracket"] for f in flips))
        np.testing.assert_allclose([f["r_prev_cm"] for f in flips], [3, -7, 3, -7], atol=1e-6)
        np.testing.assert_allclose([f["logged_next_abs_r_cm"] for f in flips], [3, 7, 3, 7], atol=1e-6)
        np.testing.assert_allclose([f["bisected_next_abs_r_cm"] for f in flips], [2, 2, 2, 2], atol=1e-6)
        p = res["predictions"]
        self.assertEqual((p["A1"]["numerator"], p["A1"]["denominator"], p["A1"]["holds"]), (4, 4, True))
        self.assertEqual((p["A2"]["value"], p["A2"]["holds"]), (0, True))
        # overshoots of >= 2 cm: ticks 0..3 pass the goal by 7, 3, 7, 3 cm; each is followed by a flip
        self.assertEqual((p["A3"]["numerator"], p["A3"]["denominator"], p["A3"]["holds"]), (4, 4, True))
        self.assertEqual((p["A4"]["numerator"], p["A4"]["denominator"], p["A4"]["holds"]), (4, 4, True))
        self.assertEqual(res["secondary"]["flips_per_arm"]["cem_cap10"], 4)
        self.assertEqual(res["secondary"]["flips_per_arm"]["cem_release"], 0)
        self.assertEqual(res["secondary"]["frozen_ticks"], [])
        # artefacts: table with one comment line, one header and four rows; compiled figure
        self.assertTrue(res["_dat_lines"][0].startswith("#"))
        self.assertEqual(len(res["_dat_lines"]), 6)
        self.assertTrue(res["figure"]["compiled"], res["figure"])
        self.assertEqual(res["_files"], ["census.json", "fig-flips.pdf", "fig-flips.tex", "flips.dat"])

    def test_b_wrong_sign_flip_is_not_a_bracket(self):
        # goal 20 cm along +x; tick 0 moves 10 cm away from it, tick 1 turns back
        res = self.pair((20, 0.5, -0.7), x_only([-10, 10, 10, 2, 1]))
        flips = res["flips"]
        self.assertEqual([(f["tick"], f["axis"]) for f in flips], [(1, "x")])
        self.assertFalse(flips[0]["true_bracket"])
        self.assertAlmostEqual(flips[0]["r_cm"], 30, places=6)
        p = res["predictions"]
        self.assertEqual((p["A1"]["numerator"], p["A1"]["denominator"], p["A1"]["holds"]), (0, 1, False))
        self.assertEqual((p["A2"]["value"], p["A2"]["holds"]), (1, False))
        self.assertEqual(p["A3"]["denominator"], 0)
        self.assertIsNone(p["A3"]["holds"])

    def test_c_frozen_previous_tick_gives_no_flip(self):
        cmd = x_only([10, -10, 2, 1, 1])
        exe = x_only([0, -10, 2, 1, 1])      # tick 0 commanded and not executed
        res = self.pair((3, 0.5, -0.7), cmd, exe)
        self.assertEqual(res["flips"], [])
        self.assertEqual([(z["arm"], z["tick"]) for z in res["secondary"]["frozen_ticks"]], [("cem_cap10", 0)])
        self.assertIsNone(res["predictions"]["A1"]["holds"])
        # control: the same commands executed do flip at tick 1
        res = self.pair((3, 0.5, -0.7), cmd)
        self.assertEqual([(f["tick"], f["axis"]) for f in res["flips"]], [(1, "x")])

    def test_d_unsaturated_component_never_flips(self):
        # 4.9 cm alternating: below the 5 cm - 10 um threshold on every tick
        res = self.pair((3, 0.5, -0.7), x_only([4.9, -4.9, 4.9, -4.9, 4.9]))
        self.assertEqual(res["flips"], [])
        self.assertEqual(res["population"]["n_saturated_components"], 0)
        # one saturated side only: 10 cm then -4.9 cm is an overshoot that no flip follows
        res = self.pair((3, 0.5, -0.7), x_only([10, -4.9, 1, 1, 1]))
        self.assertEqual(res["flips"], [])
        p = res["predictions"]["A3"]
        self.assertEqual((p["numerator"], p["denominator"]), (0, 1))
        self.assertEqual(p["classification_counts"]["unsaturated at t"], 1)
        # at the cap exactly (clampexec logs 5 cm): saturated, and it flips
        res = self.pair((3, 0.5, -0.7), x_only([5, -5, 1, 1, 1]))
        self.assertEqual([(f["tick"], f["axis"]) for f in res["flips"]], [(1, "x")])

    def test_e_coplanar_instance_is_excluded(self):
        # every visited point on the x axis: g is known up to a rotation about it
        def build(runs):
            make_unit(runs, "cem_cap10", 201, (3, 0.5, -0.7), x_only([10, -10, 10, -10, 10]))
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            runs, guard, out = tmp / "runs", tmp / "guardoff", tmp / "out"
            runs.mkdir(), guard.mkdir()
            build(runs)
            proc = subprocess.run([sys.executable, "-B", str(RUN), "--runs", str(runs), "--guardoff-runs", str(guard),
                                   "--out", str(out), "--no-population-check", "--no-figure"],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            res = json.loads((out / "census.json").read_text())
        self.assertEqual([x["instance"] for x in res["excluded_instances"]], [201])
        self.assertEqual(res["population"]["n_flips_all"], 4)       # listed
        self.assertEqual(res["population"]["n_flips_judged"], 0)    # and not judged
        self.assertEqual(res["predictions"]["A1"]["denominator"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
