#!/usr/bin/env python3
"""Synthetic code-path tests, not experimental evidence for E011 predictions.

All fabricated receipts and analysis outputs live below $TMPDIR. Exported CEM
controls are copied byte-for-byte; the repository and leaderboard stay read-only.
"""
import sys
sys.dont_write_bytecode = True

import copy
import hashlib
import json
import os

import numpy as np
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN = HERE / 'run_b.py'
REFERENCES = ('cem_release', 'cem_cap10', 'cem_clampexec')


class PartBCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ['TMPDIR'], prefix='e011-partb-test-')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.runs, self.out, self.replays = (self.base / x for x in ('runs', 'out', 'replays'))
        for arm in REFERENCES:
            shutil.copytree(ROOT / 'runs/E002-planner-loop' / arm, self.runs / arm)
        self.listing = sorted(str(p.relative_to(HERE)) for p in HERE.rglob('*'))
        self.leaderboard = ROOT / 'bench/leaderboard.jsonl'
        self.before = self.leaderboard.read_bytes() if self.leaderboard.exists() else None
        self.addCleanup(self.check_read_only)

    def check_read_only(self):
        self.assertEqual(self.listing, sorted(str(p.relative_to(HERE)) for p in HERE.rglob('*')))
        after = self.leaderboard.read_bytes() if self.leaderboard.exists() else None
        self.assertEqual(self.before, after)

    def analyze(self):
        proc = subprocess.run([sys.executable, '-B', str(RUN), '--search', 'gna',
                               '--runs', str(self.runs), '--out', str(self.out),
                               '--replays', str(self.replays)], cwd=ROOT,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.stdout = proc.stdout
        return json.loads((self.out / 'result.json').read_text())

    def test_absent_arms_are_all_pending(self):
        result = self.analyze()
        self.assertEqual(set(result['predictions']), {'B1', 'B2', 'B3', 'B4', 'B5', 'C1', 'C2', 'C3'})
        self.assertEqual({p['verdict'] for p in result['predictions'].values()}, {'pending'})
        self.assertEqual({p['verdict'] for p in result['secondary']['predictions'].values()}, {'pending'})
        for rule in ('sat10', 'bisect', 'halvesat'):
            row = result['arms'][rule + '_gna_a']
            self.assertEqual((row['status'], row['units_present']), ('missing', 0))
            self.assertEqual(row['replay']['status'], 'not run')
        self.assertTrue((self.out / 'fig-partb.tex').is_file())
        self.assertTrue((self.out / 'axis.dat').is_file())
        self.assertEqual([r['instance'] for r in result['excluded_instances']], [104])
        self.assertFalse(any(r['instance'] == 104 for r in result['axis_table']))
        if shutil.which('pdflatex'):
            self.assertTrue(result['figure']['compiled'], result['figure'])


    def build_complete(self):
        """Analytic trajectories near the controls' independently solved goals.

        Least-squares sphere differences recover the controls' g to <1 micrometre;
        all new distances are generated from this known g, not fitted by run_b.
        This fixture tests analysis, not feasibility of the controller trajectory.
        """
        self.goals = {}
        for i in range(100, 108):
            positions, rhs = [], []
            for arm in REFERENCES:
                u = json.loads((self.runs / arm / f'inst{i}-seed0/unit.json').read_text())
                p = np.vstack([np.zeros(3), np.cumsum([t['executed'] for t in u['ticks']], axis=0)])
                d = np.array([u['initial_distance_m']] + [t['distance_m'] for t in u['ticks']])
                positions.extend(p)
                rhs.extend(np.sum(p*p, axis=1) + d[0]**2 - d*d)
            self.goals[i] = np.linalg.lstsq(2*np.array(positions), rhs, rcond=None)[0]
        for rule in ('sat10', 'bisect', 'halvesat'):
            arm = rule + '_gna_a'
            for i, g in self.goals.items():
                template = self.runs / 'cem_release' / f'inst{i}-seed0/unit.json'
                u = json.loads(template.read_text())
                u.update(planner=arm, initial_distance_m=float(np.linalg.norm(g)),
                         dataset_kind='synthetic-code-path-test')
                base = np.array([[.01, .02, 0], [0, -.01, .03], [.02, .01, .01],
                                 [-.01, .02, -.02], [.01, -.03, .02]])
                if i in (100, 107):
                    base[:2, 0] = [.05, -.05]
                if i == 105:
                    base[:2, 1] = [.05, -.05]  # previous motion is AWAY: false, far-axis flip
                if i == 107:
                    base[:, 0] = [.05, -.05, .05, -.05, .05] if rule == 'sat10' else [.05, -.05, -.05, 0, 0]
                lengths, previous, seen = np.full(3, .1), np.zeros(3), np.zeros(3, dtype=bool)
                side_ticks, position = [], np.zeros(3)
                target_distance = .06 if rule == 'sat10' else .06*np.linalg.norm(g)
                if i == 107 and rule != 'sat10':
                    target_distance = .015 if rule == 'bisect' else .003
                for t, tick in enumerate(u['ticks']):
                    saturated = np.abs(base[t]) >= .05-1e-5
                    flip = saturated & (previous == -np.sign(base[t])) if t else np.zeros(3, dtype=bool)
                    if rule == 'sat10':
                        flip[:] = False
                    before, seen_before = lengths.copy(), seen.copy()
                    seen |= flip
                    halve = flip if rule != 'halvesat' else saturated & seen
                    lengths[halve] = np.maximum(.01, lengths[halve]/2)
                    command = np.where(saturated, np.sign(base[t])*lengths, base[t])
                    executed = command.copy()
                    if t == 4:
                        executed = g - np.array([target_distance, 0, 0]) - position
                    position += executed
                    tick.update(commanded=command.tolist(), executed=executed.tolist(),
                                distance_m=float(np.linalg.norm(g-position)))
                    log = dict(base_command=base[t].tolist(), saturated=saturated.tolist(), flip=flip.tolist(),
                               length_before=before.tolist(), length_after=lengths.tolist(),
                               prev_sat_sign=previous.tolist(), flipped_before=seen_before.tolist(),
                               flipped_after=seen.tolist(), executed_prev=None if not t else u['ticks'][t-1]['executed'],
                               out=command.tolist(), shrink='saturated' if rule == 'halvesat' else 'flip',
                               rule='halve_on_saturated_after_flip' if rule == 'halvesat' else 'halve_on_flip')
                    side_ticks.append(dict(tick=t, log=dict(length_rule=log)))
                    previous = np.where(saturated, np.sign(base[t]), 0)
                path = self.runs / arm / f'inst{i}-seed0'
                path.mkdir(parents=True)
                (path / 'unit.json').write_text(json.dumps(u))
                (path / 'planner_log.json').write_text(json.dumps(dict(planner=arm, instance=i, seed=0, ticks=side_ticks)))
            self.build_replay(arm)

    def build_replay(self, arm, guards=0):
        receipts, hashes = [], {}
        for i in range(100, 108):
            path = self.runs / arm / f'inst{i}-seed0/unit.json'
            u = json.loads(path.read_text())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            hashes[f'inst{i}-seed0'] = digest
            ticks = []
            for t in u['ticks']:
                ticks.append(dict(tick=t['tick'], commanded=t['commanded'], executed=t['executed'],
                                  distance_m=t['distance_m'], sim_steps=2,
                                  substeps=dict(guard_fired=[bool(guards and i == 100 and t['tick'] == 0), False],
                                                pos_zeroed=[False, False])))
            receipts.append(dict(instance=i, seed=0, complete=True, initial_distance_m=u['initial_distance_m'],
                                 input_unit_sha256=digest, ticks=ticks,
                                 probes=[dict(substeps=dict(guard_fired=[True]*10))]))
        path = self.replays / arm / 'replay.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(complete=True, planner=arm, seed=0, input_sha256=hashes,
                                        instances=receipts, dataset_kind='synthetic-code-path-test')))

    def test_complete_known_verdicts_geometry_and_fixed_scorer(self):
        self.build_complete()
        result = self.analyze()
        expected = dict(B1='holds', B2='holds', B3='holds', B4='holds', B5='fails',
                        C1='holds', C2='holds', C3='holds')
        self.assertEqual({k: p['verdict'] for k, p in result['predictions'].items()}, expected)
        self.assertEqual({p['verdict'] for p in result['secondary']['predictions'].values()}, {'holds'})
        self.assertEqual(result['predictions']['C1']['numbers']['same_instances'], 7)
        self.assertEqual(result['predictions']['B1']['numbers']['sat10_x_flips'], 4)
        self.assertAlmostEqual(result['predictions']['B5']['numbers']['median'], .94)
        self.assertEqual(result['excluded_instances'], [])
        for geo in result['geometry']:
            np.testing.assert_allclose(np.array(geo['g_cm'])/100, self.goals[geo['instance']], atol=1e-6)
        rows = result['axis_table']
        self.assertEqual(len(rows), 3*8*5*3)
        true = next(r for r in rows if (r['arm'], r['instance'], r['tick'], r['axis']) == ('bisect_gna_a', 100, 1, 'x'))
        false = next(r for r in rows if (r['arm'], r['instance'], r['tick'], r['axis']) == ('bisect_gna_a', 105, 1, 'y'))
        self.assertTrue(true['flip'] and true['true_bracket'])
        self.assertTrue(false['flip'] and false['far_axis_flip'])
        self.assertFalse(false['true_bracket'])
        self.assertAlmostEqual(true['remaining_m'], self.goals[100][0]-.1, places=6)
        self.assertEqual(result['arms']['bisect_gna_a']['diagnostics']['far_axis_false_flips'], 1)
        # Independent fixed-evaluator invocation on EXACTLY the same scored receipts.
        sys.path.insert(0, str(ROOT))
        from bench.score import summarize
        units = [json.loads(p.read_text()) for p in sorted(self.runs.glob('*/inst*-seed0/unit.json'))]
        scores = json.loads(json.dumps(summarize(units)))
        for arm, score in scores.items():
            self.assertEqual(result['arms'][arm]['fixed_scorer'], score)
        self.assertTrue(result['figure']['compiled'] if shutil.which('pdflatex') else not result['figure']['compiled'], result['figure'])
        self.assertEqual(len(list(self.out.glob('*.tex'))), 1)
        self.assertEqual(len((self.out / 'axis.dat').read_text().splitlines()), 361)
        for name, digest in result['inputs']['sha256'].items():
            self.assertEqual(hashlib.sha256(Path(name).read_bytes()).hexdigest(), digest)

    def test_partial_arm_and_missing_inputs_do_not_pass(self):
        self.build_complete()
        for i in (105, 106, 107):
            shutil.rmtree(self.runs / 'bisect_gna_a' / f'inst{i}-seed0')
        result = self.analyze()
        row = result['arms']['bisect_gna_a']
        self.assertEqual((row['status'], row['units_present'], row['judgment_status']), ('partial', 5, 'pending'))
        self.assertEqual({p['verdict'] for p in result['predictions'].values()}, {'pending'})
        # A missing replay is not evidence for zero guards. A missing sidecar only
        # blocks predicates requiring it, not distance/progress-only clauses.
        self.build_replay('sat10_gna_a', guards=1)
        (self.runs / 'sat10_gna_a/inst107-seed0/planner_log.json').unlink()
        result = self.analyze()
        self.assertEqual(result['predictions']['B1']['verdict'], 'pending')
        self.assertEqual(result['secondary']['predictions']['B1']['verdict'], 'pending')
        self.assertEqual(result['secondary']['predictions']['B2']['verdict'], 'holds')

    def change_endpoint(self, arm, instance, distance):
        path = self.runs / arm / f'inst{instance}-seed0/unit.json'
        u = json.loads(path.read_text())
        position = np.sum([t['executed'] for t in u['ticks'][:-1]], axis=0)
        u['ticks'][-1]['executed'] = (self.goals[instance]-[distance, 0, 0]-position).tolist()
        u['ticks'][-1]['distance_m'] = distance
        path.write_text(json.dumps(u))

    def test_distance_failures_and_strict_command_tolerance(self):
        self.build_complete()
        self.change_endpoint('bisect_gna_a', 107, .03)
        self.change_endpoint('bisect_gna_a', 103, .20)
        self.change_endpoint('halvesat_gna_a', 107, .05)
        self.change_endpoint('halvesat_gna_a', 103, .30)
        for i in (101, 102, 103, 104):
            path = self.runs / 'halvesat_gna_a' / f'inst{i}-seed0'
            u = json.loads((path / 'unit.json').read_text())
            side = json.loads((path / 'planner_log.json').read_text())
            self.assertEqual(u['ticks'][0]['commanded'][2], 0.)
            u['ticks'][0]['commanded'][2] = .0001  # exact boundary: NOT same
            side['ticks'][0]['log']['length_rule']['out'][2] = .0001
            side['ticks'][0]['log']['length_rule']['base_command'][2] = .0001
            (path / 'unit.json').write_text(json.dumps(u))
            (path / 'planner_log.json').write_text(json.dumps(side))
        for arm in ('bisect_gna_a', 'halvesat_gna_a'):
            self.build_replay(arm)
        result = self.analyze()
        for name in ('B1', 'B2', 'B3', 'C1', 'C2', 'C3'):
            self.assertEqual(result['predictions'][name]['verdict'], 'fails', name)
        self.assertEqual(result['predictions']['C1']['numbers']['same_instances'], 3)
        for name in ('B1', 'B2', 'B3'):
            self.assertEqual(result['secondary']['predictions'][name]['verdict'], 'fails', name)

    def test_headline_holds_and_nonunique_axis_exclusion(self):
        self.build_complete()
        for i in range(100, 108):
            self.change_endpoint('bisect_gna_a', i, .04*np.linalg.norm(self.goals[i]))
        self.build_replay('bisect_gna_a')
        result = self.analyze()
        self.assertEqual(result['predictions']['B5']['verdict'], 'holds')
        self.assertAlmostEqual(result['predictions']['B5']['numbers']['median'], .96)
        # Replace every synthetic trajectory on 104 with the real coplanar
        # control. There ARE sidecars/units, but no unique g: skip axis rows.
        original = json.loads((self.runs / 'cem_clampexec/inst104-seed0/unit.json').read_text())
        for rule in ('sat10', 'bisect', 'halvesat'):
            arm = rule + '_gna_a'
            path = self.runs / arm / 'inst104-seed0'
            u = copy.deepcopy(original)
            u['planner'] = arm
            (path / 'unit.json').write_text(json.dumps(u))
            side = json.loads((path / 'planner_log.json').read_text())
            for t in range(5):
                log = side['ticks'][t]['log']['length_rule']
                log['out'] = u['ticks'][t]['commanded']
                log['executed_prev'] = u['ticks'][t-1]['executed'] if t else None
            (path / 'planner_log.json').write_text(json.dumps(side))
            self.build_replay(arm)
        result = self.analyze()
        self.assertEqual([r['instance'] for r in result['excluded_instances']], [104])
        self.assertFalse(any(r['instance'] == 104 for r in result['axis_table']))
        self.assertEqual(result['arms']['bisect_gna_a']['diagnostics']['excluded_instances'], [104])

    def test_guard_counts_missing_flags_and_provenance(self):
        self.build_complete()
        self.build_replay('sat10_gna_a', guards=1)
        result = self.analyze()
        self.assertEqual(result['predictions']['B4']['verdict'], 'fails')
        self.assertEqual(result['arms']['sat10_gna_a']['replay']['guard_substeps'], 1)
        self.assertEqual(result['arms']['bisect_gna_a']['replay']['guard_substeps'], 0)
        path = self.replays / 'sat10_gna_a/replay.json'
        data = json.loads(path.read_text())
        del data['instances'][0]['ticks'][0]['substeps']['guard_fired']
        path.write_text(json.dumps(data))
        result = self.analyze()
        self.assertEqual(result['predictions']['B4']['verdict'], 'pending')
        self.build_replay('sat10_gna_a')
        data = json.loads(path.read_text())
        data['input_sha256']['inst100-seed0'] = '0'*64
        path.write_text(json.dumps(data))
        self.assertEqual(self.analyze()['predictions']['B4']['verdict'], 'pending')
        path.unlink()
        result = self.analyze()
        self.assertEqual(result['predictions']['B4']['verdict'], 'pending')
        self.assertEqual(result['arms']['sat10_gna_a']['replay']['status'], 'not run')


if __name__ == '__main__':
    unittest.main(verbosity=2)
