# Reproduction and publication verification

Run from `studies/ashesh/`, not the repository root: this study has its own `planners` package and `bench` interface. It does not replace the top-level team planner API.

## Dependencies

Offline scoring and numeric analysis require Python and NumPy. The focused planner tests additionally require PyTorch. Test fixtures do not require model weights or a GPU. Publication checks used Python 3.13 and the locally installed NumPy/PyTorch environment; their exact versions are in `VERIFICATION.json`.

For real-model execution, use the official RoboJEPA environment and the source commits and checkpoint SHA-256 hashes pinned in `bench/pod/run_unit.py`. The upstream source, frozen weights, RoboCasa/robosuite assets, and CUDA simulator environment must be supplied separately. `bench/test_runtime.py` and `bench/pod/test_cem_adapter.py` require those upstream imports and were **not rerun** for this publication.

The original `pod/` shell scripts are historical environment-specific drivers, not a turnkey install. No cluster access is needed for the following offline checks.

## Recompute the published reach scores

Choose an output directory outside the source tree:

```bash
cd studies/ashesh
export VERIFY_DIR="${TMPDIR:-$HOME/.cache}/robojepa-verification"
mkdir -p "$VERIFY_DIR"
python bench/score.py \
  --runs runs/E002-planner-loop \
  --out "$VERIFY_DIR/score" \
  --leaderboard "$VERIFY_DIR/leaderboard.jsonl"
```

Only complete five-tick `unit.json` receipts enter the score. Calibration is excluded. This should report eight units per arm and `complete_24_units: false` for each arm: only seed 0 of the intended three planner seeds is available.

```bash
python episodes/E011-bisect/run.py \
  --out "$VERIFY_DIR/census" --no-figure
python episodes/E010-gn-in-cube/run.py \
  --out "$VERIFY_DIR/gn"
python episodes/E011-bisect/run_b.py --search cem \
  --out "$VERIFY_DIR/length-rules"
```

The census checks its 26-unit / 130-tick / 312-axis-transition population and reports 26/28 true brackets. The E010 script should report `No gn units: no output written.` The E011 part-B script should report predictions B1–B5 and C1–C3 as pending, not score absent arms as zeros. Figure compilation is optional and depends on a LaTeX installation.

## Focused CPU tests

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
python -B -m unittest \
  bench.test_score \
  bench.pod.test_clampexec bench.pod.test_cap10 \
  bench.pod.test_gn_cube_a bench.pod.test_gn_cube_b \
  bench.pod.test_bisect_a bench.pod.test_bisect_b \
  bench.pod.test_planner_log
python -B -m unittest discover -s episodes/E011-bisect -p 'test_*.py'
python -B -m unittest discover -s episodes/E008-frozen-arm/pod -p 'test_*.py'
python -B -m unittest discover -s episodes/E009-guard-off/pod -p 'test_*.py'
```

On publication these batches passed **101, 11, 7, and 8 tests**, respectively. Synthetic fixtures in those tests establish code-path behavior only; they are not included as experimental evidence.

### Known failing exploratory test

The older `python -B -m unittest -v -f planners.test_planners` suite did **not** pass publication verification. Its finite-difference Adam contract case exceeded a strict 0.3-second planning deadline: 0.320796 seconds on this host. The first two test methods passed before fail-fast stopped the run. An initial combined invocation without CPU thread limits also timed out after 180 seconds and showed failures; it is not counted as a pass.

The timing failure is retained and disclosed rather than changing the planner or weakening the test during a publication task. Do not treat the entire study as having a green test suite or use the toy planner outputs as real-model validation.

## What was verified, and what was not

- All 24 scored reach units are complete, with five ticks each, and independent recomputation of their per-arm means, medians, and distance counts agrees with `PROGRESS.md`.
- Re-running `bench/score.py` produced a byte-identical `results.json` to the source study's saved output.
- The sign-flip census was rerun from the included real unit logs; E010 and adaptive-length analysis correctly identify missing evidence.
- Copied source/data files have publication hashes. Some historical outputs record absolute source paths; these are provenance strings, not portable input configuration. Their raw contents are preserved rather than silently rewriting scientific receipts.
- E001–E005 latent-array analyses were not rerun: their multi-gigabyte raw arrays are excluded from Git. Earlier PDF reports are included as historical reading, not newly reproduced claims.
- No model training, new simulator experiment, held-out evaluation, remote job recovery, or top-level team-runner integration was performed for publication.
