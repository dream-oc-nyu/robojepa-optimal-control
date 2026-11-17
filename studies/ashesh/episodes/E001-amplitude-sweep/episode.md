# E001 — amplitude–ranking sweep

Arms A1 (predictor gain), A2 (action interface), A3 (representation + cost). Written 2026-11-11, before any E001 data.

## Question
Under the controller the release itself plans with, and at the amplitudes it was trained on, does the 22M predictor rank candidate actions like the world does? If not, which term of the regret split carries the loss?

## Prediction (before running)
1. **Interface.** With the release controller, executed/commanded ≥ 0.8 for commands ≥ 1 cm, so commanded-input and executed-input predictions agree.
2. **Gain.** `G(a) = ΔC_pred / ΔC_real` rises with amplitude: below 0.05 at ≤ 3 mm, between 0.3 and 1 at 3 cm.
3. **Ranking.** Commanded-input predicted cost orders ≥ 80% of candidate pairs like `D`; realized cost orders ≥ 90%, off-ray candidates included.

## What would change the plan
- `G` < 0.1 at 3 cm → the predictor binds at every scale → E002 varies the predictor (next checkpoint size) on this same sweep.
- Realized cost orders < 75% of pairs → the goal cost binds → E002 varies the metric.
- All three predictions hold → the mm-scale failure of E000 was a small-action and interface effect → E002 is the first search comparison (A4).

## Design
- **Instances:** 8 start–goal instances (seeded), processed in order; each finished instance is a valid partial result. Minimum useful: 4.
- **Goal:** endpoint of a 5 cm command along a seeded unit direction `v_g`.
- **Candidates per instance (19 resets):** neutral ×2; on-ray `a·v_g` for a ∈ {±1 mm, ±3 mm, ±1 cm, ±3 cm}; off-ray 1 cm and 3 cm along ± two unit directions orthogonal to `v_g`; the goal command.
- **Controller:** the mode the release's own RoboCasa planning evaluation uses (record which; the E000 pilot used the fixed four-substep mode instead).
- **Model:** released 22M/150k predictor, frozen V-JEPA 2.1-G encoder, FP32, one model tick, two real context frames, state-delta input. Two input variants: commanded XYZ (available to a planner) and executed XYZ (oracle).
- **Sealed predictions:** commanded-input predictions are written before the action is executed.

## Fixed evaluator
`readings.py` + `analyze.py` in this directory; identical code scores E000 and E001. Per candidate: predicted cost (both inputs), realized cost, physical distance `D`. Per instance: pairs ordered like `D`, selected action, progress captured, regret split, `G(a)`, executed/commanded ratio.

## Raw schema — one `instance-XX.npz` per instance, N candidates
`names (N,) str` · `kind (N,) str ∈ {neutral, onray, offray, goal}` · `commanded_xyz (N,3) f64 m` · `executed_delta14 (N,14) f64` · `ee_world (N,3) f64 m` · `sim_seconds (N,) f64` · `rewritten (N,) bool` · `z_real, z_pred_cmd, z_pred_exec (N,2,15,20,1664) f32` · `images (N,2,240,320,3) u8`, plus `meta.json` (seeds, `v_g`, controller mode, source commits, weight hashes, timings).

## Verdict
**2026-11-12 — interface holds; the gain prediction holds toward the goal and fails pooled over directions; the ranking prediction fails.** Population: 8/8 instances (start/direction seeds 1000–1007, scene seed 0), 152 candidates, none excluded, no collector deviations; export verified by size, SHA256 and seal readback (manifest `d67ecdbc…`); evaluator unchanged (`out/results.md`). (1) Interface: executed/commanded is 0.93 [0.91, 0.94] at 1 cm and 0.98 [0.97, 0.99] at 3 cm, and commanded- and executed-input predictions agree there (median relative difference of `dC_pred` 8% at 1 cm, 0.9% at 3 cm). The blocking controller's 1 mm tolerance acts as a dead band: 1 mm commands execute 0.09 mm and 3 mm commands 2.16 mm, so the two smallest amplitudes measure the dead band more than the predictor. (2) Gain: the evaluator's `G_cmd`, a median pooled over toward, away and lateral candidates, does not rise with amplitude — 0.19, 0.34, 0.37 for 1 mm, 3 mm, 1 cm commands, then 0.09 [−0.96, 0.73] for 3 cm commands (bin [10, 30) mm, 45 candidates) against a predicted 0.3–1. Toward-goal moves alone meet the prediction: `G_cmd` 0.75 at 1 cm and 0.40 [0.32, 0.51] at 3 cm; away moves give −0.07 and lateral moves 0.04 at 3 cm. (3) Ranking: predicted cost orders 47.3% of pairs like `D` (95% interval 45.5–49.7; on-ray 59.8%, off-ray 43.3%) against a predicted ≥ 80%; realized cost orders 74.7% (71.4–77.7; on-ray 90.6%, off-ray 64.3%) against ≥ 90%. Selection is better than ranking: the realized-cost argmin is the physically best candidate in 8/8 instances, the predicted-cost argmin in 7/8 (instance-01 picks a 3 cm lateral move, 37.9 mm regret), and the regret split puts all regret on predictor + interface. Plan rules: `G < 0.1` at 3 cm fires on the pooled value (0.09) only through the away and lateral signs, so the reading "the predictor binds at every scale" is not supported as written; the realized-cost rule (< 75%) is undecided, the threshold lying inside the interval. Spreads are min–max over 8 instances unless marked as bootstrap intervals.

**Post-hoc reading** (`supplement_oddeven.py`, regrouping the evaluator's per-candidate rows; exploratory, to be tested prospectively). The low pooled gain comes from the signs of away and lateral moves. For commands ≥ 1 cm the predicted cost falls for 94 of 96 moves — 16/16 toward the goal, 15/16 away, 63/64 lateral — while the realized cost falls for 16/16, 0/16 and 28/64, and `D` falls only for the 16 toward. Splitting each ± pair on the goal ray, the direction-dependent part `[dC(+a) − dC(−a)]/2` has the right sign in 8/8 instances at 1 cm and 3 cm with gain 0.22 (0.11–0.31) and 0.26 (0.19–0.30), and the wrong sign in 7/8 at 3 mm. The direction-independent part `[dC(+a) + dC(−a)]/2` of the prediction is negative in all 48 pairs at ≥ 1 cm. The realized cost has one too on the goal ray at 3 cm, because the cost is concave in distance (realized −1.6×10⁻², predicted −1.4×10⁻²); the prediction is in excess at 1 cm (realized −1.6×10⁻³ on-ray and −0.2×10⁻³ lateral, predicted about −7×10⁻³) and for lateral moves at 3 cm (realized +6.6×10⁻³, predicted −8.6×10⁻³). In short: the direction-dependent response is attenuated to about a quarter while the direction-independent response is kept or exceeded, which is enough to flip the away and lateral signs. Selecting by the direction-dependent part alone still fails on instance-01 (7/8). Figure: `out/fig-signed-response.pdf`.

**What it does to the plan.** A scalar-gain model of the predictor (E002's `J_m = G·J`) does not capture the away and lateral signs at centimetre steps: the second-order, direction-independent response is not scaled by the same `G`, so `rho` should depend on step direction and size. Open alternatives: the zero-action prediction, the reference of every `dC`, may itself be the outlier (the direction-independent term grows smoothly from 1 mm to 3 cm, which argues against a jump at zero but does not exclude it); predictions for moving actions may be lower-variance and so closer in L1 to any target. Scope: one scene, 5 cm goals, one model tick, one checkpoint, 8 instances. Next: E003 asks whether the direction-independent term is goal-free and removable.
