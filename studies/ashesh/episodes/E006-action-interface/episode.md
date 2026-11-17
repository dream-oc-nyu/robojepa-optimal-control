# E006 — action interface: execute the action the model was shown

Arm P3. Written 2026-11-13 15:50Z, before any E006 run, with one of the eight baseline units read (`cem_release` inst100-seed0, plus the calibration unit of the same instance). Launch is conditional on expectation 4 of E002 holding on the 8 baseline units.

## Question
On reach-v0 the release's planner (embodiment 2) shows the model `u = clamp(a / 5 cm, −1, 1)` per axis and the controller executes the metric command `a` in full, from a search ball of radius 30 cm. If the same CEM output is executed as `clamp(a, ±5 cm)` per axis, so that the executed action is the action the model scored, how much of the baseline's regret disappears?

## Why
Let `r_t = g − p_t` be the remaining goal offset at tick `t` (end-effector frame of the logged `executed` vector). The simulator logs only the distance `‖r_t‖`, but the executed displacements and the six distances of a unit determine `g` by trilateration. For instance 100 (two units, 12 distances, residual below 1e-4 cm): `g = (3.27, 3.45, −5.52)` cm.

Native CEM, tick 0, both units: `a_0 = (2.93, 3.07, −17.0)` cm. The two axes inside the cube match the goal offset to 0.4 cm (ratio 0.90, 0.89); the axis outside it has the right sign and a length the model never scored. Executed in full the arm ends 11.6 cm from the goal; `clamp(a_0, ±5 cm) = (2.93, 3.07, −5)` would end 0.73 cm from it. Over the 10 ticks of the two units every commanded component whose remaining offset exceeds 3 cm has the sign of that offset (20 of 20 by hand count, to be recomputed by `prior.py`; one instance, two runs of the same seed, ticks 3–4 from an unmoved arm, so not independent). Below about 1 cm of remaining offset the commanded component is not resolved (tick 1: remaining 0.36 cm, commanded 3.6 cm).

The reading: on this instance the model and the search locate the goal; the regret is created between the planner's output and the controller.

## Arm
`cem_clampexec`: `plan = clamp(cem_release.plan(ctx), −0.05, 0.05)` elementwise, metres. Same native search, settings, RNG stream and planning time as the baseline. Nothing else changes; the harness, the evaluator and reach-v0 are untouched. Seed 0, dev instances 100–107, paired by instance against `cem_release` seed 0.

Not in this episode: a CEM that searches inside the cube. It answers a second question (does the flat region outside the cube degrade the search inside it) and runs only if prediction 2 fails on 3 or more instances.

## Prediction (before running)
1. `cem_clampexec` ends closer to the goal than `cem_release` on at least 7 of 8 instances; its median progress is at least 0.5.
2. Tick 0 is predicted by the baseline's own log: the realized distance after tick 0 is within 1.0 cm of `‖g − clamp(a_0^baseline)‖` on at least 6 of 8 instances (instances whose `g` is not determined by the baseline's displacements are excluded and counted).
3. Closest approach `min_t D_t` is at most 2 cm on at least 5 of 8 instances.
4. The arm does not hold the goal: `D_5 − min_t D_t ≥ 1 cm` on at least 4 of 8 instances, because below about 1 cm of remaining offset the commanded component is unresolved (E001 at 1 cm: toward move chosen in 6/8 even by the realized cost).
5. `rho` becomes a usable signal: over ticks with executed motion of at least 1 cm its median lies in [0.3, 3] and it is positive on at least 70% of them (baseline unit: −0.10, −4.16, 0.62, 0.07, 0.01).
6. Executed motion follows the command: executed/commanded length within [0.8, 1.2] on at least 36 of 40 ticks (baseline unit: 2 of 5 ticks executed nothing after 20 cm swings).

## What would change the plan
- 1–3 hold → `cem_clampexec` is the incumbent on dev. Seeds 1–2 for both planners, then held-out seeds 200–215 for both, then the brief. The planner question (E002 prediction 3: Adam and trust-region Gauss–Newton against CEM) restarts inside the cube against this incumbent, at `B` and at fractions of `B`, since native CEM spends 10,240 predictions per tick on a 3-dimensional search.
- 4 holds → the next module is the stopping behaviour near the goal: trust radius from `rho` (P1) against view weights (P10), chosen by whether `rho` falls toward 0 on the drifting ticks.
- 1 fails → executed motion inside the cube is not what the model scored either (4 tiled sub-actions, context actions through the same clamp): measure the model-side to executed ratio with a short embodiment-2 amplitude sweep before any further closed-loop run.
- 2 fails on 3 or more instances while 1 holds → run the cube-search arm.

## Deviation from E002's stated plan
E002 said that if expectation 6 failed, seeds 1–2 would be measured before any paired comparison. It failed: the two same-seed units of instance 100 differ by 0.9 mm at tick 0 and up to 8.7 mm later. The tick-0 difference is confined to the saturated axis (z: −16.99 against −17.08 cm; x and y agree to 0.1 mm), where the predicted cost is flat and ties decide. Final distances differ by 0.6 cm (12.93 against 12.32). The predicted effect here is about 10 cm on this instance, so seed 0 of `cem_clampexec` runs first (2.8 h) and seeds 1–2 of both planners follow before any held-out use.

## Outputs
`prior.py` (CPU, numpy): goal offsets by trilateration from the exported baseline units, the tick-0 counterfactual of prediction 2, per-axis commanded component against remaining offset → `out/prior.json`, `out/axis_response.dat`. Units under `runs/E002-planner-loop/cem_clampexec/`, scored by `bench/score.py` unchanged.

## Launch note (2026-11-13 17:53Z; after the 8 baseline units, before any E006 unit)
The launch condition failed: expectation 4 of E002 counted 23 of 40 ticks with a component above 5 cm (written: at least 28), and the baseline reaches within 2 cm on 6 of 8 instances (`episodes/E002-planner-loop/episode.md`, Verdict). The pre-written branch for that case was view weights. The arm is launched all the same, as a stated deviation, for what the 8 units show: both failures (100, 107) and all 5 unmoved ticks follow a tick-0 command that is saturated on an axis and overshoots it; saturated components pass the remaining offset in 22 of 32 (unique-`g` instances) with a median length of 14.3 cm; view weights are not expected to act on that (a judgement, untested). Saturated overshoots also occur in successes, and why the arm then stops moving in the two failures is unread. `cem_clampexec` is the 5 cm end of a dose–response in the executed cap `L` (native: no cap; E007: `L` = 10 cm).

Predictions 1–6 above stand as written and are judged as written. Expectations written now, with the baseline known (lower evidential weight, labelled as such):
- a. Closer than the baseline on 100 and 107, each by at least 5 cm at the final tick.
- b. Farther than the baseline on 103, 105 and 106, whose largest offsets (22.2, 24.5, 24.3 cm) need five saturated ticks at 5 cm; within 1 cm of the baseline on 101, 102 and 104. Prediction 1 (7 of 8 closer) is therefore expected to fail, at 2 to 4 of 8.
- c. Mean progress at least 0.75 (baseline 0.587); median progress below the baseline's 0.951.
- d. No tick without executed motion (baseline: 5 of 40).
- e. Tick-0 distances follow the counterfactual of `prior.py` on the 6 unique-`g` instances: 0.73, 14.91, 10.86, 19.62, 20.69, 19.37 cm for 100, 101, 102, 103, 105, 106 (prediction 2).

## Verdict
**2026-11-13 20:40Z; `cem_clampexec`, seed 0, dev instances 100–107, 8 units, 40 ticks, paired by instance with `cem_release` seed 0; scored by `bench/score.py` unchanged; one planner seed.** Job `e006-cem_clampexec` ran 17:54–20:27Z (2.45 h of planning, 211 s per tick median); 8 units exported and SHA256-verified.

Executing the action the model scored removes both total failures and every frozen tick, and costs the three farthest starts their last centimetres. Final distance, cm (start; native; 5 cm cap): 100 (7.29; 12.32; **0.42**), 101 (23.53; 0.38; 0.67), 102 (15.81; 0.54; 0.32), 103 (26.28; 1.43; **7.13**), 104 (9.41; 0.21; 0.30), 105 (26.93; 1.18; **4.69**), 106 (25.17; 1.79; **5.06**), 107 (7.07; 9.69; **0.77**). Progress is positive on 8 of 8 (native 6), mean 0.888 (native 0.587), median 0.917 (native 0.951). Fixed scorer: paired difference of medians −0.034, 95% interval [−0.139, 1.262]: not an incumbent. Secondary reading (`episodes/E007-cap-length/run.py`): mean paired difference +0.301 [−0.103, 0.781], interval includes zero. The whole gain is carried by 100 and 107 (+1.63, +1.26), the two instances on which the native arm enters the wrapper's joint-margin guard (E008); on the other six the cap costs 0.081 of mean progress (−0.217 to +0.014).

| written before any E006 unit | result |
|---|---|
| 1. closer than native on ≥ 7 of 8; median progress ≥ 0.5 | first clause fails: 3 of 8 (100, 102, 107); second holds: 0.917 |
| 2. tick-0 distance within 1.0 cm of `‖g − clamp(a_0^baseline)‖` on ≥ 6 of 8 | holds: 6 of 6 determinable, each within 0.07 cm (0.70 / 0.73, 14.96 / 14.91, 10.93 / 10.86, 19.61 / 19.62, 20.68 / 20.69, 19.38 / 19.37); 104 and 107 excluded as written (their primary solutions give 4.95 / 4.89 and 2.36 / 2.35) |
| 3. closest approach ≤ 2 cm on ≥ 5 of 8 | holds at exactly 5 (100, 101, 102, 104, 107: 0.30–0.77 cm) |
| 4. the arm does not hold the goal: `D_5 − min_t D_t ≥ 1 cm` on ≥ 4 of 8 | fails: 0 of 8; the final tick is the closest on all eight |
| 5. `rho` over ticks with ≥ 1 cm executed: median in [0.3, 3], positive on ≥ 70% | holds: median 1.40, positive on 30 of 31 |
| 6. executed/commanded length within [0.8, 1.2] on ≥ 36 of 40 ticks | fails as written: 31 of 40. Six of the nine are commands below 0.4 cm; the other three are 103 ticks 2–4 (below). On commands ≥ 1 cm: 28 of 31 |

Expectations written with the baseline known (lower weight): a holds (100 closer by 11.9 cm, 107 by 8.9 cm); b holds on all six named instances (103, 105, 106 farther; 101, 102, 104 within 0.3 cm of native; closer on 3 of 8, inside the expected 2 to 4); c holds (mean 0.888 ≥ 0.75; median below native); d fails as written at 1 of 40 (100 tick 4: a 0.7 mm command, converged in 0 substeps, nothing executed) and holds under the definition that counts the native 5 (< 1 mm executed on a command ≥ 1 cm: 0 of 40); e holds (prediction 2).

**Where the regret now arises** (same `g` as the baseline's trilateration; it reproduces all 40 logged distances of these units to below 1e-4 mm, including the primary solutions of 104 and 107, an independent check of `prior.py`).
- *Every capped tick ends closer.* 20 ticks have a component at the cap and the arm ends closer on 20 of 20 (native: 14 of 23 saturated ticks, 4 farther, 5 unmoved). The 20 ticks inside the cube end closer on 18, distance × 0.56 per tick (native 0.53); the two exceptions are sub-millimetre commands at 0.4–0.7 cm from the goal.
- *The cap turns the task into an `L∞` budget.* Each tick removes at most 5 cm per axis, so an offset `r` needs `⌈‖r‖∞ / 5 cm⌉` ticks: 5 of the 5 allowed for 103, 105, 106 (largest components 22.2, 24.5, 24.3 cm), at most 3 for the others. They are the three instances that end more than 1 cm behind native (101 and 104 are behind by 0.3 and 0.1 cm): three instances, one seed, and no run at a longer horizon to test it. The budget has no slack for an error, and saturation is not a calibrated report of distance: of 38 capped components (unique-`g` instances) 4 have the wrong sign (105 tick 1, y and z; 103 ticks 3–4, x) and 3 more move 5 cm on an axis with less than 5 cm left. 105 tick 1 executes 8.7 cm and gains 0.6 cm.
- *103 stalls even at 5 cm.* Its ticks 2–4 stop at 200 substeps having executed 0.57, 0.41 and 0.37 of the command, and in a rotated direction (x commanded +2.7, +5, +5 cm; executed −0.1, −0.6, −0.6). The native run stalled on the same instance (ticks 0 and 2). Not a planner effect: the E008 replay of these units finds no guard activity (margins above 1.1 rad) and a +x probe that moves 0.3 cm of 5, as on the native units; cause unread (`episodes/E008-frozen-arm/episode.md`).
- *The arm holds the goal.* Prediction 4 expected drift below 1 cm of remaining offset; instead commands shrink with the offset (100: 6.6, 0.1, 1.1, 0.1, 0.1 cm) and the residual is 0.3–0.8 cm on the five near finishes, against 0.2–0.5 cm for native on its own successes. `rho` there is large and uninformative (up to 62 and −66 on sub-millimetre motion).

**What it does to the account.** The baseline read said: inside the cube the model sets the step, outside it the model sets a sign and the sampler a length, and the failures follow long executed swings. The intervention changes only the executed length. Established on these 8 units: the baseline's log predicts the capped tick 0 to 0.07 cm (the controller executes a 5 cm command faithfully, and counterfactuals from the log are valid); no capped tick ends farther; both failures and all frozen ticks disappear. Not established: that the gain comes from executing *what the model scored* and not from executing *less*. E008 shows the two failures are entries into a controller guard that only moves above 20 cm reach, so any shorter step would avoid them, and the six guard-free instances lose 0.081. The one reading that separates the two is a tick-0 counterfactual, not executed: against a uniform rescaling of `a_0` to the same executed length, the per-axis clamp lands closer on 4 of the 6 unique-`g` instances and decisively on one (100: 0.73 against 3.28 cm; the others differ by 0.3–2.0 cm, 102 and 103 in the other direction). Reviewed once (fresh context; the numbers reproduced from the raw units; this paragraph, expectation d and the `L∞` sentence were tightened after it).

Missing controls, in order: native CEM on 100 and 107 with the guard disabled (do the failures belong to the planner or to the guard: E009); a length-matched step that ignores the per-axis structure (uniform rescale to the cap); planner seeds 1–2 and more near or low-margin starts; a longer horizon for the `L∞` reading.

**Plan.** The branch written for "1 fails" (executed motion inside the cube is not what the model scored; run an embodiment-2 amplitude sweep) is not taken: prediction 2 and the on-command ratios show the executed step is the scored step, and clause 1 fails for the reason expectation b named in advance. Prediction 2 holds, so the cube-search arm is not triggered. Prediction 4 fails, so stopping behaviour near the goal is not the next module. No incumbent on the fixed scorer; the headline question (Blocker 2) is unchanged. Next: E007 (`cem_cap10`, launched 20:44Z), the middle dose, whose `L∞` budget is 3 ticks for the far starts; then the guard-off control (E009); then baseline seeds 1–2.

Limits: one planner seed (native reruns differ at the mm level on saturated axes); 8 instances, two of them carrying the mean; the 5-tick allowance is part of reach-v0, so "bound by ticks" is a statement about this benchmark's horizon; sign errors on capped axes are counted on 6 instances; FP32 where the release runs FP16.
