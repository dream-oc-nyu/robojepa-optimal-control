# E007 — the unscored length: a cap on the executed command, as a dose–response

Arm P3. Written 2026-11-13 17:58–18:04Z (committed a7ab441 before the first readback of the E006 job), with the 8 seed-0 baseline units read (`episodes/E002-planner-loop/episode.md`, Verdict) and before any unit of `cem_clampexec` (E006, launched 17:54Z) or of this arm.

## Question
With planner embodiment 2 the model scores `u = clamp(a / 5 cm, −1, 1)` per axis and the controller executes `a`. On a saturated axis the model fixes the sign and the sampler fixes the length (baseline: 9.0–21.7 cm, median 14.3; past the remaining offset in 22 of 32 components). Executing `clamp(a, ±L)` leaves the search untouched and bounds that length. How does reach-v0 regret depend on `L`: 5 cm (E006, executed = scored), 10 cm (this arm), none (baseline)?

## Why 10 cm
A dose point between the two ends, chosen from the tick-0 counterfactual below. Saturation is not a calibrated report of distance: 5 of the 32 saturated components had less than 5 cm remaining (down to 0.7 cm), and the ratio `a_i / r_i` ≈ 0.59 is measured inside the cube only. Tick-0 counterfactual from the baseline's own commands, `‖g − clamp(a_0, ±L)‖` in cm (`g` by trilateration; 104 and 107 mirror-ambiguous, primary solution shown):

| instance | start | L = 5 | 7.5 | 10 | 12.5 | 15 | native (logged) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | 7.29 | 0.73 | 2.04 | 4.50 | 6.99 | 9.49 | 11.63 |
| 101 | 23.53 | 14.91 | 10.61 | 6.38 | 4.17 | 5.03 | 6.33 |
| 102 | 15.81 | 10.86 | 8.41 | 6.97 | 7.17 | 8.10 | 8.19 |
| 103 | 26.28 | 19.62 | 16.98 | 15.14 | 13.29 | 13.24 | 16.54 (controller cut) |
| 104 | 9.41 | 4.89 | 4.89 | 4.89 | 4.89 | 4.89 | 4.95 |
| 105 | 26.93 | 20.69 | 18.95 | 17.75 | 17.20 | 17.38 | 18.08 |
| 106 | 25.17 | 19.37 | 16.85 | 14.87 | 15.66 | 16.79 | 17.06 |
| 107 | 7.07 | 2.35 | 4.92 | 8.23 | 11.68 | 13.91 | 14.12 |

At tick 0, `L` = 10 cm is better than native or within 0.1 cm of it on 8 of 8 (103's logged native value is a controller-cut motion; its full command would have left 13.53 cm), and `L` = 5 cm is worse on the five far starts. Later ticks depend on the state reached and are not predictable from the baseline log.

## Arm
`cem_cap10`: `plan = clamp(cem_release.plan(ctx), −0.10, 0.10)` elementwise, metres. Same native search, settings, RNG stream and planning time. Harness, evaluator and reach-v0 untouched. Seed 0, dev instances 100–107, paired by instance with `cem_release` and `cem_clampexec` seed 0. A one-line output transform: one implementation, no second independent attempt.

## Prediction (before any capped unit)
1. Progress is positive on 8 of 8 instances and no tick is without executed motion (baseline: 6 of 8; 5 of 40 ticks unmoved).
2. Closer than the baseline by at least 5 cm at the final tick on 100 and 107; final distance within 1 cm of the baseline's on at least 5 of the other 6.
3. Mean progress at least 0.85 (baseline 0.587); median progress at least 0.92 (baseline 0.951).
4. Tick 0: realized distance within 1.0 cm of the `L` = 10 column on at least 5 of the 6 unique-`g` instances.
5. Ordering over `L`. Mean progress: 10 cm > 5 cm > none. Final distance on the near starts (100, 107): 5 cm < 10 cm < none. Median final distance on the five far starts (101, 102, 103, 105, 106): 5 cm above 10 cm by at least 1 cm; 10 cm within 0.5 cm of none.

## Reading on dev (written after the baseline, before any capped unit)
Primary: `bench/score.py` unchanged; incumbency follows AGENTS.md and the scorer's paired comparison of medians. On that headline neither cap is expected to win: the median over 8 instances does not move when 2 fail (baseline median 0.951 with two negative instances).

Secondary, labelled as a post-baseline reading and not an incumbency rule: progress positive on 8 of 8; mean of the paired differences in progress against `cem_release`, with a bootstrap interval over instances (not expected to exclude zero with the effect carried by 2 of 8); median within 0.03 of the baseline's. `run.py` reports this as `decision`. Whether reach-v0's headline should count failures is a change of evaluator: asked of Ashesh (FRONTIER, Blocker 2); until answered no cap is declared incumbent on the secondary reading alone and no held-out seed is used.

## What would change the plan
- 1–3 hold → `cem_cap10` is the candidate: seeds 1–2 for it and for the baseline; incumbency and held-out use wait for the headline question above. Next module after that: the half-step inside the cube (`a_i / r_i` ≈ 0.6), by executing `κ·a` with `κ` set from feedback.
- 100 or 107 still fails at 10 cm while 5 cm fixes it → a fixed cap is too coarse: set it from feedback (halve after a tick whose realized cost rose; the trust radius of P1 applied to the cap).
- 5 cm beats 10 cm in mean progress → executing only what was scored dominates and the 5-tick allowance is what binds for far starts.
- Neither cap beats the baseline's mean progress → saturation is not what causes the failures: log the controller residual on the unmoved ticks before any further arm. That diagnostic (replay the baseline's commands on 100 and 107 with the controller residual, joint margins and workspace projection logged) is the first missing control in any case and is written while this arm runs.

## Outputs
`run.py` (CPU, numpy): per-instance progress and final distance for the three caps, paired differences with a bootstrap over instances, failure and unmoved-tick counts, tick-0 predicted against realized → `out/result.json`, `out/cap.dat`, `out/trajectories.dat`. Units under `runs/E002-planner-loop/cem_cap10/`.

## Verdict
**2026-11-13 23:33Z; `cem_cap10`, seed 0, dev instances 100–107, 8 units, 40 ticks, paired by instance with `cem_release` and `cem_clampexec` seed 0; scored by `bench/score.py` unchanged; one planner seed.** Job `e006-cem_cap10` ran 20:44–23:21Z (211.9 s of planning per tick, median); 8 units exported and SHA256-verified (27 files in `runs/E002-planner-loop/`).

At 10 cm the far starts show no loss above 1 cm (one seed) and one of the two near starts is repaired; the other falls into a period-2 swing that the 5 cm cap does not have. Final distance, cm (start; none; 10 cm; 5 cm): 100 (7.29; 12.32; **1.54**; 0.42), 101 (23.53; 0.38; 0.46; 0.67), 102 (15.81; 0.54; 0.41; 0.32), 103 (26.28; 1.43; 1.38; 7.13), 104 (9.41; 0.21; 0.22; 0.30), 105 (26.93; 1.18; 1.99; 4.69), 106 (25.17; 1.79; 0.84; 5.06), 107 (7.07; 9.69; **5.88**; 0.77). Progress positive on 8 of 8; mean 0.841 (none 0.587, 5 cm 0.888); median 0.957 (0.951, 0.917). Fixed scorer: paired difference of medians against `cem_release` +0.006, 95% interval [−0.030, 0.850]: not an incumbent (the interval contains zero, and 0.006 lies inside the per-instance differences on the six instances that never fail: −0.030 to +0.038). Secondary reading (`run.py`, `decision` true): positive on 8 of 8, mean paired difference +0.254 [−0.002, 0.626], median within 0.03; as written, it declares nothing while Blocker 2 is open. Figure `out/fig-cap.pdf`.

| written before any capped unit | result |
|---|---|
| 1. positive on 8 of 8; no tick without executed motion | holds under `run.py`'s definition, which the prediction's text does not carry: 8 of 8; 0 of 40 ticks execute under 1 mm on a command of at least 1 cm (native 5). Read literally, one tick has no motion: 102 tick 4, 0.15 mm on a 1 mm command at 0.41 cm from the goal |
| 2. closer than native by ≥ 5 cm on 100 and 107; within 1 cm of native on ≥ 5 of the other 6 | first clause fails on 107 (closer by 3.81 cm; 100 by 10.78); second holds on 6 of 6 (largest gaps: 106 closer by 0.95, 105 farther by 0.81) |
| 3. mean ≥ 0.85; median ≥ 0.92 | first clause fails by 0.009 (0.841); second holds (0.957) |
| 4. tick 0 within 1.0 cm of the `L` = 10 column on ≥ 5 of 6 | holds at exactly 5: five within 0.10 cm; 103 off by 1.30 cm (controller cut at 200 substeps, 11.7 cm executed of 17.3) |
| 5. mean: 10 > 5 > none; near starts: 5 < 10 < none; far-start median: 5 above 10 by ≥ 1 cm, 10 within 0.5 cm of none | mean ordering fails (5 cm 0.888 > 10 cm 0.841 > none 0.587; the whole difference is 107); near-start ordering holds on both; far-start medians hold (4.69, 0.84, 1.18 cm) |

**The dose–response, by population.** Far starts (101, 102, 103, 105, 106), median final distance: 1.18 cm with no cap, 0.84 at 10 cm, 4.69 at 5 cm. The native arm commands more than 10 cm on a component in 22 of 40 ticks (37 components); removing that length changes none of the six instances that never fail by as much as 1 cm (105 farther by 0.81, 106 closer by 0.95: differences of the size that trajectory divergence alone could produce, with no rerun to bound it), and the `L∞` budget of E006 (`⌈‖r‖∞ / L⌉` ticks) halves to 3 of 5 ticks for the three farthest. Near starts: 100 ends at 1.54 cm (12.32 native), 107 at 5.88 (9.69). Capped ticks end closer on 14 of 16 (5 cm: 20 of 20; native saturated ticks: 14 of 23). Executed/commanded length within [0.8, 1.2] on 30 of 33 commands of at least 1 cm (the three are 103, as in both other arms). `rho` median 2.36, positive on 35 of 39 moving ticks (scorer, all 40 ticks: 2.49).

**107: a period-2 orbit of the capped step** (post hoc; one instance). Pooling the three arms' units of an instance makes the trilateration of `g` unique where one unit was mirror-ambiguous: 107 has `g` = (3.0, −2.8, 5.8) cm, residual 0.00 mm over 18 distances. Remaining offset on x before each tick of the 10 cm arm: +3.0, −6.9, +3.1, −6.9, +3.1 cm; command on x: +10, −10, +10, −10, +8.8 cm. On a saturated axis the executed step is `L·sgn(r)`, and `r ↦ r − L·sgn(r)` is an involution on `|r| < L`: the orbit `{r, r − L}` persists for as long as the model saturates at both of its points. On x it does so for all five ticks (3.0–3.1 cm remaining, three times; 6.9 cm, twice). On z it does so for two (5.75 → −4.21 → 5.88 cm) and then stops: at 5.88 cm remaining the command is 3.1 cm, unsaturated, where at 5.75 cm on tick 0 it was saturated. Every tick of 107 executes its command in full (31–47 substeps, converged), so the arm is not frozen by the guard. From the same start the 5 cm cap leaves (−2.0, 1.0, 0.7) cm, commands (−1.6, 0.5, −0.4) cm, and converges to 0.77 cm.

What this does not give is a threshold. Saturation depends on the whole state and not on the remaining offset of the axis: 107 saturates x at 3.0 cm, 100 leaves x and y unsaturated at 3.3 and 3.5 cm (commands 2.9 and 3.1 cm), and z on 107 is saturated at 5.75 and not at 5.88. So the clamp at 5 cm is not a report that 5 cm or more remain, and the condition for an orbit under a cap `L` is only the structural one above; no value of `L` is shown safe by these units, and the 5 cm cap is merely orbit-free on the 8 it ran. The tick-0 command of 107 is shared by all three arms, so the orbit's entry is one model decision observed once.

Feedback on that orbit: realized cost rose on the first swing (`rho` −1.74) and fell on the next three (1.07, 1.08, 0.62), including tick 2, where the distance rose from 6.66 to 7.44 cm. The sign of `rho` flags the swing once and then stops; the sign flip of a saturated component flags it on every tick (x: 4 of 4 transitions; z: 1).

**100: the residual is a weak gain inside the cube.** After the capped tick 0 (z: −5.5 → +4.6 cm) the commands on z are 2.4, −0.1, 0.7, 0.3 cm against 4.6, 2.2, 2.3, 1.7 cm remaining (`a_z / r_z` = 0.52, −0.05, 0.30, 0.18), with `rho` 5.6, 3.9, 11.2, 18.6: realized cost falls many times faster than predicted and the step stays short.

**What it does to the account and the plan.** Supported on 8 units, one seed: removing the sampler's length beyond 10 cm leaves the six never-failing instances within 1 cm of native and the median unchanged, which is what the tick-budget reading of the 5 cm cap's far-start cost predicts; and 107 at 10 cm fails to close without any frozen tick, by commands that stay saturated on an axis with 3 cm left, so one near-start failure survives the removal of the guard's trigger. Not established: any of this beyond one planner seed (review: the first missing control is replication, seeds 1–2 or a same-seed rerun of `cem_release` and `cem_cap10`, at least on 100, 105, 106, 107; until then sub-centimetre far-start gaps and the orbit cannot be separated from trajectory divergence); the joint margins of the 10 cm units (E008 replay, not yet run); how often saturation below 5 cm occurs (one instance here; the baseline counted 5 of 32 saturated components with less than 5 cm remaining); anything across planner seeds. Of the branches written above: 1–3 do not all hold (2 and 3 fail narrowly), so `cem_cap10` is not the candidate; "107 still fails at 10 cm while 5 cm fixes it" holds in substance (5.88 against 0.77 cm, progress 0.17) and points to a cap set from feedback; "5 cm beats 10 cm in mean" holds through 107 alone. This is the second dev episode without a gain on the fixed scorer (E006, E007): the step back of AGENTS.md is E010 (search cost inside the cube), already written. The feedback-set cap (start at 10 cm or none, shrink on evidence of a swing: a sign flip of a saturated component, which this orbit shows to be the better signal, or `rho` < 0) is entered in FRONTIER under P1/P3 with this orbit as its test case; its spec is written after E009 says whether the near-start failures are the planner's.

Launch note: job `e006-cem_cap10` launched 2026-11-13 20:44Z (shell PID 26436, Python 26472), seed 0, 8 units, about 2.5 h (to about 23:15Z); the planner's CPU test passed on the pod (4 tests). Known at launch and not when the predictions above were written: E006's verdict (5 cm cap: positive on 8 of 8, mean 0.888, median 0.917; three far starts bound by an `L∞` budget of 5 ticks) and E008's (the native failures are the wrapper's joint-margin guard, entered only on executed moves above 20 cm; the 5 cm arm stays within 0.04 rad of its start margin). Predictions 1–5 stand as written.

Reviewed once (fresh context, Claude, 2026-11-13 23:35Z): every number reproduced from the 24 raw units, `g` of 107 recomputed independently (3.015, −2.786, 5.754 cm). Changed after it: the far-start wording (no loss above 1 cm, not "nothing"), the literal reading of prediction 1, the z swing on 107, the withdrawal of a saturation threshold `s` and of `L < 2s` as a number, the sentence on `rho` (distance rose on tick 2 while the cost fell), and replication named as the first missing control.
