# E002 — closed-loop planners on reach: CEM, gradient, trust-region Gauss–Newton

Arm P1 (feedback trust region) and P2 (curvature). Written 2026-11-12, before any E002 data and before E001 results.

## Question
On the release's own reach task with the released 22M predictor, at matched planning time, does a closed-loop curvature planner that calibrates its step from realized cost beat the release's CEM, and does it beat a gradient planner without feedback?

## Why (from E000)
Write the model Jacobian as `J_m = G·J`. A gradient step scales with `G`; a Gauss–Newton step with `1/G`. The trust-region ratio `rho = dC_real / dC_pred = 1/G` is measurable after every executed tick, because the encoder cost tracks physical progress (20 of 21 pairs in E000). Gain error is therefore correctable online; direction error is what remains.

## Prediction (before running)
1. Release CEM at native settings makes progress on at least 6 of 8 dev instances and leaves median final distance above 20% of the start distance.
2. Median `rho` over executed ticks is above 3: the model under-predicts cost change.
3. At matched planning time, trust-region Gauss–Newton ends closer than CEM and closer than Adam on at least 6 of 8 dev instances.

## What would change the plan
- CEM already above 95% progress → no headroom on reach → move to the release's next harder task or a quarter of the time budget.
- Every planner below 10% progress → predictor or action interface binds → next episode calibrates the interface (feed predicted executed motion) before more planner work.
- TR-GN ties CEM → measure direction error (model cost gradient against finite-difference realized gradient in the simulator) before more variants.

## Benchmark reach-v0 (fixed; changing it is a new benchmark version)
- **Base:** the release's own RoboCasa reach planning evaluation in the pinned RoboJEPA source: its controller mode, goal image construction, action bounds, tick length and success metric. Record file and lines. If its goal construction cannot be used, fall back to a goal image from the endpoint of a seeded 4-command sequence of 5 cm steps, and say so.
- **Model:** released 22M/150k predictor, frozen V-JEPA 2.1-G encoder, FP32.
- **Instances:** dev seeds 100–107. Held-out seeds 200–215 are not run in this episode.
- **Allowance:** the release's reach step count, else 15 executed ticks.
- **Score per instance:** `progress = 1 − D_T / D_0`, with `D` the physical end-effector distance to the goal position from simulator state (evaluator only). Also the release's own reach metric.
- **Budget:** `B` = measured planning seconds per tick of release CEM at native settings on this GPU. Every planner gets `B` per `plan()` call. The harness times it and records overruns.
- **Planner contract:** `bench/interface.py`. Planners live in `planners/<name>.py` and never see simulator state.
- **Per-tick log:** commanded and executed motion, predicted / realized / model-null costs (the `Feedback` fields), `rho`, physical distance, planning seconds, predictor calls.

## Planners
- `cem_release` — the release's CEM, unmodified, native settings. The honest baseline.
- `adam` — projected Adam on the native mean-L1 cost through the differentiable rollout, warm-started, no feedback.
- `trgn` — damped Gauss–Newton on a pseudo-Huber-smoothed L1 (IRLS weights), step limited by a trust radius that is updated from `rho` after every executed tick.
Stochastic planners: 3 seeds per instance.

## Outputs
Raw per-run logs under `runs/E002-planner-loop/` (gitignored). `bench/score.py` (CPU, numpy only) turns them into `episodes/E002-planner-loop/out/results.md`, pgfplots `.dat` files and one appended row per planner in `bench/leaderboard.jsonl`.

## Calibration read (2026-11-13; one unit, not a scored unit, not the verdict)
Raw: `runs/E002-planner-loop/_calibration/inst100-seed0/unit.json`, `budget.json` (exported, SHA256-verified). Population: dev instance 100, planner seed 0, 5 ticks, release CEM at native settings, FP32.

- **Budget.** Planning seconds per tick 305.3, 216.2, 216.0, 215.7, 216.1 (15 CEM iterations on tick 0, 10 after); `B` = median = **216.1 s**. One unit costs about 21 min of GPU; 8 units 2.8 h, 24 units 8.4 h.
- **Outcome.** Distance to goal 7.29 → 11.56 → 9.93 → 12.92 → 12.92 → 12.93 cm; progress −0.77; release reward 0 on every tick. Commanded step lengths 17.5, 21.1, 22.9, 10.0, 21.3 cm from a start 7.3 cm away. `rho` −0.24, −3.49, 0.56, −0.02, −0.01 (median −0.02; prediction 2 said above 3). Predicted cost fell on 5/5 ticks; realized cost rose on ticks 0, 1, 3 and 4 (by at most 4e-4 on the two motionless ticks) and fell on tick 2, where the arm moved away from the goal (9.9 → 12.9 cm): on that tick the encoder cost and the physical distance disagree.
- **Interface.** Ticks 0–2 executed the command (error ≤ 0.8 cm on ≥ 17 cm). Ticks 3–4 executed nothing (≤ 0.1 mm for 10 and 21 cm commands; 200 controller substeps, not converged, as on tick 2). Cause unread and the controller residual is not logged; the wrapper clamps targets to a 0.40 m workspace sphere (`blocking_robot_env.py:336-347`, `robot_env.py:264`) and applies a joint-margin guard (`robot_env.py:1984ff`).
- **Why the steps are long.** With planner embodiment 2 the model receives `clamp(a / 0.05 m, −1, 1)` per axis (`embodiment_preprocessing.py`, `to_canonical12_from_droid7`), while the blocking controller executes the metric `a` in full. The native search ball has radius 0.30 m. Beyond 5 cm on an axis the predicted cost is exactly constant in that component, so the search fixes its sign and its length is set by the sampler, not by the model; the elite cost spread falls below the log's precision (4 decimals) as the sampling spread shrinks. In this unit 7 of 15 commanded components exceed 5 cm, at least one on every tick. The release's own client sends the same unclamped command (`policy_client.py:742,1248-1300`), so this is the release's native reach behaviour. Each clamped action is also copied over 4 sub-actions, so inside 5 cm the model-side and the executed displacement need not match either (ratio unmeasured), and the context actions of earlier ticks reach the model through the same clamp. This is separate from the goal-free term of E003, which was measured with embodiment 3. Reviewed once (fresh context, source and raw unit re-read; the numbers reproduced).

Expectations for the remaining instances, written at launch, before any baseline unit completed (the three predictions above stand and are judged on the 8 instances):
4. At least 28 of the 40 seed-0 ticks command a component above 5 cm. This counts how often the clamp is active; it does not by itself separate the clamp from other causes of long steps.
5. Final distance is set by the step scale and not by the start: median final distance over the 8 instances is at least 8 cm, every instance that starts within 10 cm ends farther than it started, and no instance ends within 2 cm of the goal.
6. Unit inst100-seed0 reproduces this calibration unit's commanded actions to 0.1 mm (native CEM is not stopped at `B`, so the two runs are intended to be the same computation; floating-point ties may break it).

If 4 holds, the next episode is the action interface, with the control that separates length from direction: the same CEM output executed as `clamp(a, ±5 cm)` per axis (what the model was shown), and the same CEM searching inside that cube, each against this baseline; view weights (E005) moves behind it. If 4 fails, this unit was atypical and view weights goes next as planned. If 6 fails, planner-seed variance is measured (seeds 1–2) before any paired comparison.

Run order: seed 0 for instances 100–107 first (8 units), seeds 1–2 afterwards.

## Early read of the first baseline unit (2026-11-13 15:50Z; 1 of 8 units, not the verdict)
Raw: `runs/E002-planner-loop/cem_release/inst100-seed0/unit.json` (exported while the job runs; SHA256-verified). Analysis: `episodes/E006-action-interface/prior.py`, recomputed by `check_prior.py`.

- **Expectation 6 fails.** Against the calibration unit the commanded actions differ by 0.90, 8.72, 3.16, 4.00, 3.79 mm per tick. At tick 0 the difference is on the saturated axis alone (z: −16.99 against −17.08 cm; x and y agree to 0.01 mm): where the predicted cost is flat, ties decide. Outcome is the same in kind: distance 7.29 → 12.32 cm (progress −0.69; calibration −0.77), no executed motion on ticks 3–4.
- **Goal offset from the logs.** The executed displacements and the 12 logged distances of the two units fix the goal offset by trilateration: `g = (3.27, 3.45, −5.52)` cm, residual 5e-9 m, unique (independent grid search agrees to 3e-5 mm).
- **What the planner commanded against what remained.** Tick 0: `a = (2.93, 3.07, −17.0)` cm against `g`: the two axes inside the 5 cm cube are right to 0.4 cm, the saturated axis has the right sign and three times the length. Over both units, sign of the commanded component equals the sign of the remaining offset in 20/20 components with at least 3 cm remaining and 6/8 with less than 1 cm; `cos(a, r)` is 0.89–0.99 on every tick. `clamp(a_0, ±5 cm)` would have left the arm 0.73 cm from the goal (counterfactual, not executed); the full command left it at 11.6 cm.
- Population: one instance, two runs of the same planner seed, ticks 3–4 from an unmoved arm. E006 (`episodes/E006-action-interface/episode.md`) was written from this read, before the other seven units.

## Verdict
**Baseline part (2026-11-13 18:00Z; `cem_release`, seed 0, dev instances 100–107, 8 units, 40 ticks; scored by `bench/score.py` unchanged; reviewed once). Prediction 3 (Adam, trust-region Gauss–Newton) is not run; the episode stays open for it.**

Native CEM ends within 2 cm of the goal on 6 of 8 instances and farther than it started on the other 2. Progress: 100 −0.69, 101 0.98, 102 0.97, 103 0.95, 104 0.98, 105 0.96, 106 0.93, 107 −0.37; median 0.951, mean 0.587. Final distance 0.21–1.79 cm on the six (start 9.4–26.9 cm) and 12.3, 9.7 cm on 100 and 107 (start 7.3, 7.1 cm, the two nearest). Release success (2 mm) 0 of 8. Planning 206–216 s per tick after the first (291–300 s), 2.5 h in all.

| written before | result |
|---|---|
| 1. progress on ≥ 6 of 8; median final distance > 20% of start | first clause holds at exactly 6; second fails: median final distance is 4.9% of start (1.30 cm) |
| 2. median `rho` > 3 | fails: median 1.53 (5–95%: −1.9 to 15.0; positive on 31 of 40, above 3 on 12) |
| 3. trust-region Gauss–Newton against CEM and Adam | not run |
| 4. ≥ 28 of 40 ticks command a component above 5 cm | fails: 23 of 40 (tick 0: 7 of 8 instances) |
| 5. median final ≥ 8 cm; starts within 10 cm end farther; none within 2 cm | all three fail: 1.30 cm; 2 of 3 (104 starts at 9.4 cm and ends at 0.21); 6 end within 2 cm |
| 6. same-seed rerun reproduces the commands to 0.1 mm | fails (early read): 0.9–8.7 mm |

The calibration unit was atypical in outcome, not in the clamp: instance 100 is one of the two failures.

**Where the regret arises** (goal offsets by trilateration, `episodes/E006-action-interface/prior.py`, recomputed by `check_prior.py`; `g` unique on 6 instances, mirror-ambiguous on 104 and 107, whose rows are excluded from the per-axis counts).
- *Inside the 5 cm cube the loop contracts.* On the 17 ticks with no saturated component the arm ends closer on 16, and the distance falls by a median factor of 0.53 per tick. The commanded component is a fraction of the remaining offset: `a_i / r_i` median 0.59 (IQR 0.38–0.93; 32 components with `|r_i|` ≥ 1 cm, sign right in 31). Instance 104 never saturates: 9.41 → 4.95 → 2.81 → 1.29 → 0.57 → 0.21 cm.
- *Outside it the model gives a sign and the sampler gives a length.* 32 saturated components on the unique-`g` instances: sign right in 31, length 9.0–21.7 cm (median 14.3), past the remaining offset in 22 and by more than 5 cm in 17 (one more has the wrong sign); in 5 the remaining offset was itself below 5 cm, down to 0.7 cm, so whether an axis saturates is not a reliable sign of a large offset. Instances 101 and 103 receive the same tick-0 command to four decimals, (10.58, 12.20, 16.93) cm, for goal offsets (14.2, 14.3, 12.2) and (1.9, 22.2, 14.0) cm: with all three axes saturated the command is the sampler's draw. Saturated ticks (23): closer on 14, farther on 4, unmoved on 5.
- *Both failures begin with a saturated tick-0 overshoot, as several successes do (102 moves away on tick 1 and recovers); what only the failures show is an arm that stops moving.* 100: `a_0` = (2.93, 3.07, −17.08) against `g` = (3.27, 3.45, −5.52) cm. 107: `a_0` = (15.17, −3.79, 12.75) against a start 7.07 cm away. Both then swing 20 cm per tick, the controller stops converging (200 substeps on 3 and 4 ticks) and the arm does not move on the last 2 and 3 ticks. All 5 unmoved ticks of the 40 are in these two instances; why the arm stays unmoved is unread (controller residual not logged). The two failures are also the two nearest starts, so start distance and saturated overshoot are not separated here; 104, the third start within 10 cm, never saturates and succeeds.
- *Controller.* Executed length within [0.8, 1.2] of commanded on 33 of 40 ticks; not converged on 10. Instance 103 is cut short at 200 substeps on ticks 0 and 2 (11.6 of 23.4 cm, 10.9 of 23.9 cm); the tick-0 cut cost it (16.5 cm against 13.5 for the full command) and the tick-2 cut ended 13 cm closer than the full command would have (1.36 against 14.5 cm).
- *`rho`* rises along a clean approach (104: 1.4, 2.9, 4.4, 16.5, 24.0): the predicted decrease shrinks faster than the realized one near the goal. It is not a constant gain.

**Plan.** The headroom rule (CEM above 95% progress) is met by the median at 0.951 and not by the mean at 0.587: reach-v0 stays, since the regret is two total failures, a 0.2–1.8 cm residual, and 212 s of planning per tick (median; `B` = 216.1 s) for a 3-dimensional search. Expectation 4 failed, and its branch pointed to view weights; the action interface goes next all the same, as a stated deviation, because both failures and every unmoved tick follow a saturated tick-0 overshoot (launch note in `episodes/E006-action-interface/episode.md`); that view weights would not act on it is a judgement, not a measurement. The cap on that length is run as a dose–response: 5 cm (E006, launched), 10 cm (E007), none (this baseline). Clamping at 5 cm is not expected to dominate: at tick 0 it would leave 10.9–20.7 cm on the five far instances against 6.3–18.1 cm native, and 0.73 cm against 11.6 on instance 100. Baseline seeds 1–2 before any held-out use.

Limits: one planner seed; 8 instances; the saturation account is read from logged commands and trilaterated offsets, not from an intervention (E006 and E007 are the interventions); per-axis counts exclude 104 and 107; FP32 where the release runs FP16. Missing controls, in order: the controller residual and joint or workspace state on the unmoved ticks (the failure is a frozen arm and its cause is unread); baseline seeds 1–2 (on fully saturated ticks the command is the sampler's draw); more near starts, to separate start distance from saturation; CEM searching inside the cube, the control that separates length from direction.
