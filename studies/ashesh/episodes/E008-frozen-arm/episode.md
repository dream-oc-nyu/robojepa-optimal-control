# E008 — why the arm stops: replay of the baseline's commands with the controller instrumented

Arm P3 (action interface), the first missing control of E002's verdict. Written 2026-11-13 18:45Z from the 8 seed-0 baseline units alone, before any replay and with E006 still running (2 of its 8 units complete, unscored, not used here). No model and no planner in this episode: simulator and release controller only.

## Question
Both baseline failures (instances 100, 107) end with an arm that executes nothing for 2 and 3 ticks (≤ 0.1 mm on commands of 10–21 cm, 200 controller substeps, not converged). What stops it, and can any command leave that state?

## What the logs already exclude
On all 5 unmoved ticks the command points back along the path the arm had just travelled: cosine between the command and the last executed displacement −0.88, −0.98 (instance 100, ticks 3–4) and −1.00, −0.88, −0.88 (107, ticks 2–4). A wall, a workspace boundary or a physical joint stop blocks from one side; retreat into the space just crossed stays free. The freeze is direction-independent.

The release wrapper has one direction-independent rule. Per controller substep, `RoboCasaEnv._reach_apply_orientation_hold` (`robot_env.py:1984–2107`, called from `blocking_robot_env.py:356–371`) computes the smallest joint margin

`m = min_j min(q_j − q_j^lo, q_j^hi − q_j)`

and, when `m < 0.15` rad, sets the position command to zero in all three axes; only the rotation corrective toward the home orientation remains, `ω ∝ log(R_home R^T)`. The source expects that corrective to carry the joint out of the margin. If the orientation is already at home, `ω = 0`: position command zero, rotation command zero, and the state `{m < 0.15, R = R_home}` is a fixed point of the closed loop for every commanded offset. The Panda arm has 7 joints for a 6-dimensional pose, so `q` is path-dependent: 107 freezes 6.2 cm from its own validated start after a 20 cm excursion and return.

The workspace sphere cannot do it: the blocking wrapper first projects the target onto the 0.40 m sphere around home, and for a target `t` with `‖t‖ ≤ R` and an arm at offset `a`, `a·(t − a) ≤ 0` when `‖a‖ ≥ R` and the chord stays inside otherwise, so the substep projection never scales a step toward `t`.

Both failures enter on almost the same displacement: 100 tick 2, (−12.5, 1.4, −18.5) cm; 107 tick 1, (−11.7, 0.5, −16.7) cm; cosine 0.999 between the executed displacements (0.998 between the commands; an earlier draft of this line gave 0.997). Both entry ticks moved the full commanded length and still did not converge.

Instance 103 shows the other regime: ticks 0 and 2 stop halfway (11.6 of 23.4 cm, 10.9 of 23.9 cm, 200 substeps) and the next tick moves freely (21 cm in 79 substeps), so the 200-substep cap is not a rate limit there (about 2.7 mm per substep).

## Method
`pod/replay.py` rebuilds the fixed harness's environment (`bench.harness.build_env`, same resets), steps the logged `commanded` vectors of each baseline unit as float32, and records per substep, around the unmodified hold function: position command before and after, `m`, the limiting joint and side, rotation error `‖log(R_home R^T)‖`, distance from home, and the workspace scale recomputed from the source's formula; per tick: raw and projected target, residual, executed displacement, distance to goal. After the 5 logged ticks, six probes of ±5 cm along each base axis are executed from the final state. Instances 100–107 in harness order, one fresh environment each.

## Prediction (before any replay)
1. *The replay is the logged trajectory.* Start distance equal to the logged one to 1e-6 m on 8 of 8; executed displacement and distance to goal within 1 mm of the log on at least 38 of 40 ticks.
2. *Mechanism.* On each of the 5 unmoved ticks the guard zeroes the position command on at least 190 of 200 substeps (`m < 0.15` rad) and the rotation error stays below 0.01 rad.
3. *Entry.* The guard first fires inside the last moving tick (100: tick 2; 107: tick 1), never before it; the limiting joint and side are the same in both instances.
4. *Absorbing (shifted condition).* All 6 probes execute less than 0.1 mm on 100 and 107 (12 of 12); on the six successful instances at least 34 of 36 probes execute at least 4 cm.
5. *One-sided stalls (lower confidence).* On 103's two cut ticks the guard fires on between 10 and 190 substeps with `m` within 0.02 rad of 0.15 at the end of the tick, and on no substep of the tick that follows.
6. The substep workspace projection scales no substep (0 of all recorded); the tick-level target projection is active on at most 4 of 40 ticks.

## What would change the plan
- 2–4 hold → the two failures are a fixed point of the release wrapper, entered by long executed swings and left by no command: this regret belongs to the feedback/controller module and is invisible to predictor and objective. The caps of E006 and E007 are then read as avoidance of that set, and the quantity to report across arms is entry rate against executed step length (replay each arm's units with the same script: minutes, no model). Blocker 2 is restated to Ashesh with this in hand: reach-v0's failure count measures entries into an environment guard.
- 2 fails (guard silent on the unmoved ticks) → record contact pairs and controller torque saturation on those ticks; the saturated-overshoot account of the failures has no mechanism yet.
- 4 fails on 100 or 107 (a probe moves the arm) → the state can be left: an escape move triggered by `executed ≈ 0` is a feedback rule worth an arm (P1).
- 1 fails → the simulator is not reproduced from commands and replay cannot carry counterfactuals; the residual and margin would have to be logged by the harness, which is a harness change and its own episode.

## Outputs
Raw: `runs/E008-frozen-arm/replay.json` (exported from `<root>/e008/out/`, SHA256-verified, gitignored). `run.py` (CPU, numpy) → `out/result.json`, `out/margin.dat`, `out/probes.dat`, one figure: `m` against substep for 100, 107 and 103 with the 0.15 rad line. Pod job `e008-replay`, started only on an idle GPU with no session job running (the environment renders; a few minutes).

## Verdict
**2026-11-13 20:42Z; replay of the 8 seed-0 `cem_release` units (40 logged ticks, 48 probes, 6032 controller substeps), simulator and release controller only, hold function unmodified and instrumented; analysed by `run.py` as written before the replay.** Job `e008-replay` ran 20:29–20:37Z; `runs/E008-frozen-arm/cem_release/replay.json`, SHA256-verified. Code as committed in b0d5ea3 ran at first use.

Both baseline failures end in a state of the release wrapper that nothing tried here leaves. On each of the 5 unmoved ticks the joint-margin guard zeroes the position command on 200 of 200 substeps, and none of 12 probes of ±5 cm moves the arm more than 0.01 mm (two instances, deterministic replay, 5 logged ticks and 6 probes each).

| written before any replay | result |
|---|---|
| 1. the replay is the logged trajectory (8 of 8 starts to 1e-6 m; ≥ 38 of 40 ticks within 1 mm) | holds: 8 of 8, 40 of 40, with no difference at the printed precision |
| 2. guard zeroes the command on ≥ 190 of 200 substeps of each unmoved tick; rotation error < 0.01 rad | holds: 200 of 200 on all 5; rotation error ≤ 0.0006 rad; `m` = 0.121 rad (100) and 0.113 rad (107) against the 0.15 threshold |
| 3. guard first fires inside the last moving tick; same limiting joint and side in both | first clause holds (100: tick 2, substep 45; 107: tick 1, substep 40; each tick then travels its full 22.4 and 20.4 cm before the guard holds, 155 and 160 substeps); second fails: joint 1 lower on 100, joint 3 lower on 107 |
| 4. all 6 probes < 0.1 mm on 100 and 107; ≥ 34 of 36 probes ≥ 4 cm elsewhere | holds: 12 of 12 (largest 0.010 mm); 35 of 36 (103, +x: 0.49 cm) |
| 5. 103's cut ticks: guard on 10–190 substeps, `m` within 0.02 rad of 0.15 | fails: the guard fires on 0 substeps and `m` ≥ 0.90 rad throughout |
| 6. substep workspace projection scales no substep; target projection on ≤ 4 of 40 ticks | holds: 0 of 6032; 0 of 40 |

- *Entry.* The guard is entered only during long executed moves. Of the 14 logged ticks that execute at least 15 cm, 2 enter it (both above 20 cm) and 9 pass below `m` = 0.35 rad; of the 17 ticks that execute 1–15 cm none enters and 1 passes below 0.35. Margins at the first substep are 0.175 (107), 0.32, 0.41, 0.475 (100), 0.53, 0.84, 0.87, 1.13 rad: instance 107 is reset 0.025 rad above the threshold, and the two failures are two of the four lowest start margins as well as the two nearest starts.
- *No drift out of it.* Inside the state `m` changes by less than 1e-3 rad over 9 ticks: upward on 100 (0.12081 → 0.12134), flat to falling on 107 (0.11277 → 0.11262). The threshold is 0.03–0.04 rad away, against an episode of 5 ticks.
- *Two different joints.* The nearly identical entry displacements (cosine 0.999) reach the margin on joint 1 in one instance and joint 3 in the other: the common factor is the displacement, not one joint's range.
- *The capped arm never approaches the guard (shifted condition: the same replay on the 8 `cem_clampexec` units, 20:38–20:43Z, 40 of 40 ticks reproduced, `out-cem_clampexec/`).* The guard fires on 0 substeps; on every instance the lowest margin of the episode is within 0.04 rad of the margin at its first substep (107: 0.175 rad, never lower), where the native arm drops by up to 0.51 rad (106: 0.84 → 0.33); 47 of 48 probes move at least 4 cm. The exception is again 103 +x (0.3 cm), and 103's three stalled ticks at 5 cm are the same one-sided effect: +x commanded, −0.1 to −0.6 cm executed, margins above 1.1 rad.
- *103 is a second, different effect.* Its stalls are one-sided and not the guard: the wrapper passes on the same norm-capped position command as on free ticks (norm 0.4), yet the end effector moves 0.1–0.2 mm per substep against 5 mm on a free tick, with 13–14 cm of residual, so the block is downstream of the wrapper; the +x probe is blocked and the other five are free, joint margins stay above 0.90 rad and the workspace projection is inactive. Cause unread: contact with the scene or the controller's own limits (contact pairs and controller torques were not recorded; `contact_count` is 24–35 on every substep, moving or not).

**What it does to the plan.** Predictions 2 and 4 hold and 3 holds in its first clause only (entry inside the last moving tick; the limiting joint differs). The branch written for "2–4 hold" is taken on that partial pass, since it rests on the mechanism and not on which joint: the frozen-arm regret of the native baseline belongs to the controller module and is invisible to predictor and objective, and the 5 cm cap of E006 is read first as avoidance of that set (its replay, `out-cem_clampexec/`, gives the capped arm's margins). This does not undo E006's tick-0 result: both failures had already overshot before entry (100: 7.3 → 11.6 → 10.8 cm; 107: 7.1 → 14.1 → 9.7 cm). Whether native CEM would have recovered from those overshoots without the guard, as it does on 102, is untested, so the 2-of-8 failure count of reach-v0 cannot yet be assigned to the planner or to the guard; the control is native CEM on 100 and 107 with the guard disabled (E009). Blocker 2 is restated with this. Reviewed once (fresh context; counts reproduced from the raw replay; the lead sentence, the drift bullet and this paragraph were tightened after it). Limits: one planner seed's commands; 8 instances; deterministic replay, so the 40 of 40 match is of the simulator, not of planner noise.

Code: written 2026-11-13 18:45Z after the predictions were committed (45890b3): `pod/replay.py`, `pod/run.sh`, `pod/launch.sh`, `pod/export.py`, `run.py`; 7 CPU tests with a fake environment and synthetic fixtures (`python3 -B episodes/E008-frozen-arm/pod/test_replay.py`). Simulator, job runner and transport were first exercised by this run. After the run, one change to `run.py`: the figure's panel spacing. Launch `bash episodes/E008-frozen-arm/pod/launch.sh <planner>`; export `python3 episodes/E008-frozen-arm/pod/export.py <planner>`; analyse `python3 episodes/E008-frozen-arm/run.py [--planner <planner> --out <dir>]`.
