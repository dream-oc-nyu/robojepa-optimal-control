# E011 — a length rule from feedback: bisection on a saturated sign flip

Arms P1 (radius from feedback) and P3 (action interface). Written 2026-11-14 01:00Z, with E002, E006, E007, E008 and E009 judged; with E010 unjudged (`gn_cube_a` complete on the pod and `gn_cube_b` running when the `hsrn` login expired at 00:46Z; neither exported; read so far: `gn_cube_a` on instances 100 and 107 from the job log, final 0.31 and 0.53 cm, and the planner-log lines of `gn_cube_b` on 100–102); and before any census of sign flips beyond the two trajectories quoted in E007 and E009 (107 at 10 cm, 100 with the guard off: a flip on 4 of 4 transitions each).

## Question
Planner embodiment 2 shows the model `u_i = clamp(a_i / 5 cm, −1, 1)`. On an axis where the search returns `|a_i| ≥ 5 cm` the model has scored a sign and no length; the length that is executed comes from elsewhere (the sampler: 9–21.7 cm; a fixed cap: 5 or 10 cm). No fixed length is safe: 5 cm spends the tick budget of the far starts (`⌈‖r‖∞ / L⌉`), 10 cm cycles on 107 (`r ↦ r − L·sgn r`, period 2), the sampler's lengths end in the wrapper guard or, with the guard off, in a saturated overshoot on 5 of 5 ticks (100). Can the length be set from the one signal that saw every one of those cycles, the sign flip of a saturated component, so that near starts converge and far starts keep their speed?

## Maths
One axis, remaining offset `r`, a saturated command of sign `s = sgn(a)` (right in 31 of 32 saturated components of the native units) and executed length `ℓ`. If the next command on that axis is saturated with the opposite sign, the last step passed the goal: `|r| < ℓ`. That is a bracket `B = ℓ`. With a sign oracle and `|r| ≤ B`, the step `s·B/2` gives `|r′| = | |r| − B/2 | ≤ B/2`: the bound halves on each saturated tick, where a fixed length keeps `|r|, ℓ − |r|, |r|, …`. Rule, per axis `i`, with `L₀` = 10 cm and state `B_i` (none at reset):

- `|a_i| < 5 cm`: execute `a_i` (the model scored it).
- saturated, no bracket: if the previous command on `i` was saturated with the opposite sign and that tick moved, set `B_i = |e_i|` of that tick; else execute `s·L₀`.
- saturated, bracket: execute `s·max(B_i/2, 1 cm)`, then `B_i ← B_i/2`.

From `L₀` = 10 cm the lengths after a flip are 5, 2.5, 1.25 cm: once an axis is bracketed, what is executed is inside the cube again. Far starts: an axis with `|r|` up to 25 cm takes three ticks at 10 cm, and a bracket on tick 3 leaves `|r| ≤ 2.5 cm` after tick 4 in the worst case. The rule is wrong where the sign is wrong: a false flip on a far axis caps it at 5 cm and below for the rest of the unit. How often that happens is part A.

## Part A — census of flips on the units in hand (CPU, no model)
Population: the exported seed-0 units, `cem_release` (8), `cem_clampexec` (8), `cem_cap10` (8), `cem_release_guardoff` (2): 26 units, 130 ticks, 104 transitions, 312 axis-transitions. Goal offset `g` per instance by trilateration on the pooled units of that instance; an instance whose `g` stays ambiguous is excluded and named. A *flip* at tick `t` on axis `i`: `|a_i| ≥ 5 cm − 10 µm` at `t−1` and `t`, opposite signs, and tick `t−1` moved (≥ 1 mm). A *true bracket*: `sgn e_i = sgn r_i` and `|e_i| > |r_i|` at `t−1`. Capped arms log the capped command, so saturation there is `|a_i|` at the cap or beyond 5 cm; frozen ticks (command ≥ 1 cm, executed < 1 mm) are counted apart.

Prediction (before the census):
- A1. *Precision.* At least 0.85 of flips are true brackets.
- A2. *No far-axis false alarm.* No flip on an axis with 10 cm or more remaining at `t`.
- A3. *Coverage.* Of saturated-axis overshoots of at least 2 cm at `t−1` (`t−1 ≤ 3`, tick moved), at least 0.70 are followed by a flip at `t`; the rest are ticks where the model is no longer saturated.
- A4. *One step of the rule.* At a flip, replacing the logged step on that axis by `s·|e_i(t−1)|/2` leaves `|r_i|` smaller than the logged step did on at least 0.80 of flips, and never above 5 cm.

## Part B — the arm on the model
`bisect`: the rule above on top of a search that returns a point with saturation visible. Search: `gn_cube` (the arm E010 keeps) if E010's predictions 1–4 hold on it, with the saturated face read as `|a_i| = 5 cm`; otherwise native CEM. Reference, same search, rule off: `sat10`, saturated components executed at 10 cm (with native CEM this is `cem_cap10`, except that a sampler length between 5 and 10 cm is raised to 10 cm). Only the feedback rule differs between `bisect` and `sat10`. Two independent implementations (`planners/bisect_a.py`, gpt-6-astra; `planners/bisect_b.py`, Claude). reach-v0, seed 0, dev instances 100–107, fixed scorer; E008 replay on both arms.

Prediction (before any unit; reference values from `cem_cap10`: 1.54, 0.46, 0.41, 1.38, 0.22, 1.99, 0.84, 5.88 cm, mean progress 0.841):
- B1. 107 ends within 2 cm under `bisect`; under `sat10` it ends beyond 4 cm with a flip on x on at least 3 of 4 transitions.
- B2. No instance ends more than 1 cm farther under `bisect` than under `sat10`; the far starts 103, 105, 106 end within 2 cm (median under the 5 cm cap: 5.06 cm).
- B3. Progress positive on 8 of 8; mean progress at least 0.90, above `sat10` by at least 0.03 and above the 5 cm cap (0.888).
- B4. No guard substep in the replay of either arm.
- B5. Fixed headline: median at least 0.95; the paired difference of medians against `cem_release` has an interval that contains zero.

## What would change the plan
- A1 below 0.75 or A2 violated → a flip alone is not a bracket: require a second witness (`rho` < 0 on the tick that overshot, or two flips) and write that rule before any unit.
- A3 below 0.5 → most overshoots are not seen by flips: read what the model does on the tick after (unsaturated recovery is harmless; a wrong sign is P8).
- A holds, B1 fails with flips logged → the bracket is true and the model's sign inside it is not: the half-step lands where the model is saturated the wrong way; read `a_i / r_i` per tick, next is the gain arm P4.
- B2 fails on a far start → a false bracket cost the tick budget: the census number A2 was too small a sample; add the second witness.
- B1–B4 hold → `bisect` is the interface for the module experiments that follow, and the claim owed to the bar is replication (seeds 1–2 for a CEM search; a deterministic search has one draw per instance, so held-out instances are the replication) and the length-matched control of P3.

## Not in this episode
The `rho` radius; gain `κ` inside the cube (P4); `L₀` other than 10 cm; horizon above 1; held-out seeds.

## Outputs
`run.py` (CPU, numpy): part A → `out/census.json`, `out/flips.dat`, one figure (remaining offset on the flipped axis before and after, logged against bisected). Part B units under `runs/E002-planner-loop/<arm>/`, scored by `bench/score.py` unchanged.

## Verdict
Part A (2026-11-14 01:25Z; `run.py`, the 26 units, 130 ticks, 312 axis-transitions; 7 instances with a unique `g`, instance 104 excluded as mirror-ambiguous with no flip and no overshoot under either solution; 28 flips; the flip count was recomputed from the raw units by a second script: 28, by arm 14 / 1 / 6 / 7; one draw per instance and arm; reviewed once). **A1 holds, A2, A3 and A4 fail as written.** A1: 26 of 28 flips are true brackets (0.93). A2: 9 flips have 10 cm or more remaining. Eight are true brackets after sampler-length steps of 15–22 cm, whose overshoot alone leaves more than 10 cm (6 native, 2 guard-off, none in the capped arms): the prediction did not allow for them. The ninth is the case it was written for: `cem_clampexec`, instance 105, y, a capped step of the wrong sign (+5 cm with −6.3 cm remaining) followed by the right sign reads as a flip with 11.3 cm remaining, and a shrinking bracket would strand that axis. A3: 26 of 42 overshoots of 2 cm or more are followed by a flip (0.62, written ≥ 0.70); of the other 16, 15 are followed by an unsaturated command and 1 by a saturated command of the same sign (`cem_clampexec`, 105, z). A4: the half step leaves less than the logged step on 20 of 28 (0.71, written ≥ 0.80) and up to 8.77 cm (3 above 5 cm: one true bracket from a 15.9 cm sampler step, native 105 z, 6.08 cm, and the two false flips). Post hoc, on the one arm that executes the lengths the rule would (`cem_cap10`): 6 flips on 2 of 8 units (103, 107), 6 of 6 true brackets, the half step leaves at most 2.05 cm on 6 of 6 and less than the logged step on 5 of 6. `rho` does not serve as the second witness: on the tick that overshot it is negative for 2 of those 6 flips (one tick, 107 tick 0) and it reads 0.71 on the wrong-sign tick of the false flip. The flips are not independent: 19 of 28 come from instances 100 and 107, 6 instances have any flip (all true on 4 of them: 100 8 of 8, 102 1 of 1, 103 4 of 4, 106 2 of 2; 105 1 of 2, 107 10 of 11), and 4 of the 7 guard-off flips repeat the native units' first ticks (same tick 0); without those, 22 of 24 are true brackets and 7 flips have 10 cm or more remaining. Figure `out/fig-flips.pdf`; rows in `out/census.json`.

Reading. A saturated flip marks a crossing, the last step passed the goal, on 26 of 28; with the saturated sign right on 31 of 32 native components that is largely implied by the sign accuracy, not a separate finding. The two that are not brackets are sign errors of two kinds: the earlier sign wrong on a far axis (`cem_clampexec` 105 y, 11.3 cm remaining), and the later sign wrong 0.47 cm short of the goal after an undershoot (`cem_release` 107 x, tick 2, a frozen tick). A bound taken from one flip is wrong in the first kind, on about 6 instances with flips and one draw each. The branch "a flip alone is not a bracket" is taken.

## Revision after part A (2026-11-14 01:25Z, before any unit of part B)
No bracket is taken from `|e|`. Per axis the saturated length `L_i` starts at `L₀` = 10 cm and is halved on each saturated sign flip (10, 5, 2.5, 1.25, floor 1 cm), never otherwise; a saturated component is executed at `s·L_i`. This is sign-only step adaptation, the decrease of Rprop without its increase. On a true bracket `|r| < L` the halved step leaves `|r′| < L/2` or a second flip; on a false flip the cost is a factor of two in length on one axis for the rest of the unit, and a further factor for each later flip on it. That bound is argued, not measured: the one far-axis false flip in hand is from the 5 cm arm. Against bisection it contracts one tick later when the command stays saturated on the same side. B1–B5 stand as written, for this rule. `planners/bisect_a.py` and `planners/bisect_b.py` implement the unrevised rule (CPU tests pass; unregistered, unrun): each author revises its own file before registration.

## Addendum before part B (2026-11-14 01:35Z, no unit run; exact arithmetic only): what halving on flips alone gives up, and a second schedule
Take the case the census supports on 26 of 28 flips: the sign is right and the flip is a true bracket. After a flip at length `L` the remaining offset satisfies `|r| < L`, and the halved step `L/2` leaves `|r′| < L/2`. If the next command is saturated on the *same* side, then `0 < |r′| < L/2` and the halve-on-flip rule steps `L/2` again: it passes the goal with certainty and lands in `(0, L/2)` on the other side. The bound contracts on flips only. Halving on every saturated tick after the first flip steps `L/4` there and the bound follows `L/2, L/4, L/8` (the unrevised schedule, with the length taken from `L_i` and never from `|e|`).

One axis, a sign oracle that is always saturated and always right, 5 ticks, start offsets on a 0.1 mm grid; final `|r|` in cm, worst / mean:

| start `|r₀|` | fixed 10 cm (`sat10`) | halve on flip (`bisect`) | halve after first flip (`halvesat`) |
|---|---|---|---|
| 0–5 cm | 9.99 / 7.50 | 2.49 / 0.94 | 1.00 / 0.42 |
| 5–10 cm | 4.99 / 2.50 | 1.25 / 0.58 | 1.00 / 0.42 |
| 10–25 cm | 10.00 / 5.83 | 5.00 / 1.56 | 2.50 / 0.83 |

Instance 107, `g` = (3.0, −2.8, 5.8) cm, same oracle: `sat10` ends at (7.0, 7.2, 4.2) → 10.89 cm (measured with the model, `cem_cap10`: 5.88 cm, because the model is not always saturated); `bisect` at (2.0, 2.2, 0.8) → 3.08 cm; `halvesat` at (0.25, 0.05, 0.55) → 0.61 cm. So B1 is not implied by the halve-on-flip rule: it holds only if the model leaves saturation inside the bracket on x or y. B1 stands as written, as a test of that.

What the second schedule risks is the false flip. After a flip at tick `t` = 1, 2, 3, 4 an axis can still travel 9.75, 8.75, 7.5, 5 cm under `halvesat` and 20, 15, 10, 5 cm under `bisect`. The one far-axis false flip in hand (11.3 cm remaining) would end at least 1.55 cm short under `halvesat` had it come at tick 1 at these lengths.

Arm added: `halvesat` (config `shrink='saturated'`): per axis, once a flip has occurred, `L_i` is halved on that flip and on every later saturated tick of that axis whose previous tick moved; floor 1 cm; no `|e|`. `bisect` (`shrink='flip'`) and `sat10` unchanged. Same search, same units, same replay.

Prediction (before any unit):
- C1. `bisect` and `halvesat` execute the same commands on at least 5 of 8 instances (they differ only after a flip followed by a saturated tick on the same axis; `cem_cap10` had flips on 2 of 8 units).
- C2. On 107 `halvesat` ends at least 1 cm closer than `bisect`, and within 2 cm.
- C3. On the far starts 103, 105, 106 `halvesat` ends no more than 1 cm farther than `bisect`; mean progress of `halvesat` is at least that of `bisect`.

What would change the plan:
- C2 fails with identical commands on 107 → the model leaves saturation after the first flip; the schedule is immaterial and the simpler rule (`bisect`) is kept.
- C3 fails on a far start with a false flip in the planner log → sign errors on far axes are frequent enough to price: the length rule becomes bisection under a sign-error rate (a posterior over the offset per axis, stepped to its median; probabilistic bisection), its own episode, with its primary sources added to the literature chest first.
- C2 and C3 hold → `halvesat` is the feedback rule carried forward, and B3 is read on it as well.

Code state (2026-11-14): `planners/bisect_a.py` (gpt-6-astra) and `planners/bisect_b.py` (Claude) implement `sat10`, `bisect` and `halvesat` on three searches each (CEM, `gn_cube_a`, `gn_cube_b`); the two files agree command for command on 96,000 scripted plans under both schedules, and the closed loop through the registered code reproduces the table above (CPU, no model); registered in the dev driver and launcher; no unit run.

Readings settled before any unit (2026-11-14 01:50Z; implemented in `run_b.py`, which judges B1–B5 and C1–C3 and reports B1–B3 on `halvesat` as secondary). "Within", "at least" and "no more than" are inclusive, "beyond" and "above" strict. A tick moved when the norm of the executed vector is at least 1 mm. A flip under `sat10` is read from the logged base commands with part A's definition; under the other arms from the planner log. B4 covers `sat10` and `bisect`; the replay of `halvesat` is reported beside it. C1 compares commanded vectors, the same when every component differs by less than 0.1 mm on all five ticks. B3's "above the 5 cm cap" reads `cem_clampexec`'s mean progress from its units. Geometry pools every complete arm of an instance under `runs/E002-planner-loop`. The two implementations compute the same function, so one is run (`_a`); launch order `sat10`, `bisect`, `halvesat`.

Part B: _pending_.
