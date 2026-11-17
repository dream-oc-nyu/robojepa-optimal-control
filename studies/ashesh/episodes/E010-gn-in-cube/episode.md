# E010 — what does the 212 s of search buy? Gauss–Newton inside the 5 cm cube

Arms P2 (curvature) and P3 (search inside the cube). It also runs the part of E002's prediction 3 that the data still supports. Written 2026-11-13 21:40Z, with the 8 native units, the 8 units of the 5 cm cap and both E008 replays read; with `cem_cap10` (E007) running and unread beyond liveness (1 unit published, not opened); with E009 unrun; and before any unit of a gradient or curvature planner on the model.

## Question
At horizon 1 the plan is a point `a ∈ R³`, and with planner embodiment 2 the model scores `u = clamp(a / 5 cm, −1, 1)` per axis. The predicted cost `L(a) = C(F(h, a))` is therefore a function on the box `Q = [−5, 5]³` cm, constant along each axis beyond it. Native CEM locates its minimum with 15,360 rollouts on tick 0 and 10,240 on later ticks (293 and 212 s). How much of that minimum, and of the closed-loop outcome of `cem_clampexec`, does a Gauss–Newton search on `Q` recover with at most 150 rollouts per tick?

## Why now
- E002's own rule: "CEM already above 95% progress → … a quarter of the time budget". The native median is 0.951, and E008 attributes the two failures to a wrapper guard, not to the search.
- Every arm so far costs 2.5 h of GPU for 8 units, of which 2.3 h is search. A search module that costs seconds makes each later module experiment (objective, gain, view weights, horizon) a 20-minute arm.
- Inside the cube the loop already contracts (distance × 0.53 per tick, `a_i / r_i` median 0.59, sign right in 31 of 32): the predicted cost there has a usable minimum. What is unmeasured is how smooth the path to it is at the scale of a finite difference.
- `docs/literature-chest/RETRIEVAL.md`, "Does local optimal control offer the right guarantee?": native L1 is kept as the accepted cost (no silent change of objective between solvers); a finite-radius response and a local derivative answer different questions, so the Jacobian is a central difference at a centimetre scale, not autodiff.

## Maths
Features `z(a) ∈ R^P` (`P` ≈ 10⁶), goal `g`, residual `r = z − g`, native cost `L = ½ Σ_v mean|r_v|` over the two views. With `J = ∂z/∂a ∈ R^{P×3}` from central differences of half-width `s` (6 rollouts), the IRLS weights `w = 1/√(r² + δ²)` give the step

`d(λ) = −(JᵀWJ + λI)⁻¹ JᵀW r`,  `a ← Π_Q(a + d)`,

accepted only if the native `L` falls. One linearisation costs 7 rollouts against CEM's 1,024 per iteration. A central difference at `a = 0` cancels any term that depends on `‖a‖` alone (the goal-free decrease of E003, measured under embodiment 3); the acceptance test on `L` does not, so the two can disagree, and the per-tick log of accepted and rejected steps is where that would show.

## Arm
`gn_cube`: search on `Q` as above, executed = planned (every component within ±5 cm, so the model scores exactly what the controller executes, as in `cem_clampexec`). No sampler, no RNG. At most 150 rollouts per `plan()`. The feedback-set radius of P1 is not used: `rho` is logged by the harness and left for its own episode, so that the only module changed against `cem_clampexec` is the search. Harness, evaluator, reach-v0, seed 0, dev instances 100–107; paired by instance with `cem_clampexec` and `cem_release` seed 0.

Two independent implementations from the same brief (`planners/gn_cube_a.py`, gpt-6-astra; `planners/gn_cube_b.py`, Claude), disjoint files; both are run, both are reported, and the one with the higher mean progress on the dev seeds is kept. The predictions are judged on each.

Reference, tick 0 of `cem_clampexec` (the same reset state; 15,360 rollouts): predicted decrease `L(0) − L(a)` = 0.0371, 0.0423, 0.0467, 0.0309, 0.0319, 0.0384, 0.0264, 0.0242 for instances 100–107; commands (2.93, 3.07, −5), (5, 5, 5), (5, −5, 3.84), (5, 5, 5), (−3.56, −1.07, 2.85), (5, −4.8, 5), (−1.82, −5, 5), (5, −3.79, 5) cm.

## Prediction (before any unit)
1. *The same minimum.* At tick 0 the predicted decrease is at least 0.90 of CEM's on at least 6 of 8 instances.
2. *The same command.* At tick 0 the command is within 1.5 cm of `cem_clampexec`'s on at least 5 of 8.
3. *At a hundredth of the search.* Median planning time at most 10 s per tick (CEM 211 s), at most 150 rollouts per tick (CEM 10,240).
4. *The same outcome.* Progress positive on 8 of 8; mean progress at least 0.80 (`cem_clampexec` 0.888); final distance within 1.5 cm of `cem_clampexec`'s on at least 5 of 8 (0.42, 0.67, 0.32, 7.13, 0.30, 4.69, 5.06, 0.77 cm).
5. *No incumbent on the fixed headline.* The scorer's paired difference of medians against `cem_release` is negative or its interval contains zero, as for `cem_clampexec` (−0.034 [−0.139, 1.262]); the gain is in planning time.

## What would change the plan
- 1–4 hold → the search is not what limits reach-v0, and 212 s is not needed to find what the model offers. `gn_cube` becomes the working search for the module experiments that follow (gain `κ` from feedback, view weights, objective), each a 20-minute arm; and the control owed to the bar is run at once: native CEM cut to the same number of rollouts per tick (`cem_small`, elites kept at ≥ 4), paired with `gn_cube`. Curvature is credited only if it beats that.
- 1 fails → the predicted cost is not well approximated by a quadratic at the finite-difference scale, or the box face stalls the step: read the cost along each axis (21 rollouts per axis, three axes, on the eight tick-0 states) and the dependence on `s` (1, 2, 4 cm) before any variant. This is P8, direction error, moved forward.
- 1 holds and 4 fails → the same model minimum gives a worse loop: the spread of CEM's mean was doing work (dither against the half-step). Read the per-tick `a_i / r_i` of both arms; next is the gain arm P4.
- 4 beats `cem_clampexec` by more than 0.05 in mean progress → compare predicted costs tick by tick: either CEM is under-converged at 10 iterations or a lower predicted cost is not what helps.
- 3 fails by time with the rollout count held → per-call overhead dominates: batch the line search with the linearisation.

## Not in this episode
Matched-rollout CEM (`cem_small`; follows at once if 1–4 hold); the `rho` radius (P1); horizon above 1 (P5); seeds 1–2 (the planner is deterministic; reset identity is checked at tick 0 by `cost_before` and `cost_model_before` against the CEM units); held-out seeds.

## Outputs
Units under `runs/E002-planner-loop/gn_cube_{a,b}/` (reach-v0 units, scored by `bench/score.py` unchanged). `run.py` (CPU, numpy): tick-0 predicted decrease and command against `cem_clampexec`, planning seconds and rollouts per tick, per-instance progress and final distance for `cem_release`, `cem_clampexec`, `gn_cube_a`, `gn_cube_b`, paired differences with the scorer's own bootstrap → `out/result.json`, `out/search.dat`, `out/progress.dat`, one figure (final distance against rollouts per tick, log axis). Pod jobs `e006-gn_cube_a`, `e006-gn_cube_b`, about 20 minutes each, started only on an idle GPU with no session job running, after E009.

## Verdict
_pending_ — not implemented; spec written first.
