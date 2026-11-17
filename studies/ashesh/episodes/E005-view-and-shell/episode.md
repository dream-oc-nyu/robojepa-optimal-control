# E005 — does direction survive in the view that carries little of the goal-free term, and does the raw cost rank candidates of equal step length?

Arms A3 (objective) and A4 (search). Directions P9 and P1. Written 2026-11-12 about 02:58Z after the E004 verdict and before any E005 quantity was computed. Known beforehand from E001–E004: the per-view foreign response and per-view even part of E004's descriptive block, the raw `all`-scope selection (7/8, instance-01 picks `offray1-minus-0.03`), and the class medians of the raw response. Not looked at: any per-view direction-dependent gain, any per-view concordance or selection, any within-shell ranking.

## Question
E004: the goal-free decrease of predicted cost is concentrated in view 1 (the view that moves with the arm, identified by inspection of instance-00's images only; median foreign response −0.028 at 3 cm against −0.004 in view 0), and a step-length penalty accounts for most of the ranking gain of E003's correction while selecting worse actions.
1. The release's cost weights the two views 0.5/0.5. Does the cost on view 0 alone keep the direction-dependent response and repair signs and ranking, with no reference goals and no rescaling?
2. The goal-free term depends on step length and not on direction, so it should cancel among candidates of equal step length. Does the raw cost rank the 6 candidates of a shell (1 cm or 3 cm) by physical distance? That is the comparison a fixed-radius (trust-region) search makes.

## Quantities (CPU, numpy, the E001 raw arrays; no new collection)
Per-view cost `C_v(z, g) = mean |z[v] − g[v]|`; weighted cost `C_w = w_0 C_0 + w_1 C_1`. Variants: `raw` (0.5, 0.5; `readings.cost`), `view0` (1, 0), `view1` (0, 1), and for reference E004's `disp_channel`, `foreign` and the realized cost. Realized counterparts use the same weights on `z_real`.
- For `view0` and `view1`: the E004 metric set (foreign census over 672 pairs and medians at 1 and 3 cm; own-goal sign census /16, /16, /64; on-ray direction-dependent gain = median over instances of `odd_pred / odd_real` with both in the same view's cost and E004's noise rule; `all`-scope concordance with instance bootstrap; selection /8), through the fixed evaluator.
- Shells: for each instance and each of the two shells (the 6 candidates at 1 cm; the 6 at 3 cm: toward, away, 4 lateral), `readings.concordance` of cost against `D` over the 15 pairs, pooled over instances (120 pairs per shell), split into the 9 class-distinct pairs (toward against the other 5, away against the 4 lateral) and the 6 lateral–lateral pairs; and shell selection = argmin cost within the shell is the toward candidate (/8 per shell). For `raw`, `view0`, `view1`, `disp_channel`, `foreign`, realized.
- Descriptive, no prediction: the share of the raw on-ray odd response carried by each view, `0.5·odd_v / odd_raw`, predicted and realized; `all`-scope concordance and selection along the weight path `w_0 ∈ {0, 0.1, …, 1}`.

Limit: same 8 instances as E001, E003 and E004; exploratory until the fresh sweep.

## Prediction (before running)
1. View 0 keeps direction: on-ray gain in view 0 ≥ 0.30 at 3 cm and larger than view 1's gain at both 1 and 3 cm; toward falls 16/16.
2. View 0 repairs part of the signs and ranking: away rises ≥ 10/16, lateral rises ≥ 24/64, `all`-scope concordance ≥ 62% (raw 47.3%, dispersion-matched 58.3%), selection ≥ 7/8.
3. Raw cost within a shell: pooled concordance ≥ 70% in each shell (raw `all`-scope 47.3%), ≥ 90% on the class-distinct pairs, shell selection 8/8 at 1 cm and 7/8 at 3 cm.
4. Within a shell the corrections add little: `disp_channel` and `foreign` within 5 points of raw on the class-distinct pairs.

## What would change the plan
- 3 holds → the planner for E002 searches direction at fixed radius with the native cost and sets the radius from `rho`; the goal-free term then matters only for the choice of radius, and no objective correction is needed for direction. The norm-penalized CEM stays a baseline only.
- 3 fails (class-distinct below 75%) → the term is not a function of step length alone; an objective correction is needed even at fixed radius, and the candidate is whichever of `view0`, `disp_channel` ranks shells best.
- 1 and 2 hold → the first pod comparison after the baseline is `cem_release` with view weights (1, 0) against the native (0.5, 0.5), a one-parameter change of the release objective, paired on the dev seeds.
- 1 fails (view-0 gain below 0.10) → direction is carried by the moving view; view reweighting is out, and the correction has to act inside view 1 (dispersion matching there, or a subgoal cost).

## Outputs
`run.py` → `out/result.json`, `out/summary.json`, `out/shells.dat`, one figure. `check.py` → `out/check.json` (independent recomputation). Raw arrays are read only.

## Verdict
**2026-11-12 — the direction signal the predictor passes on is stronger in view 0 (gain 0.39 at 3 cm against 0.20 in view 1), and the cost on view 0 alone is the best reference-free predicted cost on these arrays (prediction 1 holds, prediction 2 on three of four clauses). At equal step length the native cost ranks well at 3 cm and not at 1 cm (prediction 3 holds at 3 cm only; prediction 4 likewise).**

Population: the 8 E001 instances, 96 candidates ≥ 1 cm, 672 foreign pairs, 1088 `all`-scope pairs, 120 pairs per shell (72 class-distinct, 48 lateral–lateral), none excluded; raw arrays read only; evaluator imported unchanged; E001, E003 and E004 anchors reproduced. `check.py`, written without sight of `run.py`, agrees exactly on all 160 compared metrics and on the weight path; the odd-response shares and the shell intervals come from `run.py` alone. In all 16 (instance, shell) cases the toward candidate is physically closest and every lateral lies between toward and away. Numbers `out/result.json`, `out/summary.json`, `out/check.json`; figure `out/fig-view-shell.pdf` (plain pgfplots working figure). The title and one phrase of the Question were reworded after the run on the reviewer's note that view 0 is not free of the term (655/672); predictions and thresholds were not touched. Predictions 3 and 4 were computable from E004's stored per-instance costs; they had not been computed when the spec was written.

| cost | foreign < 0 (of 672), median 3 cm | on-ray gain 1, 3 cm | toward falls /16 | away rises /16 | lateral rises /64 | `all` concordance | best action /8 | class-distinct pairs /72, 1 cm, 3 cm | toward chosen in shell /8, 1 cm, 3 cm |
|---|---|---|---|---|---|---|---|---|---|
| native (0.5, 0.5) | 672, −0.0165 | 0.22, 0.26 | 16 | 1 | 1 | 47.3% | 7 | 61, 67 | 6, 7 |
| view 0 only | 655, −0.0039 | 0.25, 0.39 | 16 | 12 | 17 | 62.6% [57.9, 67.4] | 8 | 68, 71 | 7, 8 |
| view 1 only | 672, −0.0284 | 0.21, 0.20 | 16 | 0 | 0 | 43.8% | 7 | 57, 63 | 6, 7 |
| dispersion-matched | — | — | 16 | 9 | 10 | 58.3% | 8 | 65, 68 | 6, 8 |
| foreign baseline | — | — | 14 | 16 | 54 | 83.4% | 7 | 69, 69 | 7, 8 |
| realized, both views | — | 1 | 16 | 16 | 36 | 74.7% | 8 | 67, 71 | 6, 8 |
| realized, view 0 | — | 1 | 16 | 16 | 36 | 73.3% | 8 | 66, 68 | 6, 8 |
| realized, view 1 | — | 1 | 16 | 16 | 35 | 73.8% | 8 | 64, 68 | 6, 8 |

1. **Direction in view 0 (holds).** Gain 0.39 at 3 cm (predicted ≥ 0.30) and above view 1 at both amplitudes (0.25 against 0.21; 0.39 against 0.20); toward falls 16/16.
2. **Signs and ranking with view 0 (three clauses hold, one fails).** Away rises 12/16 (predicted ≥ 10), concordance 62.6% (predicted ≥ 62%, at the margin), selection 8/8 (predicted ≥ 7); lateral rises 17/64 against a predicted ≥ 24.
3. **Native cost at equal step length (holds at 3 cm, fails at 1 cm).** 3 cm: 102/120 = 85.0%, class-distinct 67/72 = 93.1% [81.9, 100], toward chosen 7/8. 1 cm: 82/120 = 68.3% (predicted ≥ 70%), class-distinct 61/72 = 84.7% [70.8, 95.8] (predicted ≥ 90%), toward chosen 6/8 (predicted 8/8).
4. **Corrections within a shell (holds at 3 cm, fails at 1 cm).** Class-distinct difference from native: dispersion-matched +1.4 points and foreign +2.8 at 3 cm; +5.6 and +11.1 at 1 cm (predicted within 5).

**Descriptive.** Weight path `w_0 = 0 … 1`: `all` concordance rises monotonically, 43.8, 44.3, 44.8, 46.0, 46.3, 47.3, 49.5, 51.7, 52.8, 55.6, 62.6%; selection is 7/8 up to `w_0 = 0.5` and 8/8 from 0.6; class-distinct pairs go 57 → 68 (1 cm) and 63 → 71 (3 cm). Share of the on-ray direction-dependent response carried by view 0 (median over instances): predicted 0.43 at 1 cm and 0.52 at 3 cm; realized 0.35 and 0.30. Lateral–lateral pairs are not ordered by any cost at 1 cm (11 to 31 of 48 across the eight costs) and reach 33–37 of 48 at 3 cm. The realized cost itself chooses the toward move in only 6/8 of the 1 cm shells, in every view weighting.

**Reading.** In the realized latents about two thirds of the direction-dependent response is in view 1, the view that moves with the arm. The predictor passes on a fifth of it there (gain 0.20) and contracts that view; in view 0 it passes on 0.39 at 3 cm and adds little of the goal-free term. The native 0.5/0.5 objective therefore takes most of its goal-free term and the weaker half of its direction from view 1. Two handles follow, both without reference goals: weight the objective toward view 0, or compare candidates at equal step length of about 3 cm, where the native cost orders class-distinct pairs in 93%. At 1 cm neither the predicted nor the realized cost chooses reliably (6/8), which bounds what a planner can resolve near the goal with this representation at one tick.

**Plan rule.** Predictions 1 holds and 2 holds but for the lateral sign: by the rule above, the first pod comparison after the baseline is `cem_release` with view weights (1, 0) against the native (0.5, 0.5), paired on the dev seeds. Prediction 3 falls between its branches at 1 cm (84.7%, between 75% and 90%), so no pre-registered branch applies there; the choice made after seeing the result is a fixed-radius direction search with the radius kept at or above about 3 cm and set from `rho`, with view weighting inside it as the second factor.

Scope: one kitchen, 5 cm goals, one model tick, one checkpoint, the 8 instances reused from E001, E003 and E004; intervals are over 8 instances; view identity rests on one instance's images. Exploratory until the fresh sweep (seeds 1008–1015). Not reviewed by a fresh context (the one review of this checkpoint covered E004 and this spec). Not run: paired intervals between view 0 and native, any closed-loop use, horizon above 1, a second scene or checkpoint, whether view 0's advantage survives when the arm leaves that camera's field of view.
