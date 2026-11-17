# E003 — is the motion term of the predicted cost goal-free, and can it be subtracted?

Arms A1 (predictor) and A3 (objective). Direction P9. Written 2026-11-12 after the E001 verdict and before any E003 quantity was computed (no cost against a foreign goal, no prediction error and no feature statistic has been looked at).

## Question
E001: for commands ≥ 1 cm the predicted cost falls for 94 of 96 moves, including 15/16 away from the goal and 63/64 lateral. The predictor never sees the goal, so `z_pred(a)` must move toward `g` in L1 for almost any `a`. Is that a property of the predicted latent alone (it would then also move toward unrelated goals), and does a goal-free baseline remove it?

## Quantities (CPU, numpy, the E001 raw arrays; no new collection)
For instance `i`, candidate `a`, own goal `g_i = z_real_i(goal)`, foreign goals `g_j`, `j ≠ i` (7 per instance, same scene, other start and direction), and `C(z, g) = mean |z − g|` exactly as in `readings.cost`:
- foreign response `F_pred(a, j) = C(z_pred_cmd,i(a), g_j) − C(z_pred_cmd,i(0), g_j)` and its realized control `F_real(a, j)` with `z_real` in place of `z_pred_cmd`;
- baseline `b_i(a) = mean_j F_pred(a, j)` and corrected cost `C_corr(a) = C(z_pred_cmd,i(a), g_i) − b_i(a)`, which a planner can compute (it needs reference latents, no simulator state);
- descriptive, no prediction: prediction error `mean |z_pred_cmd(a) − z_real(a)|` and the per-token dispersion of `z_pred_cmd(a)` against amplitude; the own-goal sign census recomputed against a moving reference (the 1 mm on-ray prediction in place of the zero-action one), to see whether the zero-action prediction is the outlier.

Added 2026-11-12 about 01:52Z, still before any E003 quantity was computed; descriptive controls with no prediction attached:
- the same correction applied to the realized cost (`C_real − mean_j F_real`), to see whether subtracting a foreign-goal baseline damages an ordering that was already good (realized `all`-scope concordance 74.7%);
- physical approach to the foreign goal, `|ee_i(a) − ee_j(goal)| − |ee_i(0) − ee_j(goal)|`, so that a negative `F_real` can be read against real motion toward `g_j`;
- shifted condition: the baseline from each single foreign goal (7 variants) and the executed-input prediction `z_pred_exec`, reported as ranges of the prediction-2 and prediction-3 counts.

Limit: the foreign goals come from the same scene, so a goal-free result here is goal-free within one kitchen.

`C_corr` is scored by the fixed E001 evaluator's pair concordance and selection on the `all` scope, unchanged, alongside the raw predicted cost.

## Prediction (before running)
1. `F_pred < 0` for ≥ 85% of the 672 (candidate ≥ 1 cm, foreign goal) pairs; `F_real < 0` for 35–65% of them.
2. With `C_corr`, cost rises for ≥ 12 of the 16 away moves (raw: 1/16) and for ≥ 40 of the 64 lateral moves (raw: 1/64); toward moves keep 16/16 decreases.
3. `all`-scope concordance of `C_corr` with `D` ≥ 60% (raw 47.3%); selected action physically best in 8/8 (raw 7/8).

## What would change the plan
- 1 and 2 hold → the term is a removable, goal-free baseline. Next episode: `cem_release` on reach-v0 with and without the baseline-subtracted cost, paired on the dev seeds; one fresh sweep (seeds 1008–1015) confirms the sign census prospectively.
- `F_pred` splits like `F_real` (35–65%) → the decrease is specific to the own goal; map where it lives on the 15×20 token grid (arm against background) before any objective change.
- 1 holds, 2 fails → the term is goal-free but not additive; test a direction-preserving alternative (cosine or per-token normalized cost) on the same arrays.

## Outputs
`run.py` → `out/result.json`, `out/foreign_response.dat`, one figure. Raw arrays are read only.

## Verdict
**2026-11-12 — the any-motion decrease of predicted cost is goal-free (prediction 1 holds, 672/672). Subtracting it repairs the away and lateral signs; predictions 2 and 3 each miss one clause, both on instance-01.**

Population: the 8 E001 instances (start/direction seeds 1000–1007, scene seed 0), the 96 candidates ≥ 1 cm, 7 foreign goals each (672 pairs), none excluded. Raw arrays read only. The evaluator's `cost`, `concordance` and `selector` are imported unchanged, and the E001 census, concordance and selection (94/96; 515/1088; 7/8) were reproduced through the same code path before use. The seven prediction counts were recomputed by a second, independent script and agree exactly. Numbers: `out/result.json`; figure `out/fig-goal-free.pdf` (plain pgfplots working figure).

1. **Goal-free (holds).** `F_pred < 0` in 672/672 pairs (toward 112/112, away 112/112, lateral 448/448) against a predicted ≥ 85%. `F_real < 0` in 359/672 = 53.4% (per instance 22 to 64 of 84) against a predicted 35–65%. The size of `F_pred` follows amplitude and not direction: median −0.008 at 1 cm and −0.0165 at 3 cm in every class. `F_real` follows physical approach to the foreign goal (same sign in 456/672, Spearman 0.43); `F_pred` does not (Spearman 0.18).
2. **Signs after subtraction (two clauses hold, one fails).** With `C_corr` the cost rises for 16/16 away moves (raw 1/16; predicted ≥ 12) and 54/64 lateral moves (raw 1/64; predicted ≥ 40). It falls for 14/16 toward moves against a predicted 16/16: instance-01's two toward moves come out at +0.0003 and +0.0014, because its baseline (−0.0051, −0.0115) is as large as its raw response (−0.0047, −0.0101).
3. **Ranking and selection (one holds, one fails).** `all`-scope concordance of `C_corr` with `D` is 907/1088 = 83.4% [78.1, 88.1] against a predicted ≥ 60% (raw 47.3%, realized cost 74.7%). The selected action is physically best in 7/8 against a predicted 8/8: instance-01 picks a 1 cm lateral move (regret 29.9 mm; raw 37.9 mm).

Shifted conditions: a single foreign goal as baseline, all 7 choices, gives away 16/16, lateral 50–55/64, toward 13–14/16, concordance 870–908/1088 and 7/8 (with executed-input predictions: toward 13–15/16, concordance 869–912); executed-input predictions with the mean baseline give 16/16, 53/64, 14/16, 905/1088 and 7/8. Control: the same correction leaves the realized cost nearly unchanged (813 → 822 of 1088, 8/8, census 16/16, 0/16, 29/64). Note that the corrected cost rises for more lateral moves (54/64) than the realized cost does (36/64).

**Descriptive (no prediction was attached).** The predicted latent contracts as the command grows: token dispersion 0.478 at rest, 0.46 at 1 cm, 0.445 at 3 cm, 0.435 at the 5 cm goal command, and mean token L1 norm 859, 845, 826, 814, in every direction class, while the realized latent stays at 0.52 and 902. A latent with less dispersion is closer in L1 to any target it is not aligned with, which is one sufficient account of a goal-free decrease; it has not been tested by intervention. Taking the 1 mm prediction as reference in place of the zero-action one leaves the census unchanged (16/16, 15/16, 63/64), so the zero-action prediction is not an outlier (E001's first open alternative). Prediction error `mean|z_pred − z_real|` grows from 0.147 at rest to 0.22 at 3 cm, 15 to 20 times the cost responses being ranked.

**Reading.** The subtraction removes the direction-independent term; it does not restore the directional gain. Median corrected responses are −0.003 and −0.008 toward (realized −0.014, −0.063), +0.004 and +0.015 away (realized +0.009, +0.033), +0.001 and +0.007 lateral (realized −0.000, +0.005) at 1 and 3 cm. The foreign-goal baseline is larger than the own-goal direction-independent term (median `F_pred` −0.008 and −0.0165 against E001's on-ray even part of about −0.007 and −0.014), so `C_corr` also carries a penalty on amplitude. Because `D` rises with amplitude for away and lateral moves, part of the 83% may come from that penalty; a pure amplitude penalty `λ‖a‖` was not run as a control. Corrected predictions ordering more pairs than the realized cost (907 against 813) is read the same way: the realized cost is flat to non-monotone for lateral moves, and nothing here shows the corrected model to be better than the encoder.

**Plan rule.** Prediction 1 holds and prediction 2 fails on one clause: the term is goal-free and not exactly additive (the baseline over-subtracts for the near own goal, and one instance has no directional signal left after it). By the rule written above, E004 tests direction-preserving alternatives on the same arrays, aimed at the contraction (dispersion-matched predictions, normalized costs), with the amplitude penalty as a control. The with/without comparison under `cem_release` on reach-v0 follows and needs the pod.

Scope: one kitchen, 5 cm goals, one model tick, one checkpoint; the same 8 instances E001 and its post-hoc reading used, so the corrected-cost numbers are exploratory until a fresh sweep (seeds 1008–1015). Not run: a second scene, a second checkpoint, foreign goals from another kitchen, the amplitude-penalty control, any closed-loop use of `C_corr`.
