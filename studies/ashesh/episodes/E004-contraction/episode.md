# E004 — does the contraction of the predicted latent carry the goal-free term, and can it be undone without reference goals?

Arms A1 (predictor) and A3 (objective). Direction P9, the "goal-free but not exactly additive" branch of E003. Written 2026-11-12 02:14Z after the E003 verdict and before any E004 quantity was computed (no rescaled latent, no per-view cost, no amplitude-penalty score has been looked at).

## Question
E003: the predicted cost falls toward every goal as the command grows (672/672), and the predicted latent loses dispersion at the same time (0.478 at rest, 0.445 at 3 cm; realized flat). Is the loss of dispersion what produces the goal-free decrease? If the dispersion is restored, does the decrease go away while the direction-dependent response stays? And is E003's foreign-goal baseline anything more than a penalty on step length?

## Why (intuition to be tested)
Write a prediction as `z = m + s·x` about its token mean, and a target as `g = m' + y`. For a target that shares little with `x` (another arm pose, a different wrist view), `mean|z − g|` grows with `s`, so a smaller `s` lowers the cost to any such target. For a target aligned with `x` (the realized outcome of the same action) a smaller `s` raises it, which fits the prediction error growing with amplitude in E003. The own goal at 5 cm sits between the two, which fits the foreign baseline over-subtracting.

## Quantities (CPU, numpy, the E001 raw arrays; no new collection)
Cost variants, each computable by a planner from `z_pred(a)`, `z_pred(0)` and goal latents, scored with the E001 evaluator's `cost`, `concordance` and `selector` unchanged:
- `raw`: `C(z_pred(a), g)`.
- `foreign`: E003's `C_corr` (reference, recomputed as a guard: 907/1088, 16/16, 54/64, 14/16, 7/8).
- `dispersion-matched`: per view and channel, `z̃(a) = μ(a) + (z(a) − μ(a))·σ(0)/σ(a)`, with `μ`, `σ` the mean and standard deviation over that view's 300 tokens of the prediction and `σ(0)` taken from the zero-action prediction; then `C(z̃(a), g)`. A scalar version (one factor per view) is reported beside it.
- `cosine`: `1 − mean over tokens of cos(z_token(a), g_token)`.
- `amplitude penalty` (control, uses no model beyond the raw cost): `C(z_pred(a), g) + λ‖a‖`, `λ` chosen on 7 instances to maximize `all`-scope concordance and scored on the 8th, over the 8 folds.

For each variant: the foreign response census of E003 (672 pairs, fraction negative, median at 1 and 3 cm), the own-goal sign census (toward falls /16, away rises /16, lateral rises /64), the direction-dependent gain on the goal ray (`[dC(+a) − dC(−a)]/2` over its realized value), `all`-scope concordance with `D` with the instance bootstrap, and selection (physically best /8).

Descriptive, no prediction: the foreign response and the own-goal direction-independent term split by camera view (agent view, wrist view); `F_pred` against the starting cost `C(z_pred(0), g_j)` over the 56 (instance, foreign goal) pairs, as a check that less aligned targets get a larger decrease; the contraction split into arm tokens and background tokens if a mask can be derived from `|z_real(goal) − z_real(0)|`.

Limit: same 8 instances as E001 and E003, one kitchen; every number here is exploratory until the fresh sweep (seeds 1008–1015).

## Prediction (before running)
1. Dispersion matching removes most of the goal-free term: median foreign response at 3 cm above −0.0066 (raw −0.0165, at least 60% removed) and `F < 0` in at most 75% of the 672 pairs (raw 100%).
2. It keeps direction: toward falls 16/16, away rises ≥ 12/16, lateral rises ≥ 32/64, and the on-ray direction-dependent gain stays within 0.15–0.35 at 1 and 3 cm (raw 0.22, 0.26).
3. The amplitude penalty reaches `all`-scope concordance ≥ 78% held-out (foreign baseline 83.4%, raw 47.3%), i.e. most of E003's ranking gain is a step-length penalty.
4. Ordering of `all`-scope concordance: foreign ≥ amplitude penalty > dispersion-matched > cosine > raw.

## What would change the plan
- 1 and 2 hold → the contraction is the carrier and is correctable inside the objective with no reference goals. Next: `cem_release` on reach-v0 with the dispersion-matched cost against the native cost, paired on the dev seeds (pod).
- 1 fails (less than 30% removed) → dispersion is a correlate, not the carrier; locate the term by view and token before any objective change.
- 3 holds and 1 fails → the practical handle is step length; go to P1 (trust radius from `rho`) with a norm-penalized CEM as the cheap baseline, and treat the foreign baseline as that penalty in disguise.
- 3 fails (held-out concordance below 65%) → the foreign baseline carries goal-specific structure a norm penalty lacks; keep it as the objective candidate for reach-v0.

## Operational definitions (fixed 2026-11-12 about 02:41Z, before `run.py` existed and before any E004 quantity was computed)
Predictions are `z_pred_cmd`; reference `0` is `neutral0`; a view is one of the 2 camera grids (300 tokens × 1664 channels each); everything in float64.
- Dispersion: `μ_{v,c}(a)`, `σ_{v,c}(a)` = mean and population standard deviation over the 300 tokens of view `v`, channel `c`. Channel version: factor `σ_{v,c}(0)/σ_{v,c}(a)` (1 where `σ_{v,c}(a) = 0`). Scalar version: factor `mean_c σ_{v,c}(0) / mean_c σ_{v,c}(a)` per view. The channel version is the one predictions 1, 2 and 4 refer to.
- Cosine: `1 − mean over the 600 tokens of cos(z_t(a), g_t)`, token vectors of 1664 channels.
- Amplitude penalty: `‖a‖` = norm of the commanded translation in metres; `λ` on the grid `0, 0.01, …, 3.00` (cost per metre); per fold the `λ` with the largest pooled `all`-scope concordance on the 7 training instances, smallest `λ` on ties; held-out concordance = pooled over the 8 held-out instances (1088 pairs), instance bootstrap as in E003.
- Foreign response of a variant: the variant's cost of candidate `a` against `g_j` minus that of `neutral0`, over the 672 pairs; medians at 1 cm and 3 cm over the 336 pairs each. For `foreign` it is `F_pred(a, j)` minus the mean over the other 6 foreign goals (centred by construction, reported for completeness). "Removed" in prediction 1 = `1 − median_3cm(variant)/median_3cm(raw)`.
- Direction-dependent gain: per instance `odd_v(a) = [dC_v(+a) − dC_v(−a)]/2` on the goal ray at 1 and 3 cm, divided by the same quantity of the realized cost (raw L1 realized cost for every variant except `cosine`, which uses the realized cosine cost), null below 10× the neutral-repeat noise; the reported gain is the median over instances, the statistic that gave E001's 0.22 and 0.26.
- Descriptive addition, no prediction: `norm only`, cost `‖a‖` alone (the `λ → ∞` limit of the penalty).
- Arm tokens (descriptive): per view the 30 tokens with the largest `mean_c |z_real(goal) − z_real(0)|`; background = the other 270.
- Independent recomputation: a second script (`check.py`, written without sight of `run.py`) recomputes every count and median named in the predictions.

## Outputs
`run.py` → `out/result.json`, `out/variants.dat`, one figure. Raw arrays are read only.

## Verdict
**2026-11-12 — restoring the dispersion removes two thirds of the size of the goal-free term and almost none of its sign (665/672 still negative) (prediction 1 split, prediction 2 fails on the away and lateral signs). A step-length penalty reproduces most of E003's ranking gain while selecting worse actions (predictions 3 and 4 hold).**

Population: the 8 E001 instances, 96 candidates ≥ 1 cm, 672 (candidate, foreign goal) pairs, 1088 `all`-scope pairs, none excluded; raw arrays read only; evaluator imported unchanged. Guards reproduced E001 and E003 (515, 813 and 907 of 1088; 672/672; 16/16, 54/64, 14/16; gains 0.221, 0.264). The per-variant metrics of the table (foreign census and medians, sign census, gains, concordance, interval, selection) were recomputed by `check.py`, written without sight of `run.py`; the two agree to 2e-16 for all 8 variants. The descriptive block, the shifted condition and the λ curve come from `run.py` alone. Numbers `out/result.json`, `out/summary.json`, `out/check.json`; figure `out/fig-contraction.pdf` (plain pgfplots working figure; the cosine and norm-only points in its left panel are in other units).

| cost | foreign response < 0 (of 672) | median at 1, 3 cm | toward falls /16 | away rises /16 | lateral rises /64 | on-ray gain 1, 3 cm | `all` concordance | best action /8 |
|---|---|---|---|---|---|---|---|---|
| raw | 672 | −0.0078, −0.0165 | 16 | 1 | 1 | 0.22, 0.26 | 47.3% | 7 |
| dispersion-matched (channel) | 665 | −0.0024, −0.0056 | 16 | 9 | 10 | 0.25, 0.29 | 58.3% [56.3, 60.3] | 8 |
| dispersion-matched (scalar) | 664 | −0.0023, −0.0055 | 16 | 9 | 10 | 0.25, 0.29 | 58.7% | 8 |
| cosine | 671 | (other units) | 16 | 4 | 2 | 0.28, 0.36 | 50.8% | 7 |
| foreign baseline (E003) | 311 (centred) | +0.0000, +0.0002 | 14 | 16 | 54 | 0.29, 0.24 | 83.4% [78.1, 88.1] | 7 |
| raw + λ‖a‖, held-out | 89 | +0.0010, +0.0099 | 10 | 16 | 62 | 0.22, 0.26 | 78.8% [74.4, 82.4] | 3 |
| ‖a‖ alone | 0 | — | 0 | 16 | 64 | 0 | 60.5% | 0 |
| realized cost | 359 | −0.0004, −0.0005 | 16 | 16 | 36 | 1 | 74.7% | 8 |

1. **Dispersion matching (split).** The median foreign response at 3 cm goes from −0.0165 to −0.0056, 66% removed against a predicted ≥ 60% (69% at 1 cm). The response stays negative in 665/672 pairs against a predicted ≤ 504. The scalar version (one factor per view) does the same.
2. **Direction (two clauses hold, two fail).** Toward moves fall 16/16 and the on-ray gain is unchanged (0.25, 0.29; predicted within 0.15–0.35). Away moves rise in 9/16 (predicted ≥ 12) and lateral moves in 10/64 (predicted ≥ 32). The selected action is physically best in 8/8 for both dispersion-matched versions; the difference from raw and from the foreign baseline is one instance (instance-01). The best candidate is always the 3 cm toward move, so on this candidate set a cost that still rewards step length selects well.
3. **Amplitude penalty (holds, at the margin).** Held-out concordance 857/1088 = 78.8% against a predicted ≥ 78%; every fold picks λ = 0.88 per metre, the in-sample optimum, so the held-out score is in effect in-sample; only λ from 0.85 to 0.91 reaches 78%, and the interval's lower end (74.4%) is below the threshold. It selects the best action in 3/8: at that λ the penalty at 3 cm (0.026) equals the median raw toward response there (−0.026), and toward moves fall in only 10/16. The on-ray gains of the penalty and of `‖a‖` alone in the table equal raw and 0 by construction (a symmetric penalty cancels in the odd part).
4. **Ordering (holds).** foreign 83.4 ≥ penalty 78.8 > dispersion-matched 58.3 > cosine 50.8 > raw 47.3.

Shifted condition: with executed-input predictions the dispersion-matched cost gives 665/672, −0.0055 at 3 cm, 16/16, 9/16, 9/64, gains 0.19 and 0.29, 627/1088, 8/8.

**Descriptive (no prediction attached).** The term is concentrated in view 1. Per-view costs: in view 1 the foreign response is negative in 672/672 with medians −0.0138 and −0.0284 at 1 and 3 cm; in view 0, 655/672 with −0.0014 and −0.0039. Predicted dispersion in view 1 falls from 0.418 at rest to 0.394, 0.359 and 0.343 at 1 cm, 3 cm and the 5 cm goal command (realized 0.467 to 0.479); in view 0 it falls from 0.479 to 0.469 at 3 cm (realized 0.510). The metadata gives the view map `[0, 2]` without camera names; in instance-00's images view 0 is a third-person view of the arm and view 1 moves with the arm (eye-in-hand); this identification rests on inspection of that one instance. The contraction is stronger on the arm tokens (dispersion ratio at 3 cm 0.86, median pooled over both views) than on the background (0.94); realized 1.00 for both. The starting cost to a foreign goal relates weakly to the size of the response over the 56 pairs (Pearson −0.29, Spearman −0.04; the pairs share instances and goals), so the "less aligned target, larger decrease" reading of the Why section has weak support that is not rank-robust.

**Reading.** (i) Contraction of the predicted latent accounts for about two thirds of the magnitude of the goal-free term under a linear rescaling of the prediction; the remaining third keeps the same sign toward every goal, so second-moment matching per view is not the whole carrier. The intervention is on the predictor's output, not on the predictor. (ii) On this candidate set pair concordance is largely a step-length score: 93 of the 128 moving candidates end farther from the goal than the start (11–12 of 16 per instance), `‖a‖` alone reaches 60.5%, and a uniform penalty recovers 31.4 of the 36.0 points E003 gained (47.3 → 78.8 against 83.4; 87%). Concordance and selection then disagree as figures of merit: the penalty ranks pairs and picks short moves. On point estimates the foreign baseline exceeds a uniform penalty of its own slope (0.55 per metre from its 3 cm median, 0.78 from its 1 cm median; uniform penalties at those λ give 72.7% and 77.3%) and keeps selection at 7/8; the two intervals overlap and no paired interval was computed. (iii) Selection rewards long steps here (the best candidate is always the 3 cm toward move); the 8/8 against 7/8 difference rests on instance-01.

**Plan rule.** No pre-registered branch fits: prediction 1 is split (66% removed, above the 30% that defines its failure branch), prediction 2 fails, prediction 3 holds. The choice made after seeing the result, and labelled as such: treat step length as the practical handle, to be tested as a trust radius set from `rho` (P1) against a norm-penalized CEM. Two qualifications from this data: a penalty tuned for ranking over-penalizes toward moves (3/8), so the penalized baseline needs its λ set by regret; and the term is concentrated in view 1, which the release's objective weights 0.5. E005 asks, on the same arrays, whether the direction signal survives in view 0 alone and whether the raw cost already ranks candidates of equal step length.

Scope: one kitchen, 5 cm goals, one model tick, one checkpoint, the 8 instances already used by E001 and E003; exploratory until the fresh sweep (seeds 1008–1015). Reviewed once (fresh context); its corrections are applied above. Not run: a paired interval between the foreign baseline and the penalty, λ selected by regret, per-view direction-dependent gain and ranking among candidates of equal step length (both in E005), any closed-loop use of these costs, a second scene or checkpoint.
