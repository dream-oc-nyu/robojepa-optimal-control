# Fixed evaluator results

Instances: 8. Values are median [min, max] over instances; amplitude rows first take each instance's median.
Goal and neutrals are excluded from amplitude summaries. Amplitudes are unsigned norms in mm.

## Interface: executed / commanded

| Commanded mm | Instances | Candidates | Executed mm | Ratio |
|---|---|---|---|---|
| 1 | 8 | 16 | 0.0875595 [0, 0.107723] | 0.0875595 [0, 0.107723] |
| 3 | 8 | 16 | 2.15759 [2.11772, 2.23519] | 0.719195 [0.705907, 0.745062] |
| 10 | 8 | 48 | 9.26256 [9.08205, 9.37955] | 0.926256 [0.908205, 0.937955] |
| 30 | 8 | 48 | 29.2837 [29.1743, 29.5579] | 0.976124 [0.972477, 0.985262] |

## Gain: dC_pred / dC_real

| Executed bin mm | Candidates | Cmd valid instances | G cmd | Exec valid instances | G exec |
|---|---|---|---|---|---|
| [0, 1) | 16 | 6 | 0.186669 [-0.470687, 0.788895] | 6 | 0.0076442 [-0.0303975, 0.0820766] |
| [1, 3) | 16 | 8 | 0.344171 [-0.487029, 4.00374] | 8 | 0.18184 [-0.332275, 2.08462] |
| [3, 10) | 48 | 8 | 0.370224 [-0.982301, 1.98885] | 8 | 0.32692 [-0.93643, 1.75608] |
| [10, 30) | 45 | 8 | 0.0924875 [-0.96045, 0.725476] | 8 | 0.0878314 [-0.985494, 0.736242] |
| [30, inf) | 3 | 3 | -0.0704664 [-0.0710901, 0.061784] | 3 | -0.060794 [-0.0708183, 0.0619922] |

## Ranking and selection

| Set | Selector | Ordered / pairs | Ties (cost / D / both) | Fraction [bootstrap 95% CI] | Progress captured | Regret mm | Selected |
|---|---|---|---|---|---|---|---|
| all | realized | 813 / 1088 | 0 / 8 / 1 | 0.747243 [0.714154, 0.776654] | 1 [1, 1] | 0 [0, 0] | instance-00:onray-plus-0.03, instance-01:onray-plus-0.03, instance-02:onray-plus-0.03, instance-03:onray-plus-0.03, instance-04:onray-plus-0.03, instance-05:onray-plus-0.03, instance-06:onray-plus-0.03, instance-07:onray-plus-0.03 |
| all | predicted-cmd | 515 / 1088 | 0 / 9 / 0 | 0.473346 [0.454963, 0.497243] | 1 [-0.290076, 1] | 0 [0, 37.8684] | instance-00:onray-plus-0.03, instance-01:offray1-minus-0.03, instance-02:onray-plus-0.03, instance-03:onray-plus-0.03, instance-04:onray-plus-0.03, instance-05:onray-plus-0.03, instance-06:onray-plus-0.03, instance-07:onray-plus-0.03 |
| all | predicted-exec | 502 / 1088 | 0 / 0 / 9 | 0.461397 [0.443934, 0.482537] | 1 [-0.290076, 1] | 0 [0, 37.8684] | instance-00:onray-plus-0.03, instance-01:offray1-minus-0.03, instance-02:onray-plus-0.03, instance-03:onray-plus-0.03, instance-04:onray-plus-0.03, instance-05:onray-plus-0.03, instance-06:onray-plus-0.03, instance-07:onray-plus-0.03 |
| onray | realized | 203 / 224 | 0 / 2 / 1 | 0.90625 [0.875, 0.928683] | 1 [1, 1] | 0 [0, 0] | instance-00:onray-plus-0.03, instance-01:onray-plus-0.03, instance-02:onray-plus-0.03, instance-03:onray-plus-0.03, instance-04:onray-plus-0.03, instance-05:onray-plus-0.03, instance-06:onray-plus-0.03, instance-07:onray-plus-0.03 |
| onray | predicted-cmd | 134 / 224 | 0 / 3 / 0 | 0.598214 [0.558036, 0.642857] | 1 [1, 1] | 0 [0, 0] | instance-00:onray-plus-0.03, instance-01:onray-plus-0.03, instance-02:onray-plus-0.03, instance-03:onray-plus-0.03, instance-04:onray-plus-0.03, instance-05:onray-plus-0.03, instance-06:onray-plus-0.03, instance-07:onray-plus-0.03 |
| onray | predicted-exec | 131 / 224 | 0 / 0 / 3 | 0.584821 [0.540179, 0.633929] | 1 [1, 1] | 0 [0, 0] | instance-00:onray-plus-0.03, instance-01:onray-plus-0.03, instance-02:onray-plus-0.03, instance-03:onray-plus-0.03, instance-04:onray-plus-0.03, instance-05:onray-plus-0.03, instance-06:onray-plus-0.03, instance-07:onray-plus-0.03 |
| offray | realized | 144 / 224 | 0 / 0 / 0 | 0.642857 [0.584821, 0.6875] | 5.45634 [1, 17.4313] | 3.44341 [0, 7.42959] | instance-00:offray0-plus-0.01, instance-01:offray0-plus-0.01, instance-02:offray0-minus-0.01, instance-03:offray0-minus-0.03, instance-04:offray1-minus-0.03, instance-05:offray0-minus-0.03, instance-06:offray0-minus-0.01, instance-07:offray1-plus-0.03 |
| offray | predicted-cmd | 97 / 224 | 0 / 0 / 0 | 0.433036 [0.357143, 0.504464] | 13.7065 [10.0982, 17.4313] | 7.23258 [6.79863, 7.93464] | instance-00:offray1-minus-0.03, instance-01:offray1-minus-0.03, instance-02:offray1-minus-0.03, instance-03:offray0-minus-0.03, instance-04:offray0-minus-0.03, instance-05:offray0-plus-0.03, instance-06:offray1-minus-0.03, instance-07:offray0-minus-0.03 |
| offray | predicted-exec | 93 / 224 | 0 / 0 / 0 | 0.415179 [0.348214, 0.477679] | 13.7065 [10.0982, 17.4313] | 7.23258 [6.79863, 7.93464] | instance-00:offray1-minus-0.03, instance-01:offray1-minus-0.03, instance-02:offray1-minus-0.03, instance-03:offray0-minus-0.03, instance-04:offray0-minus-0.03, instance-05:offray0-plus-0.03, instance-06:offray1-minus-0.03, instance-07:offray0-minus-0.03 |

Regret split (mm): predictor+interface 0 [0, 37.8684]; representation+cost 0 [0, 0]; total 0 [0, 37.8684].

Concordance = strictly concordant / all pairs; exact ties are separate, not successes. All-set includes the first neutral; ray subsets contain only their named kind. Argmin ties use input order. Subset progress uses the first-neutral distance and that subset's best D.
Gain is null for zero dC_real or |dC_real| < 10 times neutral-repeat noise. Ratio/cosine/progress with zero denominators are null. JSON preserves null; plot tables use nan.
Bootstrap: 2,000 resamples of whole instances, seed 0, percentile 95% interval; unavailable below four instances. Bins are [0,1), [1,3), [3,10), [10,30), [30,inf) mm; commanded amplitudes group at 1e-6 mm precision.
E001 raw arrays scored as supplied; meta.json is retained, but model provenance and prediction sealing are not independently verified by this evaluator.
