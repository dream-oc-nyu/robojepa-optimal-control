# Progress and evidence — 17 November 2026

This is a synthesis of existing experiments, not a new experimental run. Latest completed exported analysis: E011 part A, 14 November. The numerical reach comparison below was recomputed from the included unit logs for publication.

## 1. What we have built

A fixed reach benchmark separating the planner from simulator-only evaluation state; a release-CEM adapter; logging of commanded/executed motion, predicted/realized cost, physical distance, planning time and budget overruns; a paired instance-level scorer; and experiment-specific analysis and controller replay tools.

Implemented planners include native CEM, execution caps, projected Adam, trust-region Gauss–Newton, two independent box-constrained Gauss–Newton implementations, and fixed/flip-adaptive length rules. Implementation and CPU tests do not establish performance on the real model: their evaluation states differ below.

## 2. Completed native-reach comparison

**Population:** dev instances 100–107; planner seed 0 only; five executed ticks per unit; frozen released 22M/150k predictor and V-JEPA 2.1-G encoder; FP32; planner embodiment 2 (`franka_omron`), native blocking reach controller. No model training. Seeds 1–2 of the planned three-seed comparison are missing; held-out instances 200–215 are unused in the published evidence.

Per-unit progress is `1 − final_distance / initial_distance`. The fixed headline is median progress over instances (with seed medians within each instance when multiple seeds exist). Mean progress and count of improving instances are secondary diagnostics, not replacement criteria.

| Planner / execution rule | Units | Median progress | Mean progress | Ends closer than start | Final distance < 2 cm |
|---|---:|---:|---:|---:|---:|
| Native CEM | 8 | 0.951 | 0.587 | 6/8 | 6/8 |
| CEM, per-axis 5 cm execution cap | 8 | 0.917 | 0.888 | 8/8 | 5/8 |
| CEM, per-axis 10 cm execution cap | 8 | 0.957 | 0.841 | 8/8 | 7/8 |

The release's stricter final success indicator is **0/8 for all three arms**; a 2 cm diagnostic is not release success. Median native planning time is 211.879 seconds per tick on the recorded hardware. Planning time excludes observation encoding and controller execution. Native CEM overruns the measured allowance on some ticks; this is not yet a matched-time optimizer comparison.

Paired bootstrap differences of medians versus native CEM (95% intervals, resampling eight instances):

- 5 cm cap: −0.034 [−0.139, 1.262].
- 10 cm cap: +0.006 [−0.030, 0.850].

**Neither is a new incumbent under the fixed criterion.** The small sample, one planner seed, and wide intervals limit generalization. Same-seed native reruns can differ on saturated components; small final-distance differences require replication.

Evidence: [`unit.json` receipts](runs/E002-planner-loop/), [fixed scorer](bench/score.py), [full scorer output](episodes/E002-planner-loop/out/results.json), [E006](episodes/E006-action-interface/episode.md), [E007](episodes/E007-cap-length/episode.md).

## 3. What the diagnostics explain

### Action representation and execution are different maps

For planner embodiment 2, the model sees a per-axis action proportional to `clamp(a / 0.05 m, −1, 1)`, while the blocking controller executes the full metric command. Beyond 5 cm on an axis, predicted cost cannot distinguish command lengths along that saturated component. Native CEM can therefore choose a useful sign but an excessive length.

The cap interventions expose a trade-off: short steps avoid the two baseline failures but can consume the five-tick allowance on farther starts. A 10 cm cap preserves more of the far-start progress, yet one near-start trajectory oscillates. This is a state-dependent behavior; no fixed cap is demonstrated universally safe.

### A controller guard explains the frozen trajectories, but not all overshoot

[E008](episodes/E008-frozen-arm/episode.md) replayed all 40 native ticks, reproducing their motion. On the five unmoved ticks, the release wrapper's joint-margin guard zeroed position commands throughout all 200 substeps of each tick. The 5 cm and 10 cm capped replays did not activate that guard.

[E009](episodes/E009-guard-off/episode.md) disabled the guard for the two failing baseline instances, with one seed-0 draw each. Instance 107 recovered to 1.57 cm; instance 100 still ended at 10.39 cm while repeatedly overshooting on a saturated axis. These two units are a diagnostic intervention, **not valid additions to the unchanged reach-v0 leaderboard**. Replication is still needed.

### Predicted latent cost has a goal-independent motion term

[E001](episodes/E001-amplitude-sweep/episode.md) and [E003](episodes/E003-goal-free-term/episode.md)–[E005](episodes/E005-view-and-shell/episode.md) examine eight amplitude-sweep instances, with 96 moves of at least 1 cm and 672 foreign-goal pairs. Predicted cost fell for 94/96 moves, including physically unhelpful ones. The foreign-goal analysis found a decrease on 672/672 pairs; adjusting that term improved ranking on the same candidate set. Dispersion and per-camera analyses further separated response components.

These are exploratory reanalyses of the **same data**, not independent confirmations. They use **embodiment 3**, unlike the embodiment-2 reach study above. Their response gains and signs must not be transferred between embodiments without measurement. The raw latent arrays are not included in Git; derived tables, figures and analysis code are included.

### Sign flips can guide length adaptation, with important counterexamples

[E011 part A](episodes/E011-bisect/episode.md) analyzes 26 exported units (24 reach units plus two guard-off units): 130 ticks and 312 axis-transitions. Seven instances have a uniquely recovered goal offset; instance 104 is excluded from the geometric flip analysis. Of 28 saturated sign flips, 26 bracket a true per-axis overshoot. The flips are correlated: 19 come from instances 100 and 107, and four guard-off flips repeat native first-tick evidence. Removing those duplicates gives 22/24.

This motivates adaptive length rules, but does not validate them. The census contains wrong-sign counterexamples, and halving only after flips can still overshoot when the sign stays unchanged. Both flip-only and post-flip saturated-tick halving rules are implemented; their model-based closed-loop comparison has not been run in the exported evidence.

## 4. Completed versus pending

| Work | Status in this publication |
|---|---|
| Initial one-context sensitivity / interface investigation | Historical PDFs included; later centimeter-scale and embodiment-specific results qualify the early interpretation |
| E001 amplitude sweep | Completed; derived outputs included; bulk latent arrays excluded |
| E003–E005 objective, contraction and view analyses | Completed exploratory reanalysis of E001; not held-out validation |
| E002 native CEM baseline | Eight seed-0 units exported and scored; remaining planner seeds pending |
| E006 / E007 5 cm and 10 cm caps | Eight units each exported and scored; neither meets incumbent criterion |
| E008 controller replays | Completed for native CEM and both capped arms; replay receipts included |
| E009 guard-off control | Two diagnostic units exported; repeated-seed control pending |
| E010 box-constrained Gauss–Newton | Code and CPU tests included. Source notes report remote execution, but complete exported units and a scored verdict are absent here; no performance claim |
| E011 part A sign-flip census | Completed from exported logs |
| E011 part B adaptive lengths | Implemented and CPU-tested; no real-model evaluation units published |
| Adam versus trust-region Gauss–Newton | Implemented; toy tests are not a real-model comparison; wall-time contract test currently fails locally (see verification) |
| Held-out evaluation, matched-budget CEM, second predictor size, other tasks | Not established by this study |

## 5. Next scientific checks

1. Recover complete E010 trajectories and search logs, score both GN implementations, and compare with CEM at matched rollout/time budgets.
2. Replicate the native and capped baselines across planner seeds; repeat the guard-off control.
3. Test fixed 10 cm, flip-halving, and post-flip halving on the same underlying search.
4. Confirm an actual development gain on held-out instances before extending claims to another checkpoint size or task.

These are pending research directions, not completed results or jobs launched by this publication.
