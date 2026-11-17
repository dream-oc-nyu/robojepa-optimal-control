# Ashesh — latent-model diagnostics and reach planning

**Published 17 November 2026; experimental evidence through 14 November 2026.**

This study asks which part of a frozen world-model controller limits physical progress: representation, predictive dynamics, goal cost, action interface, search, or feedback. It contains the diagnostic experiments, planner implementations, exported reach trajectories, and analysis completed so far. It is separate from the repository's top-level four-task runner; these planners have **not** been integrated into that runner.

**Current conclusion:** action saturation and a controller guard explain important failures of native CEM on this small reach study. Execution caps improve mean progress but neither beats the fixed median-based scorer with a paired interval excluding zero. **No held-out planner improvement is established.** Gauss–Newton comparisons and adaptive step-length policies remain pending.

## Start here

- [Progress and evidence](PROGRESS.md): findings, experimental populations, completed versus pending work.
- [Reproduction and verification](REPRODUCE.md): offline scoring, CPU tests, dependencies, and known limitations.
- [Native CEM and execution-cap results](episodes/E002-planner-loop/out/results.md).
- [Original episode specifications and verdicts](episodes/): E001–E011, including failed predictions and protocol deviations. These are dated historical records, not a statement that every proposed run completed.
- [Earlier two-page diagnostic report](docs/preliminary-20261111/report.pdf) and [earlier detailed report](paper/v004/note.pdf): historical, one-context investigations preceding the reach study, not the current benchmark results.

## Layout

| Path | Contents |
|---|---|
| `bench/` | Fixed reach-v0 planner interface, rollout adapter, execution harness, scorer, and historical launch drivers |
| `planners/` | Native CEM adapter; 5 cm / 10 cm execution caps; Adam; trust-region Gauss–Newton; two box-constrained GN implementations; adaptive length rules; CPU tests |
| `episodes/` | Questions, pre-run predictions, analysis code, compact results, and figures |
| `runs/E002-planner-loop/` | 24 scored seed-0 units: eight each for native CEM, 5 cm cap, and 10 cm cap; one separate calibration unit |
| `runs/E008-frozen-arm/` | Controller replays for those three arms |
| `runs/E009-guard-off/` | Two diagnostic guard-disabled units, separate from reach-v0 scoring |
| `docs/`, `paper/` | Earlier reports, clearly superseded by the newer episode evidence where applicable |

## Scope of this publication

This is a curated snapshot of the research workspace at source commit `9f243ab37341444d03eae9954624a63410519037`, supplemented by the exported run receipts and existing E001 collection code that were not tracked there. [The publication manifest](PUBLICATION.json) records each copied file's source hash and tracking status. It is not a publication of the workspace's private Git history.

Excluded: credentials, personal instructions and conversations, cluster administration/ownership records, model checkpoints, simulator assets, the bulk E001 latent arrays, downloaded third-party papers, and redundant earlier draft bundles. The E001–E005 output tables and analysis code are included, but full recomputation of those latent-array analyses needs the separately retained raw arrays. The earlier PDFs are archival reports, not self-contained reproducibility bundles.

Historical `pod/` launch/export scripts retain their original cluster/session assumptions for provenance. They are **not portable setup scripts**; do not run them against a shared cluster without adapting and authorizing the environment. No remote experiment was launched for this publication.

The existing repository license applies; upstream RoboJEPA code and model weights are not vendored here.
