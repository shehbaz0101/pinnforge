# Stage G — claim to evidence

`python -m pinnforge.claims check` loads [docs/claims.yaml](../claims.yaml) and the JSON files named there. The command fails when a registered number does not match its evidence at the rounding implied by the published token, when a registered comparison is false, when a superiority phrase in a scanned document has no registered claim, or when a registered claim gives the FNO inverse an unqualified ranking on `hard_ood`.

Scanned superiority phrases are `best`, `beats`, `beat`, `beating`, `outperform`, `outperforms`, `outperforming`, `most accurate`, `win`, `wins`, and `edge`. The scan covers `README.md`, `docs/README.md`, `CHANGELOG.md`, `PROJECT_STATUS.md`, the demo report header (`docs/demo/result.md` and `docs/demo/index.html`), `docs/research/strong_baselines.md`, `docs/v02/*.md`, and the stage reports at the repository root.

Every decimal, percent, or scientific-notation token in the README, the docs map, the changelog, `PROJECT_STATUS.md`, the demo report header, and this note must belong to a registered claim. Version strings, loopback addresses, and Python minor versions are ignored. Every number in `docs/research/strong_baselines.md` must equal a numeric leaf of `docs/research/strong_baselines_scores.json` or `docs/research/strong_baselines_protocol.json` when both are formatted with Python's `.16g`. Stage-report tables are not rewritten here, and this checker does not recompute them. Their superiority phrases are registered. `tests/test_published_scores.py` still locks the changelog headlines to the score files.

## What is NOT claimed

The FNO inverse is not the most accurate method on `hard_ood`. On the Stage D `sensors32_bursts` mask, mean relative viscosity error sorts as sensor-lift solver `0.0138`, hybrid FNO `0.0337`, data-only FNO `0.0495`, weak form `0.0898`, closed form `0.349`. Those readings are three significant figures of [docs/research/strong_baselines_scores.json](../research/strong_baselines_scores.json). The sensor-lift solver has lower error than both FNO arms. The oracle initial-condition solver is lower still (`0.00251`). It sees the full field at `t = 0`, so it is a ceiling rather than a sensor-only estimator.

The smooth demo is a different experiment. There the closed form wins (`0.41%` versus `4.97%` mean relative error). That result does not transfer to `hard_ood`.

Stage D shows both FNO arms below the closed-form residual least squares on `hard_ood`, with zero failures against nine for the closed form. The same arms sit above the sensor-lift solver. The closed-form comparison is not a ranking of every estimator scored later.

No width, mode count, depth, or learning rate is best outside the set frozen before the run that measured it. Choosing the epoch with the lowest validation relative L2 is a training rule, not a ranking of inverse methods.

The checker does not retrain, and it does not edit protocol files, score files, or stage-report numbers.
