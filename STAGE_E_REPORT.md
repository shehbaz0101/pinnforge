# Stage E report

Sparse and noisy observation stress for the viscosity inverse on the Stage A harder Burgers pilot. The question was frozen in [docs/v02/stage_e_stress_protocol.json](docs/v02/stage_e_stress_protocol.json) before any test score on this grid. The version string stays `0.1.0`. `hard_ood` was not refit. `λ` stays 0. The Stage A–D tables were not edited.

On the Stage D mask (`32` sensors, `4` bursts, clean field), both operator arms still recover `ν` on `hard_ood` at about `3.5%` and `5.1%` mean relative error with `0` failures, against `0.349` and `9` failures for closed-form residual least squares. That advantage holds through `1%` of field standard deviation. It breaks at `2%` noise, where both arms cross `10%` mean relative error and start failing the `0.5` rule, and it breaks at `16` equispaced sensors on the clean field. At `5%` noise the closed form is the better `hard_ood` estimator. Dropping from `4` bursts to `2` does not move either operator arm. One burst has no operator target.

## Contract

[docs/v02/stage_e_stress_protocol.json](docs/v02/stage_e_stress_protocol.json), sha256 `f53f6d3d98f49335a6e64108048f74d7353b8df22543995851a07d6e428fed19`. `frozen_before_test_evaluation` is true. The file holds no scores.

- Noise is additive i.i.d. Gaussian. The scale is the fraction times the population standard deviation (`ddof = 0`) of that instance's clean saved field, over all times and all spatial nodes. Fraction `0` adds nothing. One epsilon field is drawn per `(noise_seed, instance_id)` from PCG64 with seed sequence `[20260930, noise_seed, instance_id]`. Every mask and every positive fraction uses that same field. Fractions: `0`, `0.1%`, `0.5%`, `1%`, `2%`, `5%`.
- Sensors: `32`, `16`, `8`, equispaced, stride `1024 / n_sensors`. Bursts of length `5` at `dt = 0.01`. Starts: `4 → (0, 24, 48, 72)`, `2 → (0, 48)`, `1 → (0,)`. The clean `32` / `4` cell is the Stage 5 `sensors32_bursts` mask.
- The grid is the `14` conditions in the protocol, not a full factorial. Interaction cells are scored and are not used to place an onset.
- Failure rule, unchanged: `ν̂ <= 0` or relative error `> 0.5`. Slices: full test (`128`), `hard_ood` (`ν <= 0.027028120493367818`, `30` instances), its complement (`98`), and `ν < 0.02` (`20`).
- Operator noise seed is `0` only. Closed form also repeats seeds `1` and `2` on the noise axis. Breakdowns use seed `0`, the seed every method shares.
- Operator aggregation is the mean and sample standard deviation across seeds `0–4`, `ddof = 1`, no dropped seeds. An onset is the first cell on the ordered axis whose across-seed mean relative error is `> 0.10`, or whose across-seed mean failure count is `> 0`. If the reference cell already meets the predicate, the onset is that cell.

## Estimators

Closed-form residual least squares is the Stage 4 normal equation on the condition mask. The operator is the Stage D sparse rollout: sensor mean square, `λ = 0`, leftmost minimizer on `linspace(0.005, 0.1, 191)`. The training-split mean viscosity does not read the test field. Dense least squares is the clean full field, scored once, and is the horizontal ceiling on the curves. It is not recomputed under noise or under a sparse mask.

The `191`-point argmin is located by a golden-section search. A stride-`16` subsample with more than one local minimum is replaced by the full grid for that instance. Seed `0` of each arm was checked against the full grid on the clean reference cell; the search matched. Separately, data-only seed `0` matched the full-grid argmin on all `128` test instances at clean `32` sensors, at `5%` noise with `32` sensors, and at clean `8` sensors. Those three curves each had a single valley except one `8`-sensor instance, and the golden argmin still matched there. Other cells use the same rule and were not each recomputed on the full grid. A one-burst mask has no target burst, so the operator is recorded as inapplicable rather than as a fabricated `ν̂`.

Both arms were fit again for this stage, on CPU torch `2.14.1+cpu`. They are not the Stage D checkpoint bytes. Stage D used `2.14.0+cpu`. Data-only selected epochs match Stage B (`28, 28, 29, 26, 25`). Validation relative L2 is close on four seeds. Seed `1` is `4.61313463e-03` against the published `3.897175648928502e-03`. Hybrid selected epochs are `28, 27, 28, 30, 30`, against the Stage C weight-selection epochs `29, 30, 28, 22, 30`. The clean-mask inverse below is the number this retrain produced. The Stage D test table was not edited.

| arm | seed | epoch | this validation relative L2 | published |
| --- | ---: | ---: | ---: | ---: |
| data-only | 0 | 28 | `3.97835027e-03` | `3.98207875714179e-03` |
| data-only | 1 | 28 | `4.61313463e-03` | `3.897175648928502e-03` |
| data-only | 2 | 29 | `4.94498874e-03` | `4.8875304865287515e-03` |
| data-only | 3 | 26 | `4.29440050e-03` | `4.319636697915502e-03` |
| data-only | 4 | 25 | `4.27476092e-03` | `4.274131299715281e-03` |
| hybrid `1e-2` | 0 | 28 | `3.40973463e-03` | `3.126559837192686e-03` |
| hybrid `1e-2` | 1 | 27 | `3.24222430e-03` | `2.77956952462546e-03` |
| hybrid `1e-2` | 2 | 28 | `3.54288487e-03` | `3.625878423223408e-03` |
| hybrid `1e-2` | 3 | 30 | `3.44815282e-03` | `3.551079761562272e-03` |
| hybrid `1e-2` | 4 | 30 | `3.33555038e-03` | `3.0552787063192867e-03` |

Published data-only values are the Stage B validation column. Published hybrid values are `val_relative_l2` in [docs/v02/stage_c_weight_selection.json](docs/v02/stage_c_weight_selection.json) for weight `0.01`.

## Reference cell

Clean field, `32` sensors, `4` bursts. Closed-form `hard_ood` mean relative error is `0.3494591802034973` with `9` failures, the Stage A record. The assembler refuses the file if that cell moves.

| estimator | `hard_ood` mean rel | `hard_ood` failures |
| --- | ---: | ---: |
| closed-form LS | `0.3494591802034973` | `9` |
| data-only, this retrain | `0.05103685408807077 ± 0.015490216575987417` | `0` |
| hybrid `1e-2`, this retrain | `0.035184584545685724 ± 0.004937207443766122` | `0` |
| Stage D data-only, published | `0.049487208469027544 ± 0.0144232271670265` | `0` |
| Stage D hybrid, published | `0.03368266375533967 ± 0.0034734247381767768` | `0` |
| dense LS, clean field | `0.0013208386837906374` | `0` |
| training mean | `2.98108138535235` | `30` |

The retrain sits inside the Stage D seed spread. Both arms still have no `0.5`-rule failure on any slice of the clean reference mask. Dense LS remains about `27` times tighter than the hybrid mean on this slice.

## Noise

Held fixed: `32` sensors, `4` bursts, noise seed `0`. Operator cells are mean ± sample standard deviation across five seeds. Failure counts for an operator are that same mean. Curves: [mean relative error](docs/v02/stage_e/mean_rel_error_vs_noise_hard_ood.svg), [failure count](docs/v02/stage_e/n_failures_vs_noise_hard_ood.svg). The other slices are `mean_rel_error_vs_noise_*.svg` and `n_failures_vs_noise_*.svg` in [docs/v02/stage_e](docs/v02/stage_e).

### `hard_ood` (30)

| noise | closed form | data-only | hybrid `1e-2` |
| --- | ---: | ---: | ---: |
| 0 | `0.3495` / 9 | `0.0510 ± 0.0155` / `0` | `0.0352 ± 0.0049` / `0` |
| 0.1% | `0.3497` / 9 | `0.0507 ± 0.0153` / `0` | `0.0357 ± 0.0078` / `0` |
| 0.5% | `0.3499` / 9 | `0.0614 ± 0.0177` / `0` | `0.0449 ± 0.0109` / `0` |
| 1% | `0.3488` / 8 | `0.0876 ± 0.0216` / `0` | `0.0692 ± 0.0230` / `0` |
| 2% | `0.3424` / 8 | `0.1788 ± 0.0609` / `1.8 ± 1.6` | `0.1832 ± 0.0446` / `2.4 ± 1.1` |
| 5% | `0.2990` / 8 | `0.7504 ± 0.1975` / `15.4 ± 4.0` | `0.8354 ± 0.1429` / `17.8 ± 2.6` |

Each cell is mean relative error / failure count. The `10%` line and the first failure both arrive at `2%` for both arms. At `1%` the five-seed mean is still under `10%` with zero failures. Data-only seed `2` alone is already `0.1157` at `1%`, and hybrid seed `0` alone is `0.1039`. The onset rule uses the five-seed mean, so those single seeds do not move it. At `5%`, closed form (`0.299`, `8` failures) is ahead of both arms.

Per seed at `2%`, data-only `hard_ood` failures are `4, 1, 3, 1, 0` and mean relative errors are `0.229, 0.171, 0.253, 0.113, 0.129`. Seed `4` has no `0.5`-rule failure and is still over `10%`. Hybrid failures are `4, 1, 3, 2, 2`.

### Complement (98)

| noise | closed form | data-only | hybrid `1e-2` |
| --- | ---: | ---: | ---: |
| 0 | `0.0460` / 0 | `0.0160 ± 0.0087` / `0` | `0.0125 ± 0.0035` / `0` |
| 0.1% | `0.0459` / 0 | `0.0161 ± 0.0086` / `0` | `0.0123 ± 0.0036` / `0` |
| 0.5% | `0.0443` / 0 | `0.0185 ± 0.0076` / `0` | `0.0151 ± 0.0027` / `0` |
| 1% | `0.0402` / 0 | `0.0246 ± 0.0072` / `0` | `0.0218 ± 0.0043` / `0` |
| 2% | `0.0333` / 0 | `0.0441 ± 0.0094` / `0` | `0.0418 ± 0.0082` / `0` |
| 5% | `0.0978` / 0 | `0.1327 ± 0.0288` / `1.8 ± 2.2` | `0.1356 ± 0.0275` / `2.6 ± 3.1` |

Closed form does not cross `10%` and does not fail on this slice through `5%` on seed `0`. Seed `1` at `5%` is `0.1008` with `0` failures, so one extra draw sits just over the line. Seeds `1` and `2` are a draw check, not the onset. Both operator arms cross `10%` and record their first complement failures at `5%`.

### Full test (128) and `ν < 0.02` (20)

Closed-form full-test mean relative error starts at `0.1171` with the same `9` failures, so both onsets are the reference cell. It stays near that level through `2%` (`0.1057` / `8`) and rises to `0.1450` / `8` at `5%`.

Data-only full-test relative-error onset is `5%` (`0.2775 ± 0.0682`). The first failures are at `2%` (`1.8 ± 1.6`), while the mean relative error there is still `0.0757`. Hybrid matches that pattern: failures at `2%` (`2.4 ± 1.1`, mean relative error `0.0749`), relative-error onset at `5%` (`0.2996 ± 0.0537`).

On `ν < 0.02`, closed form starts at `0.4697` / `9` and stays there. Data-only crosses `10%` at `1%` (`0.1035 ± 0.0216`) with `0` failures, and the failures arrive at `2%` (`1.8`). Hybrid crosses both at `2%` (`0.2289 ± 0.0526`, `2.4` failures). At `1%` the hybrid mean on this slice is `0.0857 ± 0.0270` with `0` failures.

## Sensors

Held fixed: fraction `0`, `4` bursts, seed `0`. Curves: [mean relative error](docs/v02/stage_e/mean_rel_error_vs_sensors_hard_ood.svg), [failure count](docs/v02/stage_e/n_failures_vs_sensors_hard_ood.svg).

### `hard_ood`

| sensors | closed form | data-only | hybrid `1e-2` |
| --- | ---: | ---: | ---: |
| 32 | `0.3495` / 9 | `0.0510 ± 0.0155` / `0` | `0.0352 ± 0.0049` / `0` |
| 16 | `0.6533` / 16 | `0.1754 ± 0.0182` / `1.0 ± 0.0` | `0.1612 ± 0.0084` / `1.0 ± 0.0` |
| 8 | `2.3601` / 30 | `2.0687 ± 0.0600` / `20.4 ± 1.5` | `2.0297 ± 0.0181` / `18.6 ± 1.1` |

Both arms cross `10%` and record their first failure at `16` sensors. The failure count `1.0 ± 0.0` is the same instance on every seed of both arms: instance `179`, `ν = 0.00659586`. At `8` sensors every `hard_ood` instance fails the closed form (`30/30`). The operators fail about two thirds of the slice and the mean relative error is about `2`.

### Complement

| sensors | closed form | data-only | hybrid `1e-2` |
| --- | ---: | ---: | ---: |
| 32 | `0.0460` / 0 | `0.0160 ± 0.0087` / `0` | `0.0125 ± 0.0035` / `0` |
| 16 | `0.1467` / 3 | `0.0190 ± 0.0061` / `0` | `0.0148 ± 0.0030` / `0` |
| 8 | `0.9716` / 72 | `0.2064 ± 0.0141` / `12.2 ± 2.2` | `0.1838 ± 0.0151` / `9.8 ± 2.2` |

Closed form crosses `10%` and fails at `16` sensors. Both operator arms are still under `2%` with `0` failures at `16` sensors. Their complement onset is `8` sensors.

Full-test closed form is already over `10%` at `32` sensors (`0.1171` / `9`) and goes to `0.2655` / `19` at `16`, then `1.2970` / `102` at `8`. Operator full-test failures start at `16` sensors (`1.0`, mean relative error still `0.056` and `0.049`). The relative-error onset is `8` sensors (`0.643` and `0.616`). On `ν < 0.02` both operator onsets are `16` sensors (`0.238` and `0.224`, one failure).

## Bursts

Held fixed: fraction `0`, `32` sensors, seed `0`. Curves: [mean relative error](docs/v02/stage_e/mean_rel_error_vs_bursts_hard_ood.svg), [failure count](docs/v02/stage_e/n_failures_vs_bursts_hard_ood.svg).

| bursts | slice | closed form | data-only | hybrid `1e-2` |
| --- | --- | ---: | ---: | ---: |
| 4 | `hard_ood` | `0.3495` / 9 | `0.0510 ± 0.0155` / `0` | `0.0352 ± 0.0049` / `0` |
| 2 | `hard_ood` | `0.3174` / 9 | `0.0510 ± 0.0101` / `0` | `0.0402 ± 0.0086` / `0` |
| 1 | `hard_ood` | `0.2964` / 9 | inapplicable | inapplicable |
| 4 | complement | `0.0460` / 0 | `0.0160 ± 0.0087` / `0` | `0.0125 ± 0.0035` / `0` |
| 2 | complement | `0.0470` / 0 | `0.0168 ± 0.0085` / `0` | `0.0129 ± 0.0038` / `0` |
| 1 | complement | `0.0474` / 0 | inapplicable | inapplicable |

Neither operator arm crosses `10%` or fails on the way from `4` bursts to `2`, on any slice. The one-burst cell is the inapplicable onset, not a `0.5`-rule failure. Closed form keeps the same `9` failures. Its `hard_ood` mean relative error falls from `0.3495` to `0.3174` to `0.2964`. The complement stays under `5%` with `0` failures at one burst.

## Interaction cells

These four cells are not on the onset axes.

| condition | closed form `hard_ood` | data-only | hybrid `1e-2` |
| --- | ---: | ---: | ---: |
| 1%, 16 sensors, 4 bursts | `0.6495` / 15 | `0.2094 ± 0.0221` / `2.0 ± 0.7` | `0.1998 ± 0.0180` / `1.6 ± 0.9` |
| 1%, 8 sensors, 4 bursts | `2.3823` / 30 | `2.0806 ± 0.0615` / `19.8 ± 0.8` | `2.0324 ± 0.0187` / `19.0 ± 1.2` |
| 5%, 8 sensors, 4 bursts | `2.4588` / 27 | `2.6648 ± 0.1694` / `27.4 ± 1.1` | `2.8846 ± 0.1285` / `28.2 ± 0.8` |
| 1%, 32 sensors, 2 bursts | `0.3202` / 9 | `0.0828 ± 0.0140` / `0` | `0.0829 ± 0.0104` / `0` |

`1%` noise with `2` bursts is still under `10%` with `0` operator failures. `1%` noise with `16` sensors is worse than either axis alone (`2` failures, about `20%`). At `5%` noise and `8` sensors the closed form has the lower `hard_ood` mean relative error.

## Where each method breaks

Onsets are the first cell on that axis. "Already" means the reference cell (`0%`, `32` sensors, `4` bursts). "Not crossed" means no scored cell on that axis met the predicate. The operator one-burst cell is listed only as inapplicable.

| method | slice | noise, rel `> 10%` | noise, first failure | sensors, rel `> 10%` | sensors, first failure | bursts |
| --- | --- | --- | --- | --- | --- | --- |
| closed form | `hard_ood` | already (`0.349`) | already (`9`) | already | already | already; still `9` at `1` burst |
| data-only | `hard_ood` | `2%` (`0.179`) | `2%` (`1.8`) | `16` (`0.175`) | `16` (`1`) | not crossed at `2`; `1` burst undefined |
| hybrid | `hard_ood` | `2%` (`0.183`) | `2%` (`2.4`) | `16` (`0.161`) | `16` (`1`) | not crossed at `2`; `1` burst undefined |
| closed form | complement | not crossed through `5%` (`0.0978`) | not crossed (`0`) | `16` (`0.147`) | `16` (`3`) | not crossed at `1` burst |
| data-only | complement | `5%` (`0.133`) | `5%` (`1.8`) | `8` (`0.206`) | `8` (`12.2`) | not crossed at `2`; `1` burst undefined |
| hybrid | complement | `5%` (`0.136`) | `5%` (`2.6`) | `8` (`0.184`) | `8` (`9.8`) | not crossed at `2`; `1` burst undefined |
| closed form | full test | already (`0.117`) | already (`9`) | already | already | already |
| data-only | full test | `5%` (`0.277`) | `2%` (`1.8`) | `8` (`0.643`) | `16` (`1`) | not crossed at `2` |
| hybrid | full test | `5%` (`0.300`) | `2%` (`2.4`) | `8` (`0.616`) | `16` (`1`) | not crossed at `2` |
| closed form | `ν < 0.02` | already (`0.470`) | already (`9`) | already | already | already |
| data-only | `ν < 0.02` | `1%` (`0.103`) | `2%` (`1.8`) | `16` (`0.238`) | `16` (`1`) | not crossed at `2` |
| hybrid | `ν < 0.02` | `2%` (`0.229`) | `2%` (`2.4`) | `16` (`0.224`) | `16` (`1`) | not crossed at `2` |

The training mean is past both predicates on every slice at the reference cell (`hard_ood` `2.981`, `30` failures). It does not move along the axes. Dense LS is not on these axes. Its clean `hard_ood` mean relative error is `0.0013208386837906374` with `0` failures.

## Failure examples

Operator example rows mark an instance if any seed failed. The failure counts in the tables above are the mean of the per-seed counts, which is the onset rule.

At `2%` noise, `32` sensors, `4` bursts, four data-only instances have at least one failing seed. All four are in `ν < 0.02`. The hats sit above the truth on every seed except data-only seed `3` on instance `530`.

| instance | `ν` | data-only mean rel | data-only `ν̂` by seed | hybrid mean rel | closed form, seed 0 |
| ---: | ---: | ---: | --- | ---: | ---: |
| 185 | `0.006878` | `0.527` | `0.0115, 0.0100, 0.0110, 0.0105, 0.0095` | `0.614` | `0.757`, failure |
| 530 | `0.007821` | `0.496` | `0.0120, 0.0135, 0.0160, 0.0080, 0.0090` | `0.765` | `0.934`, failure |
| 83 | `0.011835` | `0.445` | `0.0185, 0.0165, 0.0185, 0.0160, 0.0160` | `0.487` | `0.464`, not a failure |
| 338 | `0.007689` | `0.379` | `0.0125, 0.0105, 0.0110, 0.0100, 0.0090` | no seed failed | `0.880`, failure |

Instance `83` is the one new operator failure the closed form does not share at this noise level: closed-form relative error `0.464` stays under `0.5`, and every operator seed estimates `0.016` to `0.0185`.

At clean `16` sensors the only operator failure, on every seed of both arms, is instance `179`, `ν = 0.00659586`. Data-only hats are `0.0185, 0.0195, 0.0205, 0.0185, 0.0180` (mean relative error `1.881`). Hybrid hats are `0.0190, 0.0185, 0.0185, 0.0185, 0.0180` (mean relative error `1.805`). Closed form on the same mask estimates `0.0164`, relative error `1.483`, also a failure. On the clean `32`-sensor mask this instance is not a closed-form failure (`0.498`, just under the rule) and neither operator arm fails it.

No operator estimate in these tables is nonpositive. The grid floor is `0.005`, so a nonpositive `ν̂` is outside the search. At `5%` noise, `hard_ood` grid-floor and grid-ceiling counts are `0` for both arms. The failures are interior overestimates. On data-only seed `0` at `5%`, `29` of `30` `hard_ood` estimates sit above the truth, and the median ratio `ν̂ / ν` is `1.81`. At clean `8` sensors the mean grid-ceiling count on `hard_ood` is `1.8` (data-only) and `2.0` (hybrid): a few estimates sit on `ν = 0.1`. Closed form at `8` sensors has `5` nonpositive estimates and `102` failures out of `128`.

## Why

Closed form on this pilot is limited by aliasing of `u_xx` on the coarse sensor grid. Stage A and Stage D already placed the `9` clean `32`-sensor failures inside `hard_ood` and inside `ν < 0.02`. Noise up to `5%` of the field standard deviation does not add a complement failure on seed `0`, and the `hard_ood` mean relative error moves from `0.349` down to `0.299`. The alias error is larger than that noise. Cutting the sensor count makes the stencil worse: complement failures appear at `16`, and `8` sensors fails `102` of `128` instances. Cutting bursts does not. Two bursts and one burst keep the same `9` failures, and the `hard_ood` mean relative error falls. The normal equation loses rows, and the rows it keeps still see the same spatial stencil.

The operator was trained on clean windows. A noisy opening burst, lifted onto the `1024` grid, feeds the rollout a field the network did not see in training. On the small viscosities the sensor minimum moves upward: a larger `ν` damps the noise that the lift carried in, and the later bursts then match a smoother prediction. At `2%` the four data-only problem instances are all estimated high. The sensor curve also flattens. On `hard_ood`, the median of `(max − min) / min` of the evaluated sensor curve is `348` for data-only and `575` for hybrid on the clean reference, `19` and `17` at `2%`, and `3.3` and `2.9` at `5%`. At `8` sensors that median is about `1.5`. The `191`-point argmin is then a shallow valley, and relative error on `ν ≈ 0.007` crosses `0.5` as soon as `ν̂` moves by a few thousandths. The complement, where `ν` is larger, absorbs the same absolute shift and stays under `10%` until `5%` noise or `8` sensors.

Hybrid `1e-2` is the better clean-mask arm (`0.035` against `0.051`) and it does not move the `hard_ood` onsets. At `5%` noise and at `5%` with `8` sensors its `hard_ood` mean is higher than data-only. The residual weight was not retuned for noise. `λ` stays `0`, which was the Stage D validation choice on the clean mask.

At `5%` noise the closed form is ahead on `hard_ood` (`0.299` / `8` against `0.750` / `15.4` and `0.835` / `17.8`). The stencil does not try to explain the noise as viscosity. The learned rollout does.

## What is not claimed

- The `hard_ood` cut, the Stage 5 mask, and the Stage A, B, C, and D protocols were not retuned. No previously reported number was edited.
- `λ` was not re-selected under noise or sparsity. No new regularizer and no learned inverse were added.
- Noise seeds `1` and `2` were not scored for either operator arm. The one-burst mask has no operator estimate.
- Dense least squares was not shown noisy or sparse observations. The training mean does not read the test field.
- These checkpoints are a retrain on torch `2.14.1+cpu`, not the Stage D weight files. The clean reference cell is close to the published Stage D inverse and is not a claim of bitwise reproduction.
- The golden search was checked against the full grid on the curves named above. It was not repeated for every seed and every condition.
- Beating the closed form at `0` to `1%` noise is not the dense ceiling, and it does not survive `2%` noise or `16` sensors on `hard_ood`.
- A single seed can cross `10%` before the five-seed mean does. The onset in the table is the mean.

## Files

- Protocol: [docs/v02/stage_e_stress_protocol.json](docs/v02/stage_e_stress_protocol.json)
- Scores: [docs/v02/stage_e_scores.json](docs/v02/stage_e_scores.json)
- Curves: [docs/v02/stage_e](docs/v02/stage_e)
- Scoring: `pinnforge.operator.stage_e` and `pinnforge.operator.stage_e_fit`, command `python -m pinnforge.operator stage-e-stress`
