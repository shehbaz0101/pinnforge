# Stage 5 report

Scalar viscosity from sparse observations of the Stage 2 Burgers pilot. The residual is the Stage 4 central stencil. The version string stays `0.1.0`. This run used Python 3.12.3 and NumPy 2.4.4. The scored command does not import torch. A marked test uses the CPU torch wheel 2.14.0+cpu only to check that Adam on the same residual moves off a bad initialization. There is one observation pattern, one ablation of its time clock, and a dense residual reference. There is no new Fourier layer.

`pinnforge.reference.burgers.reference_solution` still raises `NotImplementedError`. `pinnforge eval` still does not load these trajectories.

## Task

Each pilot trajectory solves

    u_t + u u_x = ν u_xx,    x ∈ [-1, 1],    t ∈ [0, 1],

with one unknown scalar `ν` per instance. The stored field is `N = 256` by 101 frames at `save_dt = 0.01`. The inverse reads a preregistered subset of those samples and returns one `ν̂`. It does not read the label `ν` of the instance it scores. The 512/128/128 instance split is the one in [docs/stage2/pilot_manifest.json](docs/stage2/pilot_manifest.json). The test command opens test files only. The baseline is the training-split mean of `ν` stored in that manifest, `0.05970840096412111`, the same mean as [STAGE3_REPORT.md](STAGE3_REPORT.md). Validation and test viscosities are not in that average.

The forward map from Stages 3 and 4 still takes `ν` as a known channel. This stage does not roll that network out and does not attach a head to it. On the sensors below, the Stage 4 residual is linear in `ν`,

    R = a − ν b,    a = u_t + u u_x,    b = u_xx,

so the minimizer of the mean of `R²` is the normal equation `ν̂ = (a · b) / (b · b)`. `a` and `b` are identified by evaluating `central_burgers_residual` at probe viscosities 1 and 2. The spectral derivative, the Nyquist rule, and the central time difference are unchanged. A field with no `u_xx` energy on the sensors would be rejected. None of the 128 test instances was rejected.

That critical point is not projected onto `ν > 0`. A nonpositive estimate, or a relative error above `0.5`, is counted as a failure. The threshold is `FAILURE_RELATIVE_ERROR` and was fixed with the mask. It is a reporting rule, not an accuracy target.

## Observation pattern

`PREREGISTERED_OBSERVATION` in `pinnforge.operator.inverse` is named `sensors32_bursts`. It was the pattern in the source that produced the test table. It was not edited after that command.

| item | value |
| --- | --- |
| spatial sensors | 32 equispaced nodes, stride 8 on `N = 256` |
| sensor indexes | `0, 8, 16, …, 248` |
| Nyquist mode of that subgrid | 16, the mode count the Stage 3 Fourier layer keeps |
| time bursts | frames `[0, 5)`, `[24, 29)`, `[48, 53)`, `[72, 77)` |
| samples per instance | `32 × 20 = 640` |
| full grid | `256 × 101 = 25856` |
| fraction kept | `640 / 25856 = 5 / 202` |
| residual nodes per burst | 3 (frames of length 5) |
| `Δt` | `0.01`, the pilot `save_dt` |

Burst starts `0, 24, 48, 72` are Stage 3 window starts (`window_starts` on 101 frames with the default stride 8). Length 5 is enough for three Stage 4 residual nodes and does not reuse a frame across bursts. The sensor grid is `grid(32)`: subsampling every eighth pilot node is the periodic grid of 32 points on `[-1, 1)`.

The ablation `sensors32_stride8` keeps the 32 sensors and replaces the bursts with frames `0, 8, …, 96`. The central difference then uses `Δt = 0.08`. That series is the Stage 3 stride used as a clock. It is scored in the same command. It is not the reported pattern, and its error was not used to change the bursts.

`dense_reference` uses every saved frame and every grid point. It is the same normal equation without the mask. It is a stencil ceiling for this estimator, not a sparse observation.

No other sensor count was scored on the test split.

## Pilot regeneration

The arrays under `artifacts/burgers_pilot/` are gitignored. This workspace regenerated them with the Stage 2 command. The solver-config SHA-256 was `2a0c1c3f078ca81f15bf5c33874bc1403fa6a1a30bda1f484bbc6df5d0d48928`, the same value as [STAGE2_REPORT.md](STAGE2_REPORT.md). The test command checks `field_sha256`. It completed, so the loaded test fields match the committed manifest.

```bash
python -m pinnforge.reference.numerical pilot --output artifacts/burgers_pilot --train 512 --val 128 --test 128
```

## Command

```bash
python -m pinnforge.operator inverse \
  --pilot artifacts/burgers_pilot \
  --manifest docs/stage2/pilot_manifest.json \
  --split test \
  --output runs/stage5_inverse/eval_test.json
```

`pinnforge fno inverse` is the same command. `runs/` is gitignored. The JSON is not part of the commit. The numbers below are what that command wrote. There is no weight update: the training split is not an optimizer loop. Its only role in this command is the manifest mean used as the baseline.

## Test instances

128 held-out instances. Absolute error is `|ν̂ − ν|`. Relative error is that quantity divided by `ν`. Correlation is the Pearson correlation of `ν̂` with `ν`. The baseline predicts the training mean for every instance, so it has no correlation. `n_worse_than_baseline` counts instances whose absolute error exceeds the baseline's absolute error. Mean `|R|` and mean `R²` are averages over instances of the means on that instance's observed residual nodes. Every instance has the same node count, so those averages are the pooled means. Mean `R²` is the objective. Mean `|R|` is not.

| quantity | `sensors32_bursts` | `sensors32_stride8` | `dense_reference` | train-mean baseline |
| --- | ---: | ---: | ---: | ---: |
| mean absolute error | 6.091698430189056e-05 | 1.2233428205098741e-03 | 2.778977972493421e-05 | 1.8113569504806366e-02 |
| median absolute error | 5.4106791962559714e-05 | 1.1323268339862823e-03 | 2.702063578514588e-05 |  |
| max absolute error | 4.79561603902251e-04 | 6.6339364837716824e-03 | 9.03934399344164e-05 |  |
| mean relative error | 1.3303134076090104e-03 | 2.195097632186294e-02 | 4.9296765640965002e-04 | 3.7550362086741246e-01 |
| median relative error | 9.032389564128716e-04 | 1.9422334761223917e-02 | 4.4167183018990102e-04 |  |
| max relative error | 1.8379471956251635e-02 | 6.650116886013349e-02 | 1.2616261445804054e-03 |  |
| correlation | 0.9999942242288621 | 0.9988280882732062 | 0.9999996165126923 |  |
| failures | 0 | 0 | 0 |  |
| nonpositive `ν̂` | 0 | 0 | 0 |  |
| worse than baseline | 0 | 3 | 0 |  |
| mean `R²` at `ν̂` | 2.184366476109703e-04 | 3.0462768639357035e-03 | 4.344494244759158e-06 |  |
| mean `R²` at true `ν` | 2.2227588700083467e-04 | 3.19188975129102e-03 | 4.428816224138276e-06 |  |
| mean `|R|` at `ν̂` | 4.25034088861501e-03 | 2.262908841322621e-02 | 5.38173097897033e-04 |  |
| mean `|R|` at true `ν` | 4.150272479119723e-03 | 2.1365667808624412e-02 | 4.937298003735958e-04 |  |

On the reported pattern the mean absolute error is about 297 times smaller than the training-mean baseline (`1.8113569504806366e-02 / 6.091698430189056e-05`). It is about 2.19 times the dense residual least squares on the same instances (`6.091698430189056e-05 / 2.778977972493421e-05`). The stride-8 clock is about 20 times the primary mean absolute error. Mean `R²` at `ν̂` is below mean `R²` at the true `ν` on every row, which is what the normal equation guarantees up to roundoff. Mean `|R|` is slightly higher at `ν̂` than at the true `ν`. `|R|` was not minimized.

Three test instances have primary relative error above `0.01`. All three have `ν` near the bottom of the pilot range `[0.02, 0.10]`, where the 32-point subgrid aliases more of the nonlinear cascade. None of them crosses the `0.5` failure rule. Each is still well below the baseline absolute error.

| instance | `ν` | `ν̂` | absolute error | relative error | baseline absolute error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 212 | 0.026092240573817566 | 0.026571802177719817 | 4.79561603902251e-04 | 1.8379471956251635e-02 | 3.3616160390303546e-02 |
| 433 | 0.021062207314113984 | 0.021409251070774855 | 3.470437566608711e-04 | 1.647708388229157e-02 | 3.864619365000713e-02 |
| 0 | 0.02113302146636266 | 0.021428668037639643 | 2.9564657127698415e-04 | 1.3989791840582926e-02 | 3.857537949775845e-02 |
| 758 | 0.022481024295849583 | 0.022655310329569436 | 1.742860337198525e-04 | 7.752584198400112e-03 | 3.722737666827153e-02 |
| 496 | 0.02961965002224777 | 0.02973485673969144 | 1.1520671744367009e-04 | 3.889536755401788e-03 | 3.008875094187334e-02 |

The stride-8 ablation loses to the baseline on instances 110, 506, and 705. Their viscosities are `0.059076114017104886`, `0.05934571624050025`, and `0.06053239206201111`, all within about `0.001` of the training mean, so the constant baseline's absolute error is `6.322869470162251e-04`, `3.626847236208597e-04`, and `8.239910978899978e-04`. The stride-8 absolute errors on those three are `1.3234886430000536e-03`, `2.229328464058912e-03`, and `9.873587459144975e-04`. The primary pattern does not lose to the baseline on any test instance.

The dense mean `|R|` at the true viscosity, `4.937298003735958e-04`, is not the Stage 4 window floor `0.00023046862854424522`. The node sets differ: this reference uses every saved frame on the full grid, and Stage 4 used the `with_input` window stencil. The `1e-8` label gate is a comparison of the spectral solver with a finer integration. It is not a viscosity tolerance, and this run does not resimulate trajectories at `ν̂`.

## What is not claimed

- No new architecture. The Fourier layer from Stages 3 and 4 is not loaded. The inverse is the normal equation on the existing residual.
- No learned inverse map, and no claim that a network would do better or worse on this mask.
- No statement that 32 sensors, these four bursts, or `Δt = 0.01` is optimal outside this pilot. The stride-8 ablation was scored and was not substituted for the bursts. Other sensor counts were not scored on test.
- The normal equation is the unique minimizer of this quadratic when `‖u_xx‖` on the sensors is positive. That is not a claim that `ν` is identifiable from every other sparse mask, or outside `ν ∈ [0.02, 0.10]` and the pilot's four-mode initial data.
- No multi-seed study. The estimator is deterministic. The trajectories and the split are the Stage 2 pilot.
- No field error under the recovered viscosity. The secondary numbers are the residual on the observed nodes only.
- The test error does not meet the Stage 2 `1e-8` solver gate. That gate was never a target for `ν̂`.
- CPU only. The scored path is NumPy. A different BLAS can move the last digits of a spectral derivative. The Adam check is a unit test, not this table.
- The coordinate-PINN Burgers reference is still unimplemented. `pinnforge train` is still the MLP residual trainer.

## Tests

```bash
python -m ruff check .
python -m pytest
```

Unit tests lock the sensor indexes, reject a shared or non-uniform series, check that the normal equation is a critical point of the Stage 4 residual, recover a Cole–Hopf viscosity more accurately than a constant baseline, and check that a constant field is rejected. The command-line test scores one test file and does not require the train files to exist. A marked test runs Adam from `ν = 0.8` on a Cole–Hopf burst until the residual mean square drops and the estimate is closer to the closed form than the initialization was, and closer to the truth than a constant `0.1`. In this workspace, with the CPU torch wheel and the `api` extra:

```text
343 passed, 6 skipped
```

The six skips are the no-torch cases in `tests/test_ml_import.py`. Ruff reported `All checks passed!` under the CI selection (E4, E7, E9, F, I).
