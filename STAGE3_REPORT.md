# Stage 3 report

Data-only 1D Fourier neural operator on windows from the Stage 2 Burgers pilot, based on `27b54745235912451d33d7676b1778697da19894`. The version string stays `0.1.0`. This run used Python 3.12.3, NumPy 2.4.4, and the CPU torch wheel 2.14.0. It is a supervised baseline. It is not a physics-informed operator and it is not an inverse-viscosity fit.

`pinnforge.reference.burgers.reference_solution` still raises `NotImplementedError`. `pinnforge eval` and `pinnforge demo --equation burgers` still score residual metrics. They do not load these windows.

## Task

Each pilot trajectory is a periodic solution of

    u_t + u u_x = ν u_xx,    x ∈ [-1, 1],    t ∈ [0, 1],

saved at `N = 256` and `save_dt = 0.01` (101 frames). The operator learns

    (u(·, t), …, u(·, t + 7Δ), ν)  ↦  (u(·, t + 8Δ), …, u(·, t + 15Δ))

with `Δ = 0.01`. Eight input frames and eight target frames is 0.08 time units. Stride 8 means successive input windows on one trajectory do not overlap. The target block of one pair is the input block of the next pair. That reuse stays inside the same instance.

On 101 frames the starts are `0, 8, …, 80` (11 windows). The last target frame used is index 95 (`t = 0.95`). Frames 96 through 100 are shorter than one window and are dropped.

Why this window, and not a claim that it is optimal:

- The task is to map solution windows, not a single initial condition to the whole trajectory.
- The pilot initial data has Fourier modes 1 through 4 only, and `ν` is at least 0.02, so 0.08 is a short push of a smooth field. A one-frame input would also be a Markov stepper. Eight frames is a short history.
- Stride equal to the input length keeps the supervised pairs from stacking the same input frames. It is not a split rule. The split rule is the instance id.
- No other window, width, mode count, or learning rate was trained for this report. Validation chooses the epoch only.

## Split and normalization

The instance split is the one committed in [docs/stage2/pilot_manifest.json](docs/stage2/pilot_manifest.json): 512 train, 128 validation, 128 test. It is the Fisher–Yates assignment from `assign_splits(512, 128, 128, 20260926)`. Windows are cut only after that assignment. A window's frames come from one trajectory. Training reads train and validation files only. Test files are read by `eval`.

`u` uses the training statistics recorded in that manifest, not a new fit on the loaded batch:

| quantity | value |
| --- | ---: |
| training `u` mean | 1.2021494137744596e-18 |
| training `u` std (ddof 0) | 0.3469596293138632 |

The same affine map is applied to input frames and target frames, on every split. Files stay raw. The network trains in that normalized space.

Viscosity is an instance parameter, constant in space and time. It is not part of the `u` mean and standard deviation. It enters as one extra input channel, constant along `x`:

    ν̂ = (ν − μ_ν) / σ_ν

`μ_ν` and `σ_ν` are the population mean and standard deviation (`ddof = 0`) of `ν` on the training instance records in the manifest. Validation and test viscosities do not enter those two numbers.

| quantity | value |
| --- | ---: |
| training `ν` mean | 0.05970840096412111 |
| training `ν` std (ddof 0) | 0.023151551663097007 |

Absolute time is not an input. The window of fields is the state. `ν` is passed explicitly because the forward map depends on it and the later inverse problem is out of scope here.

## Model and loss

The network is a 1D Fourier neural operator in the sense of Li, Kovachki, Azizzadenesheli, Liu, Bhattacharya, Stuart, and Anandkumar (2021): a pointwise lift, spectral convolution on the lowest retained modes plus a pointwise 1x1 branch, GELU between the Fourier layers, and a pointwise projection to the eight target frames. This repository does not rerun their benchmarks. The layer is the usual one, sized so a CPU job can train it.

Spectral weights are stored as separate real and imaginary float parameters. The checkpoint loads with `weights_only=True`.

| item | value |
| --- | --- |
| width | 32 |
| Fourier modes kept | 16 |
| Fourier layers | 4 |
| input channels | 9 (8 frames + `ν`) |
| output channels | 8 |
| projection hidden width | 64 |
| parameters | 138248 |
| dtype of the weights | float32 |
| device | cpu |
| optimizer | Adam |
| learning rate | 0.001, constant |
| batch size | 32 |
| seed | 0 |
| epochs | 30, plus epoch 0 before any step |

The training loss is the mean squared error between the prediction and the target window in normalized `u`. It is not a relative loss and it is not a PDE residual.

Relative L2 is the Stage 2 discrete ratio `||prediction − target|| / ||target||` on the flattened window, after denormalizing with the manifest mean and standard deviation. Model selection minimizes the mean of that score over validation windows. Ties would keep the earliest epoch. Test instances are not in the loss and not in the selection.

A persistence baseline repeats the last input frame across the eight target frames. It is not trained. It is reported on the same test windows as the network.

## Pilot regeneration

The arrays under `artifacts/burgers_pilot/` are gitignored. This workspace regenerated them with the Stage 2 command. NumPy was 2.4.4, the same version recorded in the manifest. The solver-config SHA-256 printed by the generator was `2a0c1c3f078ca81f15bf5c33874bc1403fa6a1a30bda1f484bbc6df5d0d48928`, which matches [STAGE2_REPORT.md](STAGE2_REPORT.md).

```bash
python -m pinnforge.reference.numerical pilot --output artifacts/burgers_pilot --train 512 --val 128 --test 128
```

The command refuses to overwrite an existing directory. Training and evaluation check each loaded field against `field_sha256` in the committed manifest. Both the train/validation load and the test evaluation completed, so all 768 fields matched that manifest. The directory manifest's training `u` mean and standard deviation also matched the committed values.

## Measured run

```bash
python -m pinnforge.operator train \
  --pilot artifacts/burgers_pilot \
  --manifest docs/stage2/pilot_manifest.json \
  --output runs/stage3_fno \
  --epochs 30 --batch-size 32 --lr 0.001 \
  --width 32 --modes 16 --layers 4 \
  --input-frames 8 --output-frames 8 --stride 8 \
  --seed 0

python -m pinnforge.operator eval \
  --pilot artifacts/burgers_pilot \
  --manifest docs/stage2/pilot_manifest.json \
  --checkpoint runs/stage3_fno/checkpoint.pt \
  --split test --batch-size 32 \
  --output runs/stage3_fno/eval_test.json
```

`pinnforge fno train` and `pinnforge fno eval` call the same commands. `runs/` is gitignored. The checkpoint is not part of the commit. The numbers below are what those commands printed.

Counts: 5632 training windows (512 instances), 1408 validation windows (128 instances), 1408 test windows (128 instances).

Validation relative L2 is not monotone. The minimum is epoch 24. Epoch 30 is worse, so the selected checkpoint is not the last epoch.

| epoch | train MSE (normalized) | val MSE (normalized) | val mean relative L2 |
| ---: | ---: | ---: | ---: |
| 0 | 9.41568670e-01 | 8.96522182e-01 | 1.00777404e+00 |
| 1 | 8.80762754e-04 | 9.59537546e-04 | 2.93785513e-02 |
| 2 | 3.51106641e-04 | 3.88555510e-04 | 1.95808153e-02 |
| 3 | 2.13076723e-04 | 2.39365578e-04 | 1.46949331e-02 |
| 4 | 1.41348846e-04 | 1.60218788e-04 | 1.19638006e-02 |
| 5 | 9.27583108e-05 | 1.07567267e-04 | 9.72022398e-03 |
| 6 | 6.64799531e-05 | 7.72311112e-05 | 8.07070232e-03 |
| 7 | 4.90895492e-05 | 5.68805608e-05 | 6.89916456e-03 |
| 8 | 3.96215202e-05 | 4.42328094e-05 | 6.06432475e-03 |
| 9 | 3.35401658e-05 | 3.91488582e-05 | 5.66600756e-03 |
| 10 | 2.61635537e-05 | 3.02457232e-05 | 4.91420140e-03 |
| 11 | 3.21340734e-05 | 3.51598931e-05 | 5.78491007e-03 |
| 12 | 1.97876277e-05 | 2.31322232e-05 | 4.26255054e-03 |
| 13 | 1.75177914e-05 | 2.07580319e-05 | 4.07260832e-03 |
| 14 | 5.70981826e-05 | 5.55534953e-05 | 7.57231516e-03 |
| 15 | 1.29320048e-05 | 1.57691598e-05 | 3.43007455e-03 |
| 16 | 1.86155880e-05 | 2.00646608e-05 | 4.12591586e-03 |
| 17 | 1.90855653e-05 | 2.06700115e-05 | 4.16952068e-03 |
| 18 | 1.69484588e-05 | 1.84036848e-05 | 4.17939243e-03 |
| 19 | 1.75812948e-05 | 1.89599916e-05 | 4.42983466e-03 |
| 20 | 1.66790566e-05 | 1.87067496e-05 | 4.12193593e-03 |
| 21 | 1.13014817e-05 | 1.24811748e-05 | 3.22484302e-03 |
| 22 | 1.41764951e-05 | 1.61271725e-05 | 3.65458173e-03 |
| 23 | 3.61741644e-05 | 3.62501363e-05 | 6.37548626e-03 |
| 24 | 7.91276855e-06 | 9.08341907e-06 | 2.76753155e-03 |
| 25 | 1.61581983e-05 | 1.69469839e-05 | 3.97324103e-03 |
| 26 | 6.50065239e-05 | 6.34540471e-05 | 8.06698069e-03 |
| 27 | 1.05028845e-05 | 1.15395900e-05 | 3.12466007e-03 |
| 28 | 8.08135541e-06 | 9.24094106e-06 | 2.84027420e-03 |
| 29 | 8.76342567e-06 | 9.36960328e-06 | 2.78633325e-03 |
| 30 | 4.67671076e-05 | 4.52254878e-05 | 6.72423502e-03 |

Selected checkpoint: epoch 24. At that epoch the logged training MSE is `7.912768548435074e-06`, the validation MSE is `9.083419073881463e-06`, and the validation mean relative L2 is `0.002767531549100545`.

### Test instances

Scored once, after selection, on the 128 held-out instances (1408 windows).

| quantity | value |
| --- | ---: |
| normalized MSE | 7.633101968509345e-06 |
| mean relative L2 | 0.002647639938017037 |
| median relative L2 | 0.002366013937204172 |
| pooled relative L2 | 0.0029849555753526983 |
| persistence mean relative L2 | 0.06638760492279226 |

The test mean relative L2 is lower than persistence on these windows. An untrained network at epoch 0 has validation mean relative L2 about 1.01. The fit is real on this pilot. It is also specific to this pilot: low-mode initial data, `ν ∈ [0.02, 0.10]`, and a 16-mode Fourier layer, which is wider than the band the initial conditions use.

The Stage 2 label gate is relative L2 `1e-8` for the spectral solver against a finer integration. The test mean relative L2 above is `2.65e5` times that gate. It is not a solver error and it does not meet that gate. The gate was never a target for this network.

## What is not claimed

- No new architecture. The Fourier layer is the standard 1D form.
- No physics residual, no PINO loss, and no inverse problem for `ν`.
- No statement that the window, width, mode count, depth, or learning rate is best. They were fixed before this run. Only the epoch was chosen on validation.
- No multi-seed study. The seed is 0.
- No claim that the same error holds at lower viscosity, at higher initial modes, or on a different grid.
- The coordinate-PINN Burgers reference is still unimplemented.
- `pinnforge train` is still the MLP residual trainer. `pinnforge fno` is this operator.

## Tests

```bash
python -m ruff check .
python -m pytest -m "not ml" -q --tb=line
```

With the CPU torch wheel, `python -m pytest` is the full suite. Window tests check that an instance id cannot sit in two splits, that targets are the frames after the input, that `u` and `ν` statistics come from the training split recorded in the manifest, and that loading train and validation does not require the test files. A marked test trains a tiny network until the normalized training MSE falls below half of its epoch-0 value. The full suite in this workspace, after that code was in place:

```text
322 passed, 6 skipped
```

The six skips are the no-torch cases in `tests/test_ml_import.py`. Ruff reported `All checks passed!` under the CI selection (E4, E7, E9, F, I).
