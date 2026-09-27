# Stage 4 report

Physics-informed losses on the Stage 3 Burgers FNO. The network, the window, the instance split, the normalization, and the checkpoint rule are the ones in [STAGE3_REPORT.md](STAGE3_REPORT.md). The version string stays `0.1.0`. This run used Python 3.12.3, NumPy 2.4.4, and the CPU torch wheel 2.14.0+cpu. It is a single-seed comparison of three loss modes on the Stage 2 pilot. It is not an inverse problem for `ν`, and it is not a new architecture.

`pinnforge.reference.burgers.reference_solution` still raises `NotImplementedError`. `pinnforge eval` still does not load these windows.

## Task and protocol

The operator and the data contract are unchanged from Stage 3:

    (u(·, t), …, u(·, t + 7Δ), ν)  ↦  (u(·, t + 8Δ), …, u(·, t + 15Δ))

with `Δ = 0.01`, stride 8, and the 512/128/128 instance split in [docs/stage2/pilot_manifest.json](docs/stage2/pilot_manifest.json). Windows are cut only after that split. `u` uses the training mean and population standard deviation in the manifest. `ν` is a constant input channel, normalized by the training-split mean and population standard deviation of viscosity. Relative L2 is the Stage 2 discrete ratio `||prediction − target|| / ||target||` after denormalizing `u`. The checkpoint is the epoch with the lowest validation mean relative L2. Ties would keep the earliest epoch. Test instances are not in the loss and not in the selection.

| item | value |
| --- | --- |
| width | 32 |
| Fourier modes kept | 16 |
| Fourier layers | 4 |
| parameters | 138248 |
| optimizer | Adam |
| learning rate | 0.001, constant |
| batch size | 32 |
| seed | 0 |
| epochs | 30, plus epoch 0 before any step |
| device | cpu |

Counts: 5632 training windows, 1408 validation windows, 1408 test windows.

The pilot arrays under `artifacts/burgers_pilot/` are gitignored. This workspace regenerated them with the Stage 2 command. The solver-config SHA-256 was `2a0c1c3f078ca81f15bf5c33874bc1403fa6a1a30bda1f484bbc6df5d0d48928`, the same value as [STAGE2_REPORT.md](STAGE2_REPORT.md). Training and evaluation check `field_sha256`. Both completed, so the loaded fields match the committed manifest.

```bash
python -m pinnforge.reference.numerical pilot --output artifacts/burgers_pilot --train 512 --val 128 --test 128
```

## Residual

The PDE is the periodic viscous Burgers equation solved by the pilot,

    u_t + u u_x = ν u_xx,    x ∈ [-1, 1].

`ν` in the residual is the instance viscosity, denormalized with the training affine map. It is known. It is not a predicted field.

Spatial derivatives are `pinnforge.reference.numerical.solver.spectral_derivative`: `k_m = 2 π m / L` with `L = 2`, the Nyquist multiplier set to 0, and `(i k)^p` on the unnormalized FFT. That is the same multiplier the pilot solver uses. The time derivative is a second-order central difference on the saved frames, which are `Δt = 0.01` apart:

    (u_t)^n = (u^{n+1} − u^{n−1}) / (2 Δt),
    R^n = (u_t)^n + u^n (u_x)^n − ν (u_xx)^n.

The training term is the mean of `R²` over the residual nodes in the batch. The reported magnitude is the mean of `|R|` on the same nodes, evaluated on test predictions after denormalizing. A data-only run still reports that magnitude. It does not put `R` in the optimizer.

Two index sets are implemented. `T = 8` predicted frames.

- `with_input` prepends the last two input frames. Those frames are data. `R` is formed at the last input time and at predicted indexes `0 .. T−2`. The node at the last input time uses the first predicted frame as its forward neighbor, so the rollout is tied to the discrete stencil of the known history. The last predicted frame is not itself a residual node. It appears as the forward neighbor of the previous node.
- `target_interior` uses only the prediction. `R` is formed at predicted indexes `1 .. T−2`. The first and last predicted frames are not residual nodes.

The measured arms use physical `u`: the network output is denormalized before `R` is formed. A `normalized` switch is the same residual divided by the training `σ_u`,

    R / σ_u = û_t + (σ_u û + μ_u) û_x − ν û_xx,

which has the same zeros and a different mean square. It was not one of the measured runs.

The central difference on the saved labels is not zero. On the test windows, with `with_input`, the mean `|R|` of the targets is `0.00023046862854424522`. That is the truncation of this stencil on the pilot trajectories. It is the floor for the reported magnitude. The `1e-8` spectral label gate is a different comparison, between the solver and a finer integration, and it is not this floor.

## Losses

`pinnforge fno train --loss` selects the scalar Adam sees.

| mode | optimized scalar |
| --- | --- |
| `data` | mean squared error of normalized `u` |
| `residual` | mean of `R²` |
| `hybrid` | data MSE + `residual_weight` × mean of `R²` |

`--residual-weight` is required for `hybrid` and rejected otherwise. Selection stays on validation mean relative L2 for every mode. The logged `train_mse` is always the normalized data MSE, including when that term is absent from the optimizer. `train_objective` is the scalar above, as a mean over the training windows rather than a mean of per-batch means.

The hybrid weights were fixed before any test evaluation:

    1e-6,  1e-4,  1e-2

They are `PREREGISTERED_HYBRID_WEIGHTS` in `pinnforge.operator.defaults`. At epoch 0 of the data-only control the training data MSE is about `0.942` and the training residual MSE is about `79`, so `1e-2` puts the two hybrid terms on the same order at initialization, `1e-4` is data-leaning, and `1e-6` is almost data-only. The weight used for the test hybrid is the one whose selected epoch has the lowest validation mean relative L2. The other two weights were not scored on test.

| weight | selected epoch | validation mean relative L2 | validation MSE | validation mean `|R|` | fit seconds |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1e-6 | 24 | 0.002711017500317143 | 9.080479231917168e-06 | 0.028149185451980727 | 140.9257968999998 |
| 1e-4 | 28 | 0.0023110003306861635 | 7.171766761925659e-06 | 0.023783996443462005 | 123.44746618900012 |
| 1e-2 | 25 | 0.0019948163435653432 | 5.520648651251481e-06 | 0.01912392853733853 | 130.34722616999989 |

The selected weight is `1e-2`. That choice is the minimum on this validation set. It is not a search over other weights, schedules, or seeds.

After that choice, one more hybrid was trained with the same weight and `target_interior` instead of `with_input`. The scope was not chosen on test.

## Commands

Shared flags for every training run: `--epochs 30 --batch-size 32 --lr 0.001 --width 32 --modes 16 --layers 4 --input-frames 8 --output-frames 8 --stride 8 --seed 0`, pilot `artifacts/burgers_pilot`, manifest `docs/stage2/pilot_manifest.json`. Field hashes were checked. `runs/` is gitignored.

```bash
python -m pinnforge.operator train --pilot artifacts/burgers_pilot --manifest docs/stage2/pilot_manifest.json \
  --output runs/stage4_data --loss data \
  --epochs 30 --batch-size 32 --lr 0.001 --width 32 --modes 16 --layers 4 \
  --input-frames 8 --output-frames 8 --stride 8 --seed 0

python -m pinnforge.operator train --pilot artifacts/burgers_pilot --manifest docs/stage2/pilot_manifest.json \
  --output runs/stage4_residual --loss residual \
  --residual-scope with_input --residual-space physical --dt 0.01 \
  --epochs 30 --batch-size 32 --lr 0.001 --width 32 --modes 16 --layers 4 \
  --input-frames 8 --output-frames 8 --stride 8 --seed 0

python -m pinnforge.operator train --pilot artifacts/burgers_pilot --manifest docs/stage2/pilot_manifest.json \
  --output runs/stage4_hybrid_w1em6 --loss hybrid --residual-weight 1e-6 \
  --residual-scope with_input --residual-space physical --dt 0.01 \
  --epochs 30 --batch-size 32 --lr 0.001 --width 32 --modes 16 --layers 4 \
  --input-frames 8 --output-frames 8 --stride 8 --seed 0

python -m pinnforge.operator train --pilot artifacts/burgers_pilot --manifest docs/stage2/pilot_manifest.json \
  --output runs/stage4_hybrid_w1em4 --loss hybrid --residual-weight 1e-4 \
  --residual-scope with_input --residual-space physical --dt 0.01 \
  --epochs 30 --batch-size 32 --lr 0.001 --width 32 --modes 16 --layers 4 \
  --input-frames 8 --output-frames 8 --stride 8 --seed 0

python -m pinnforge.operator train --pilot artifacts/burgers_pilot --manifest docs/stage2/pilot_manifest.json \
  --output runs/stage4_hybrid_w1em2 --loss hybrid --residual-weight 1e-2 \
  --residual-scope with_input --residual-space physical --dt 0.01 \
  --epochs 30 --batch-size 32 --lr 0.001 --width 32 --modes 16 --layers 4 \
  --input-frames 8 --output-frames 8 --stride 8 --seed 0

python -m pinnforge.operator train --pilot artifacts/burgers_pilot --manifest docs/stage2/pilot_manifest.json \
  --output runs/stage4_hybrid_w1em2_interior --loss hybrid --residual-weight 1e-2 \
  --residual-scope target_interior --residual-space physical --dt 0.01 \
  --epochs 30 --batch-size 32 --lr 0.001 --width 32 --modes 16 --layers 4 \
  --input-frames 8 --output-frames 8 --stride 8 --seed 0
```

Test scoring uses the `with_input` physical residual for every arm, so the magnitude column is the same definition. The interior arm was also scored with `target_interior`, which is the stencil it trained on.

```bash
python -m pinnforge.operator eval --pilot artifacts/burgers_pilot --manifest docs/stage2/pilot_manifest.json \
  --checkpoint runs/stage4_data/checkpoint.pt --split test --batch-size 32 \
  --residual-scope with_input --residual-space physical --dt 0.01 \
  --output runs/stage4_data/eval_test.json

python -m pinnforge.operator eval --pilot artifacts/burgers_pilot --manifest docs/stage2/pilot_manifest.json \
  --checkpoint runs/stage4_residual/checkpoint.pt --split test --batch-size 32 \
  --residual-scope with_input --residual-space physical --dt 0.01 \
  --output runs/stage4_residual/eval_test.json

python -m pinnforge.operator eval --pilot artifacts/burgers_pilot --manifest docs/stage2/pilot_manifest.json \
  --checkpoint runs/stage4_hybrid_w1em2/checkpoint.pt --split test --batch-size 32 \
  --residual-scope with_input --residual-space physical --dt 0.01 \
  --output runs/stage4_hybrid_w1em2/eval_test.json

python -m pinnforge.operator eval --pilot artifacts/burgers_pilot --manifest docs/stage2/pilot_manifest.json \
  --checkpoint runs/stage4_hybrid_w1em2_interior/checkpoint.pt --split test --batch-size 32 \
  --residual-scope with_input --residual-space physical --dt 0.01 \
  --output runs/stage4_hybrid_w1em2_interior/eval_test_with_input.json
```

`pinnforge fno train` and `pinnforge fno eval` call the same commands. Fit seconds below are `time.perf_counter` around `fit_fno` only. Loading and hashing the pilot is extra.

## Data-only control

`--loss data` reproduced the Stage 3 table on this machine. Epoch 0 validation mean relative L2 is `1.0077740417464434`. The selected epoch is 24, with training MSE `7.912768548435074e-06`, validation MSE `9.083419073881463e-06`, and validation mean relative L2 `0.002767531549100545`. Epoch 30 is worse than epoch 24, as in [STAGE3_REPORT.md](STAGE3_REPORT.md). The per-epoch figures in that report match this run at the precision printed there. The test row below matches the Stage 3 test row exactly, including the persistence baseline. Fit time was `122.774681891` seconds.

## Test instances

Scored once, after validation selection, on the 128 held-out instances (1408 windows). Mean `|R|` is the `with_input` physical residual. The label floor for that definition is `0.00023046862854424522`. Persistence mean relative L2 is `0.06638760492279226` on every row.

| arm | loss | scope | selected epoch | fit seconds | normalized MSE | mean relative L2 | median relative L2 | pooled relative L2 | mean `|R|` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| data-only | data MSE | `with_input` (metric only) | 24 | 122.774681891 | 7.633101968509345e-06 | 0.002647639938017037 | 0.002366013937204172 | 0.0029849555753526983 | 0.026948399308743022 |
| residual | mean of `R²` | `with_input` | 30 | 135.53680518300007 | 5.111731194568095e-06 | 0.0021247785453398283 | 0.0018822944943963876 | 0.002442706814229172 | 0.0165123678187072 |
| hybrid, weight 1e-2 | data MSE + 0.01 mean of `R²` | `with_input` | 25 | 130.34722616999989 | 4.644715708193792e-06 | 0.0019757427324667695 | 0.0018147579555961068 | 0.0023284499567162754 | 0.018321191785774682 |
| hybrid, weight 1e-2 | same weight | `target_interior` | 29 | 138.04908070300007 | 6.5511784414552795e-06 | 0.002481454976810026 | 0.0023446157258963955 | 0.002765330568349206 | 0.016424706473547417 |

At the selected epoch the residual-only training objective, which is the training residual MSE, is `0.0006799896560897794`. The selected hybrid objective is `1.2905945055070746e-05`, from data MSE `4.850851465833613e-06` plus `0.01` times training residual MSE `0.0008055093589237134`.

On this seed, the selected hybrid has the lowest test mean relative L2 of the four rows. The residual-only arm has a lower test mean `|R|` than that hybrid and a higher test mean relative L2. The interior hybrid is closer to the data-only field error than to the `with_input` hybrid. Scored with the stencil it trained on, its test mean `|R|` is `0.01188035560884418`, and the label floor for that smaller index set is `0.00020262179900820733`.

Every learned mean `|R|` is still about 70 to 120 times the `with_input` label floor. The Stage 3 test mean relative L2 is `2.65e5` times the `1e-8` solver gate. The hybrid's `0.0019757427324667695` is about `2.0e5` times that gate. Beating persistence on these windows is the same statement Stage 3 already made for the data-only fit. The pilot is still low-mode initial data with `ν ∈ [0.02, 0.10]`.

## What is not claimed

- No new architecture. The Fourier layer is the Stage 3 1D form. The residual helper is a stencil, not a network.
- No inverse problem for `ν`. Viscosity is an input, taken from the instance.
- No statement that `1e-2`, the window, the width, the mode count, the depth, or the learning rate is best outside this preregistered set. Only the epoch, and the hybrid weight inside `{1e-6, 1e-4, 1e-2}`, were chosen on validation.
- The weights `1e-6` and `1e-4` were not evaluated on test.
- No multi-seed study. The seed is 0.
- The normalized-space residual was not a measured arm.
- The test error does not meet the Stage 2 `1e-8` label gate, and the predicted residual does not meet the central-difference floor of the labels.
- CPU only. A different BLAS or torch build can move the non-data runs. The data-only run matched the Stage 3 figures on this wheel.
- The coordinate-PINN Burgers reference is still unimplemented. `pinnforge train` is still the MLP residual trainer.

## Tests

```bash
python -m ruff check .
python -m pytest
```

Residual tests check a stationary sine wave against the hand formula ` (π/2) sin(2 π x) + ν π² sin(π x) `, a constant field, the Cole–Hopf central residual at a fine `Δt`, and the identity between the physical residual and `σ_u` times the normalized residual. A marked test trains a tiny hybrid until its training objective is below the epoch-0 value, and checks that the torch stencil matches the numpy stencil. Window and split tests are unchanged. In this workspace, after the measured runs:

```text
332 passed, 6 skipped
```

The six skips are the no-torch cases in `tests/test_ml_import.py`. Ruff reported `All checks passed!` under the CI selection (E4, E7, E9, F, I).
