# Project status

PINNForge is a sandbox for physics-informed neural networks on a few classic
ODE and PDE residuals. It trains a small network on the residual, scores a
checkpoint against an analytical or manufactured field, and serves that path
on localhost. An offline demo trains a checked-in sample on CPU.

**Status:** v0.1.0 remains the tagged freeze. Stage 1 correctness, the Stage 2 Burgers reference, the Stage 3 data-only FNO baseline, the Stage 4 residual ablations, the Stage 5 sparse viscosity recovery, the Stage A harder Burgers pilot, and the Stage B multi-seed data-only FNO on that pilot are unreleased and keep the `0.1.0` version string. See [STAGE1_REPORT.md](STAGE1_REPORT.md), [STAGE2_REPORT.md](STAGE2_REPORT.md), [STAGE3_REPORT.md](STAGE3_REPORT.md), [STAGE4_REPORT.md](STAGE4_REPORT.md), [STAGE5_REPORT.md](STAGE5_REPORT.md), [STAGE_A_REPORT.md](STAGE_A_REPORT.md), and [STAGE_B_REPORT.md](STAGE_B_REPORT.md).

## Days 1–10

| Day | Delivered |
| --- | --- |
| 1 | Package scaffold, equation schemas, harmonic closed form |
| 2 | Collocation, initial-condition, and boundary samplers |
| 3 | MLP and residual autograd |
| 4 | Adam training loop, checkpoints, and metrics |
| 5 | Evaluation against reference fields and residual stats |
| 6 | Experiment config and `pinnforge run --config` |
| 7 | FastAPI service and `pinnforge serve` |
| 8 | Path sandbox, per-client rate limit, offline socket guard |
| 9 | `pinnforge demo` on checked-in samples |
| 10 | Freeze at v0.1.0, this status note, changelog |

## Install

Python 3.11 or newer. Install a CPU build of torch first so the `ml` extra
does not pull a CUDA wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[ml,api]"
```

`pinnforge demo`, `pinnforge train`, `pinnforge eval`, `pinnforge run`,
`pinnforge residual`, and the train, eval, and run HTTP routes need the
`ml` extra. `pinnforge serve` needs the `api` extra. Schemas, sampling,
`pinnforge sample`, and `GET /health` and `GET /equations` do not need
torch. Tests use `pip install -e ".[dev,api]"`. Lint uses
`pip install -e ".[lint]"`.

## Quickstart

```bash
pinnforge demo
pinnforge serve
```

`pinnforge demo` trains `samples/configs/harmonic.yaml` on CPU (2 epochs,
a width-(8, 8) network) and prints the loss and the error against the
closed form. `--equation poisson` uses `samples/configs/poisson.json`.
`--equation burgers` uses `samples/configs/burgers.yaml` and prints residual
metrics, because Burgers has no reference field. It does not open a network
connection. `pinnforge serve` binds to `127.0.0.1:8000`.

## Tests and CI

CI runs on pull requests and on pushes to `main`. The unit-test job installs
`pip install -e ".[dev,api]"` and runs `pytest -m "not ml"` on Python 3.11
and 3.12. A separate job installs the CPU torch wheel, then the same extras,
and runs the full suite on Python 3.12. A lint job runs `ruff check .`
(rules E4, E7, E9, F, I). HTTP tests use FastAPI `TestClient` and do not
bind a port. Nothing in the suite downloads weights.

## Stage 1

Correctness and packaging on top of the v0.1.0 freeze. The version string is still `0.1.0` because the CLI and import tests pin that string. This branch does not move the `v0.1.0` tag.

- Periodic penalties match the field and the derivative along the periodic axis at both endpoints, at each boundary row's other coordinates. Neumann penalties match the outward normal derivative. A condition the penalty cannot represent raises. `w_bc = 0` still drops boundary enforcement, on purpose.
- A Poisson reference is returned only when it matches the source, the domain, and every prescribed boundary condition. Source `one` on `[0, 1]` with zero Dirichlet ends is `x(1 - x) / 2`. Anything else with no match is unavailable. The Burgers evaluation hook is still `NotImplementedError`.
- Train, validation, and test draws use separate `SeedSequence` streams. `pinnforge sample` still uses `numpy.random.default_rng(seed)`. The validation draw is hashed into `manifest.json` and is not in the loss.
- Evaluation reports held-out initial and boundary errors, plus relative and max field error beside the residual. A zero field on the default oscillator has a near-zero residual and relative L2 of 1.
- Demo configs ship in `pinnforge.data` and are read with `importlib.resources`. The repository `samples/` tree remains the checkout fallback.
- `relu` is rejected for the strong-form residual. `tanh` and `silu` stay. Checkpoints record an interval, a run manifest, Adam state, and the torch RNG, and resume when that state is present. Loads use `weights_only=True`.

## Stage 2

A Fourier reference for periodic viscous Burgers, separate from the coordinate-PINN trainer. [STAGE2_REPORT.md](STAGE2_REPORT.md) has the method, the convergence numbers, and the pilot manifest.

- Dealiased (3/2 rule) Fourier derivatives with `k_m = 2π m / L` on `L = 2`, and ETDRK4 in float64. Cole–Hopf and a second-order finite-difference scheme are the cross-checks.
- On the pilot settings (`N = 256`, `dt = 1e-3`) the convergence initial condition at `ν = 0.05` differs from an `N = 1024`, `dt = 2.5e-4` run by about `1.09e-12` in final-time relative L2. The preregistered label tolerance is `1e-8`.
- The pilot is 512 train / 128 validation / 128 test trajectories, split by problem instance before any windowing. Normalization statistics are fit on the training fields only and are not applied to the files. The arrays are about 147 MiB and are gitignored; `docs/stage2/pilot_manifest.json` records seeds, hashes, and the split.
- Viscosity in the pilot is drawn from `[0.02, 0.10]`. A probe at `ν = 0.005` missed the `1e-8` resolution gate and is not in the labels.
- No inverse-viscosity fit in this stage. The data-only Fourier neural operator is Stage 3. Stage 4 adds residual and hybrid losses on that same operator. Stage 5 recovers `ν` from sparse sensors.

## Stage 3

A data-only 1D Fourier neural operator on windows cut from the Stage 2 pilot. [STAGE3_REPORT.md](STAGE3_REPORT.md) has the commands and the measured test error.

- The train, validation, and test split is the instance split in `docs/stage2/pilot_manifest.json`. Windows are cut after that assignment. `u` uses the training mean and standard deviation recorded in the manifest. Viscosity is a spatially constant input channel, normalized by the training-split mean and population standard deviation of `ν`.
- The default window is 8 input frames and 8 target frames with stride 8 (`save_dt = 0.01`, so the lead time is 0.08). The loss is mean squared error in normalized `u`. The checkpoint is the epoch with the lowest validation mean relative L2. Test instances are not used for that choice.
- On the reported CPU run (width 32, 16 modes, 4 layers, seed 0, epoch 24) the test mean relative L2 is about `2.65e-3`. A persistence baseline on the same windows is about `6.64e-2`. That is a measured fit on this pilot, not a physics-informed model and not a match to the `1e-8` label gate.
- `pinnforge.reference.burgers.reference_solution` still raises `NotImplementedError`. `pinnforge eval` does not load these windows.

## Stage 4

Residual and hybrid losses on the Stage 3 FNO, same windows and the same validation relative L2 checkpoint rule. [STAGE4_REPORT.md](STAGE4_REPORT.md) has the stencil, the commands, and the test table.

- `--loss data` reproduces the Stage 3 metrics on the reported CPU wheel. `--loss residual` optimizes the mean square Burgers residual. `--loss hybrid` adds that term with a weight chosen from `{1e-6, 1e-4, 1e-2}` on validation only. The selected weight in the report is `1e-2`.
- The residual uses the pilot spectral derivative and a central difference at `Δt = 0.01`. `ν` is the known instance viscosity. Test mean relative L2 for the selected hybrid is about `1.98e-3`, compared with about `2.65e-3` for the data-only control. Both beat persistence (about `6.64e-2`) and both stay far above the `1e-8` label gate.
- No inverse viscosity, no new architecture, and no claim that the preregistered weight is optimal outside that set. Sparse recovery of `ν` is Stage 5.

## Stage 5

Scalar viscosity from sparse sensors on the Stage 2 pilot, using the Stage 4 Burgers residual. [STAGE5_REPORT.md](STAGE5_REPORT.md) has the mask, the command, and the test table.

- The observation pattern is fixed in code before the test score: 32 equispaced sensors and four bursts of five consecutive frames. That is 640 samples out of each `101 × 256` trajectory. The estimator is the normal equation for the Stage 4 central residual. It does not fit a new network and it does not read test viscosities.
- On the 128 held-out instances the mean absolute error in `ν` is about `6.09e-5`, against about `1.81e-2` for the training-split mean. Mean relative error is about `1.33e-3`. Correlation is about `0.999994`. No test instance is worse in absolute error than that constant baseline. The same command's stride-8 clock is coarser and is not the reported pattern.
- The version string stays `0.1.0`. This is not the `1e-8` solver gate, and it does not claim that `ν` is unique from every other sensor mask.

## Stage A

A harder Burgers pilot in parallel with Stage 2. [STAGE_A_REPORT.md](STAGE_A_REPORT.md) has the family, the convergence tables, and the manifest. Stage B is the first operator trained on it.

- Initial data is `tanh_bandlimited`: an 8-mode polynomial with amplitudes `Uniform(-1, 1) / sqrt(m)`, shaped by `tanh(3 p_hat)`, then projected onto modes `|m| <= 48`, mean removed, and max-abs normalized on 8192 nodes. Viscosity is `Uniform(0.005, 0.10)`. The Stage 2 family and the Stage 2 manifest are unchanged.
- Labels are `N = 1024`, `dt = 2.5e-4`, `save_dt = 0.01`, `t ∈ [0, 1]`, the same dealiased ETDRK4 solver. On the steepest draws at `ν = 0.005`, space-time relative L2 against `N = 2048`, `dt = 1.25e-4` is about `4e-11`. The preregistered gate is `1e-9`. The Stage 2 grid (`N = 256`, `dt = 1e-3`) misses that gate on this family.
- The split is 512 / 128 / 128 by problem instance. Arrays under `artifacts/burgers_hard_pilot/` are gitignored. `docs/stage_a/pilot_manifest.json` records seeds, field SHA-256s, viscosity statistics, and the solver-config hash. `docs/v02/pilot_protocol.json` is the same protocol block. `hard_ood` is `ν` at or below the training quartile `0.027028120493367818`.
- The Stage 5 `sensors32_bursts` estimator is not retuned. On the harder test split its mean absolute error is `0.0030159391726451274` (about 50 times the Stage 5 table) and 9 of 128 instances exceed relative error `0.5`, all inside `hard_ood`. Dense least squares on the same files does not. The record is `docs/v02/inverse_stress.json`. Stage A does not train the operator. Stage B does, with data loss only.

## Stage B

Five-seed data-only 1D FNO on the Stage A pilot. [STAGE_B_REPORT.md](STAGE_B_REPORT.md) has the protocol, the commands, and the tables. The version string stays `0.1.0`.

- The contract in `docs/v02/stage_b_train_protocol.json` was frozen before the test scores: seeds 0 through 4, the Stage 3 window and architecture, 30 epochs, normalized data MSE, no early stopping, and the Stage A `hard_ood` cut. `--protocol` rejects a run that leaves that contract. The Stage 2 pilot path still works without it.
- `fno slices` scores the full test split, `hard_ood` (`ν <= 0.027028120493367818`, 30 instances), and the complement (98). `fno aggregate` reduces one scalar per seed with the sample standard deviation (`ddof = 1`). The committed summary is `docs/v02/stage_b_scores.json`.
- Full-test mean relative L2 is `4.298970125314308e-03 ± 3.608651258270467e-04`. `hard_ood` is `7.049252763443842e-03 ± 3.032654479273341e-04`. The complement is `3.457046868744042e-03 ± 3.865637014640921e-04`. One-step persistence is about `8.46e-2`, `9.78e-2`, and `8.05e-2` on those three slices. The `hard_ood` mean is about twice the complement on every seed. Both beat persistence. Neither meets the `1e-9` label gate.
- The Stage A sensor inverse is not retrained. Its nine `0.5`-rule failures remain inside `hard_ood`. Forecasting `u` with `ν` given does not remove that stress.

## Stage C

Residual and hybrid losses on the same harder pilot, five seeds. [STAGE_C_REPORT.md](STAGE_C_REPORT.md) has the protocol, the validation weight choice, and the test tables. The version string stays `0.1.0`.

- [docs/v02/stage_c_train_protocol.json](docs/v02/stage_c_train_protocol.json) was frozen before the test scores. Seeds, width, modes, depth, window, epochs, and learning rate match Stage B. The hybrid grid is `{1e-6, 1e-4, 1e-2}`. `hard_ood` is still `ν <= 0.027028120493367818`. The data-only arm is the Stage B aggregate and was not retrained.
- The selected weight is `1e-2`, the lowest mean validation relative L2 across the five seeds. The other two weights were not scored on test.
- On `hard_ood`, hybrid mean relative L2 is `5.215427637735405e-03 ± 6.736875927560861e-04`, against the Stage B data-only `7.049252763443842e-03 ± 3.032654479273341e-04`. The hybrid number is lower on every seed. Residual-only is `5.704581248709277e-03 ± 7.767311465741109e-04`, also lower on every `hard_ood` seed, and higher than data-only on seed 1 for the full test and the complement. The `hard_ood` mean stays about twice the complement. Neither arm meets the `1e-9` label gate. No inverse model was trained.

## Known limits

- CPU-first. `TrainConfig.device` is `cpu` only. `pinnforge demo` allows at
  most 5 epochs. The checked-in samples use 2 epochs and a width-(8, 8)
  network. A longer fit stays on `pinnforge train`.
- Localhost bind. `pinnforge serve` defaults to `127.0.0.1:8000`. `0.0.0.0`
  and `::` are refused unless `--allow-remote` is set. There is no
  authentication.
- Offline-by-design. Train, eval, run, demo, and the API refuse non-loopback
  TCP connects. Loopback stays open. The package does not download weights.
- Free and public only. No API keys, no paid services, no secrets, and no
  private data.
- Burgers evaluation still has no reference field. It reports residual
  metrics and boundary penalties, not a field error. Spectral trajectories
  are a separate package and are not loaded by `pinnforge eval`. The Stage 3
  and Stage 4 FNO scores windows from those trajectories; it is not wired
  into `pinnforge eval`. Stage 5 reads the same trajectories only to
  recover `ν` from a sensor mask. It is also not wired into `pinnforge eval`.
- `w_bc = 0` does not enforce boundary conditions. An eval config with
  `n_bc = 0` or `n_ic = 0` does not score that condition.
- Resume requires a checkpoint that stored the optimizer, the torch RNG,
  and the metrics history, and that matches the equation, counts, widths,
  activation, learning rate, and seed. The spectral Burgers solver and the
  Burgers FNO use their own files. They are not part of coordinate-PINN
  checkpoint resume. The FNO checkpoint does not resume an Adam run.

## Release tag

Stage 1 does not move `v0.1.0`. The steps below are the freeze procedure from that tag.

`pyproject.toml` and `pinnforge.__version__` are `0.1.0`. The annotated tag
`v0.1.0` is created on the tip of `feat/day10-freeze`. A squash merge of
this pull request produces a new commit on `main` and leaves that tag on
the branch tip. After the squash SHA is on `main`, retarget the tag (the
existing tag has to move):

```bash
git fetch origin main
git push origin :refs/tags/v0.1.0
git tag -a v0.1.0 <squash-sha> -m "PINNForge v0.1.0"
git push origin v0.1.0
```

Do not point `v0.1.0` at any commit other than that squash SHA on `main`.
