# Changelog

## Unreleased

Stage E stresses the viscosity inverse under additive noise and coarser masks. The version string stays `0.1.0`. The Stage A–D tables are not edited. `λ` stays 0.

- [docs/v02/stage_e_stress_protocol.json](docs/v02/stage_e_stress_protocol.json) freezes the noise model, the sensor and burst grid, the noise seeds, and the unchanged `0.5` failure rule before the test scores.

Stage D fits viscosity from the unchanged `sensors32_bursts` mask with the Stage B data-only FNO and the Stage C hybrid `1e-2` FNO. The version string stays `0.1.0`. The Stage A–C tables are not edited.

- [docs/v02/stage_d_inverse_protocol.json](docs/v02/stage_d_inverse_protocol.json) freezes the mask, the 191-point grid on `[0.005, 0.1]`, seeds 0 through 4, the `0.5` relative-error rule, and the Stage A `hard_ood` cut before the test scores. The residual weight is chosen on validation `hard_ood` only. Both arms select `λ = 0`.
- On test `hard_ood`, data-only mean relative error is `0.049487208469027544 ± 0.0144232271670265` and hybrid `1e-2` is `0.03368266375533967 ± 0.0034734247381767768`, against `0.3494591802034973` and 9 failures for the closed-form residual least squares. Both operator arms have 0 failures on that slice. Tables are in [STAGE_D_REPORT.md](STAGE_D_REPORT.md) and [docs/v02/stage_d_scores.json](docs/v02/stage_d_scores.json).

Stage C trains residual and hybrid losses on the Stage A harder Burgers pilot. The version string stays `0.1.0`. The data-only arm is the Stage B table and is not retrained. There is no learned inverse.

- [docs/v02/stage_c_train_protocol.json](docs/v02/stage_c_train_protocol.json) freezes seeds 0 through 4, the Stage B architecture, the Stage 4 hybrid weights `{1e-6, 1e-4, 1e-2}`, and the Stage A `hard_ood` cut before the Stage C test scores. The weight is chosen on validation only. The selected weight is `1e-2`.
- On `hard_ood`, the hybrid mean relative L2 is `5.215427637735405e-03 ± 6.736875927560861e-04`, lower than the Stage B data-only `7.049252763443842e-03 ± 3.032654479273341e-04` on every seed. Residual-only is also lower on that slice and is not lower on every seed for the full test. Tables are in [STAGE_C_REPORT.md](STAGE_C_REPORT.md) and [docs/v02/stage_c_scores.json](docs/v02/stage_c_scores.json).

Stage B trains the Stage 3 data-only 1D FNO on the Stage A harder Burgers pilot. The version string stays `0.1.0`. The coordinate-PINN Burgers hook still raises `NotImplementedError`. There is no hybrid loss and no learned inverse in this stage.

- [docs/v02/stage_b_train_protocol.json](docs/v02/stage_b_train_protocol.json) freezes five seeds, the Stage 3 window and architecture, normalized data MSE, and the Stage A `hard_ood` cut before the test scores. The cut is not refit.
- `pinnforge fno` accepts `docs/stage_a/pilot_manifest.json` as well as the Stage 2 manifest. `fno slices` scores the full test split, `hard_ood`, and the complement. `fno aggregate` reports the mean and the sample standard deviation across the frozen seeds. The numbers are in [STAGE_B_REPORT.md](STAGE_B_REPORT.md) and [docs/v02/stage_b_scores.json](docs/v02/stage_b_scores.json).
- On these five seeds the `hard_ood` mean relative L2 is about twice the complement, and both slices beat the one-step persistence baseline. The Stage A `sensors32_bursts` inverse failures are unchanged.

Stage A adds a harder periodic Burgers pilot beside the Stage 2 dataset. The version string stays `0.1.0`. That stage does not itself train an operator. The coordinate-PINN Burgers hook still raises `NotImplementedError`.

- `tanh_bandlimited` initial data (modes through 48, `tanh(3 p_hat)` projected and max-normalized) and viscosities in `[0.005, 0.10]`. Labels use `N = 1024`, `dt = 2.5e-4`. The preregistered relative-L2 gate is `1e-9`. Counts, hashes, and the convergence tables are in [STAGE_A_REPORT.md](STAGE_A_REPORT.md) and [docs/stage_a/](docs/stage_a/).
- [docs/v02/pilot_protocol.json](docs/v02/pilot_protocol.json) freezes that family before any operator training. `hard_ood` is the training-split quartile of `ν` (`0.027028120493367818` on this draw). Later stages score that slice on its own.
- The unchanged Stage 5 `sensors32_bursts` residual least squares, scored on the harder test split, has mean absolute error about 50 times the Stage 5 easy-pilot table and 9 failures of the `0.5` relative-error rule, all inside `hard_ood`. The record is [docs/v02/inverse_stress.json](docs/v02/inverse_stress.json). No operator is trained.
- Stage 2 commands keep their defaults: `python -m pinnforge.reference.numerical pilot`. The harder set is `hard-pilot`.

Stage 5 recovers each pilot instance's viscosity from a preregistered sparse sensor mask. The version string stays `0.1.0`. The coordinate-PINN Burgers hook still raises `NotImplementedError`.

- `pinnforge fno inverse` minimizes the Stage 4 central Burgers residual in the scalar `ν`. The minimizer is the normal equation on 32 equispaced sensors and four short frame bursts. Test absolute and relative errors are in [STAGE5_REPORT.md](STAGE5_REPORT.md). No new Fourier layer is trained.

Stage 4 physics-informed losses on the Stage 3 FNO. The version string stays `0.1.0`. The coordinate-PINN Burgers hook still raises `NotImplementedError`. There is no inverse-viscosity fit in that stage.

- `pinnforge fno --loss` selects normalized data MSE, a discrete Burgers residual, or a weighted sum. The residual uses the pilot spectral derivative and a central time difference. The checkpoint is still the best validation relative L2. Measured test numbers are in [STAGE4_REPORT.md](STAGE4_REPORT.md).

Stage 3 data-only Fourier neural operator. The version string stays `0.1.0`. The coordinate-PINN Burgers hook still raises `NotImplementedError`. That baseline does not put a PDE residual in the loss. There is no inverse-viscosity fit.

- 1D FNO on temporal windows of the Stage 2 Burgers pilot, split by problem instance before windowing. `u` uses the training mean and standard deviation in `docs/stage2/pilot_manifest.json`. Viscosity is a constant input channel normalized on the training split only.
- Supervised mean squared error in normalized space, validation relative L2 for epoch selection, and test relative L2 in [STAGE3_REPORT.md](STAGE3_REPORT.md). Commands: `pinnforge fno` and `python -m pinnforge.operator`.

Stage 2 periodic Burgers reference. The version string stays `0.1.0`. The coordinate-PINN Burgers hook still raises `NotImplementedError`.

- Dealiased Fourier spectral solver with ETDRK4 for `u_t + u u_x = ν u_xx` on `x ∈ [-1, 1]`, `t ∈ [0, 1]`, in `pinnforge.reference.numerical`.
- Cole–Hopf and finite-difference cross-checks, plus a convergence study under `docs/stage2/`.
- Seeded low-frequency initial conditions, viscosities in `[0.02, 0.10]`, and a 512/128/128 pilot split by problem instance. The manifest is committed. The trajectory arrays are regenerated with `python -m pinnforge.reference.numerical pilot`.

Stage 1 correctness. The version string stays `0.1.0`.

- Soft penalties for outward Neumann flux and for periodic equality of the field and its derivative at paired endpoints. A training spec whose boundary condition cannot be represented is rejected.
- Poisson references are returned only when they match the source, the domain, and every prescribed boundary condition. Source `one` on `[0, 1]` with zero ends is `x(1 - x) / 2`.
- Separate train, validation, and test RNG streams, logged in the run manifest. Evaluation reports held-out initial and boundary errors and relative and max field error.
- Demo configs are package data, read with `importlib.resources`.
- `relu` is rejected for strong-form second-derivative residuals. Checkpoints can be written on an interval, record a manifest, and resume Adam and RNG state. Loads use `weights_only=True`.

## 0.1.0 — 2026-09-26

First public freeze of the physics-informed sandbox.

- Equation schemas for the harmonic oscillator, Burgers 1D, and a Poisson toy, plus a harmonic closed form.
- Seeded collocation, initial-condition, and boundary samplers.
- CPU MLP, autograd residuals, and an Adam trainer with checkpoints and metrics.
- Evaluation against the harmonic closed form or a manufactured Poisson field. Burgers reports residual metrics only.
- Experiment files and `pinnforge run --config`.
- Localhost FastAPI service (`pinnforge serve`) with a path sandbox, a rate limit, and an offline socket guard.
- Offline demo (`pinnforge demo`) on the checked-in samples.

[PROJECT_STATUS.md](PROJECT_STATUS.md) is the longer status note.
