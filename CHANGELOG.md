# Changelog

## Unreleased

Stage A adds a harder periodic Burgers pilot beside the Stage 2 dataset. The version string stays `0.1.0`. No operator is trained on it. The coordinate-PINN Burgers hook still raises `NotImplementedError`.

- `tanh_bandlimited` initial data (modes through 48, `tanh(3 p_hat)` projected and max-normalized) and viscosities in `[0.005, 0.10]`. Labels use `N = 1024`, `dt = 2.5e-4`. The preregistered relative-L2 gate is `1e-9`. Counts, hashes, and the convergence tables are in [STAGE_A_REPORT.md](STAGE_A_REPORT.md) and [docs/stage_a/](docs/stage_a/).
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
