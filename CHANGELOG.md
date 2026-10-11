# Changelog

## Unreleased

### v06 The Well shear_flow inverse

- **FAIL.** `python -m pinnforge.v06` recovers `nu` and `kappa` from sparse probes of a named subset of The Well `shear_flow`. The protocol [docs/v06/protocol.json](docs/v06/protocol.json) was committed before the test split was scored. Raw HDF5 is not committed. PySR is skipped. CI runs smoke only and never downloads The Well. The subset is CC BY 4 (Ohana et al.).
- The Well shear_flow inverse fails the preregistered bars. The in-distribution median-rel bar and the OOD median-rel bar fail. The OOD-over-in-dist gap bar, the interval coverage bar, the hybrid-below-baselines bar, and the noise-robustness bar pass. The note is [docs/v06/report.md](docs/v06/report.md). The scores are [docs/v06/scores.json](docs/v06/scores.json). This is a recorded negative result.

### v0.5.2 Navier-Stokes WSINDy fix

- `python -m pinnforge.v052` rescores the v0.5.1 density grid for Navier-Stokes only, after the bump-center stride diagnosed in [docs/v052/diagnosis.md](docs/v052/diagnosis.md). The protocol [docs/v052/protocol.json](docs/v052/protocol.json) was committed before the test split was scored. Burgers was not rescored. No v0.5.1 or v0.5 artifact was edited. The FNO-denoise arm is skipped. CI runs smoke only.
- The v0.5.1 NS `64x64` noise-0 miss was a rank-one `force_y`/`const`/`force_x` lock: centers sat one Kolmogorov period apart. The generator PDE already matched the library true support. After the registered stride, the NS control passes and the frozen bar is first met at `32x32`. Burgers remains `64x64` from v0.5.1. The note is [docs/v052/report.md](docs/v052/report.md). The scores are [docs/v052/scores.json](docs/v052/scores.json).

### v0.5.1 sensor-density study

- `python -m pinnforge.v051` sweeps WSINDy on `8x8`, `16x16`, `32x32`, and full-field `64x64` probes of the existing 2D Burgers and Navier-Stokes generators. The protocol [docs/v051/protocol.json](docs/v051/protocol.json) was committed before the test split was scored. The v0.5 artifacts were not edited. The FNO-denoise arm is skipped. CI runs smoke only.
- Burgers first reaches the frozen bar at `64x64`. Navier-Stokes reaches it at none. The `64x64` noise-0 control passes on Burgers and fails on Navier-Stokes; that NS miss is a method or implementation failure, not a sparsity finding. The note is [docs/v051/report.md](docs/v051/report.md). The scores are [docs/v051/scores.json](docs/v051/scores.json).

### v0.5 2D PDE term discovery

- **FAIL.** `python -m pinnforge.v05` recovers which frozen library terms are present on the existing 2D Burgers and vorticity-form Navier-Stokes generators, from sparse noisy sensors, using numpy WSINDy on raw probes and on fields denoised by the reused 2D FNO. The protocol [docs/v05/protocol.json](docs/v05/protocol.json) was committed before the test split was scored. PySR is skipped because it needs Julia. CI runs smoke only.
- Both problems fail the preregistered support-recovery bar. The note is [docs/v05/report.md](docs/v05/report.md). The scores are [docs/v05/scores.json](docs/v05/scores.json). This is a recorded negative result.

### v0.4.1 2D viscosity inverse

- **FAIL.** `python -m pinnforge.v041` uses a 64-instance calibration split and an FNO-bracket local search on the same 2D Burgers and Navier-Stokes problems. The protocol [docs/v041/protocol.json](docs/v041/protocol.json) was committed before the test split was scored. The v0.4 artifacts were not edited. The v0.4 pass bars were not retuned.
- Both problems fail the preregistered bars. The hard_ood gap bar and the coverage bar pass. The in-distribution error bar, the warm-versus-cold error bar, and the time bar fail. The note is [docs/v041/report.md](docs/v041/report.md). The scores are [docs/v041/scores.json](docs/v041/scores.json). This is a recorded negative result.

### v0.4 2D viscosity inverse

- **FAIL.** `python -m pinnforge.v04` generates periodic 2D Burgers and vorticity-form Navier-Stokes pilots, trains a 2D FNO, and scores the frozen inverse methods. The protocol [docs/v04/protocol.json](docs/v04/protocol.json) was committed before the test split was scored. The protocol, scores, and report were not edited after seeing the test numbers.
- Both problems fail the preregistered bars. The time bar passes. The in-distribution error bar, the warm-versus-cold error bar, and the coverage bar fail. The note is [docs/v04/report.md](docs/v04/report.md). The scores are [docs/v04/scores.json](docs/v04/scores.json). This is a recorded negative result.

## 0.3.0 (2026-10-11)

Release of the work on `main` after the `v0.2.2` tag. The package version is `0.3.0`. Tag `v0.3.0` is cut on the merge of this pull request. The measured scores below were not recomputed or edited for this release.

### v0.3 warm-start study

- `python -m pinnforge.operator warmstart {validate,test,report,smoke}` refines an FNO first guess with a capped Gauss-Newton solver fit in log viscosity, then adds split-conformal intervals and an abstain flag. The solve cap, interval scale, and abstain statistic were chosen on validation. The protocol [docs/v03/warmstart_protocol.json](docs/v03/warmstart_protocol.json) was committed before the test split was scored. The networks were retrained with the frozen Stage B, C, and F commands because the published checkpoints are not committed.
- On `hard_ood` the FNO hybrid warm start passes all three preregistered bars: error within the limit set from the sensor-lift solver, wall clock against the cold search re-timed in the same run, and full-test coverage. The re-run cold search matches the committed strong-baseline estimates. At 2% and 5% noise the warm start has higher `hard_ood` error than the cold search, and so does the warm start with restricted-range operators. The note is [docs/v03/warmstart.md](docs/v03/warmstart.md). The scores are [docs/v03/warmstart_scores.json](docs/v03/warmstart_scores.json).

### Strong baselines

- `python -m pinnforge.operator strong-baselines` scores a weak-form identifier and two solver-in-the-loop fits, one from a sensor-lifted initial condition and one from the true initial field (a ceiling), on the Stage D `sensors32_bursts` mask. The protocol [docs/research/strong_baselines_protocol.json](docs/research/strong_baselines_protocol.json) was frozen before the test scores. The FNO rows are the committed Stage D estimates and were not retrained.
- On `hard_ood` the sensor-lift solver has lower mean relative error than both FNO arms, and the weak form has higher error than both. The note is [docs/research/strong_baselines.md](docs/research/strong_baselines.md). The scores are [docs/research/strong_baselines_scores.json](docs/research/strong_baselines_scores.json). No Stage D or Stage E number was edited.

### Stage G: claim to evidence

- `python -m pinnforge.claims check` reads [docs/claims.yaml](docs/claims.yaml) and fails on a registered number that does not match its score file, a false registered comparison, an unregistered superiority phrase, or an unqualified FNO ranking on `hard_ood`. CI runs it as the `claim-to-evidence` job. The note is [docs/v02/stage_g_claims.md](docs/v02/stage_g_claims.md).
- The README and demo sentence about the FNO on `hard_ood` now says lower error than the closed form and higher than the sensor-lift solver.

### Stage H: docs and package surface

- The README leads with the problem, the `hard_ood` table from the strong-baseline scores, the solver and FNO wall clock, the Stage E and Stage F breakdowns, a quickstart, the stage reports, and what is not claimed. The demo tables stay in a shorter section. Every number and comparison in it is registered in `docs/claims.yaml`.
- [docs/README.md](docs/README.md) maps the reports, protocols, and score files for Stages A–G, the strong baselines, the demo, and the v0.1 Stages 1–5. It is in the claim scan.
- `pinnforge --help` lists every subcommand on one line. `pinnforge claims check` is the same command as `python -m pinnforge.claims check`. Smoke tests cover the module help, the console script, and the forwarded `fno` and `claims` help.
- This changelog records the `v0.2.0`, `v0.2.1`, and `v0.2.2` tags and adds the Stage F entry that the `v0.2.0` release note lists.

## 0.2.2 (2026-10-10)

Tag `v0.2.2` on `d808c5d`, the merge of [PR #25](https://github.com/shehbaz0101/pinnforge/pull/25). GitHub release: "v0.2.2: reports match the CLI". Measured demo scores are unchanged. `result.md` and `index.html` were rewritten from the existing [docs/demo/result.json](docs/demo/result.json). The plots and that JSON were not regenerated.

```bash
python -c "import json; from pathlib import Path; from pinnforge.demo_inverse import rewrite_reports; rewrite_reports(Path('docs/demo'), json.loads(Path('docs/demo/result.json').read_text()))"
```

- `result.md` and `index.html` lead with the aligned percent table and the one-line reading that `pinnforge demo burgers-inverse` prints. The scientific-notation table stays under that reading. `result.json` keeps full precision.

## 0.2.1 (2026-10-10)

Tag `v0.2.1` on `2d0cbe4`, the merge of [PR #23](https://github.com/shehbaz0101/pinnforge/pull/23). GitHub release: "v0.2.1: demo polish". Measured demo scores are unchanged. The plots were redrawn from [docs/demo/result.json](docs/demo/result.json) and the Fourier solver:

```bash
python -c "import json; from pathlib import Path; from pinnforge.demo_inverse import redraw_plots; redraw_plots(Path('docs/demo'), json.loads(Path('docs/demo/result.json').read_text()))"
```

- Demo plots use Matplotlib's bundled DejaVu Sans at 120 DPI. The previous 3x5 bitmap drew `closed`, `base`, and `rel error` as jagged mixed-case pixels.
- `pinnforge demo burgers-inverse` prints an aligned table with relative errors in percent and a one-line reading of that table. `result.json` and `result.md` keep full precision.
- `--output` on that command accepts an absolute directory. `POST /demo/burgers-inverse` still requires a relative path inside the sandbox.

## 0.2.0 (2026-10-10)

Tag `v0.2.0` on `6b255f9`, the merge of [PR #22](https://github.com/shehbaz0101/pinnforge/pull/22). GitHub release: "v0.2.0: Burgers viscosity inverse demo". The stage studies below were run while the package version was `0.1.0`. Their tables were not recomputed or edited for this release. Headlines in those tables are checked against `docs/v02/*.json` by `tests/test_published_scores.py`.

- `pinnforge demo burgers-inverse` trains a small data-only 1D FNO on 24 periodic Burgers trajectories (`N = 64`, 4 Fourier modes, viscosity uniform on `[0.02, 0.08]`, 40 Adam epochs, seed 0) and recovers viscosity on 4 held-out instances from 32 equispaced sensors. The closed form is the normal equation of the central residual. The FNO estimate is a grid search on sensor rollout error. The training-mean viscosity is the constant baseline. The run recorded in this commit is [docs/demo/result.json](docs/demo/result.json). The README table is that file. Command: `pinnforge demo burgers-inverse --output runs/burgers-inverse`.
- `POST /demo/burgers-inverse` serves the same demo on the localhost API. It shares the existing rate limit.
- On this smooth sample the closed form is more accurate than the FNO. Stage D uses the `sensors32_bursts` mask on the harder pilot. That study is separate from this command.

Stage E stresses the viscosity inverse under additive noise and coarser masks. These runs were made at package version `0.1.0`. The Stage A–D tables are not edited. `λ` stays 0.

- [docs/v02/stage_e_stress_protocol.json](docs/v02/stage_e_stress_protocol.json) freezes the noise model, the sensor and burst grid, the noise seeds, and the unchanged `0.5` failure rule before the test scores.
- On `hard_ood`, closed-form residual least squares is already past `10%` mean relative error and already has `9` failures at the clean `32`-sensor, `4`-burst mask. Both operator arms stay under `10%` with `0` failures through `1%` of field standard deviation, and both cross `10%` and start failing at `2%` noise and at `16` clean sensors. At `5%` noise the closed form is ahead on that slice. Tables and curves are in [STAGE_E_REPORT.md](STAGE_E_REPORT.md) and [docs/v02/stage_e_scores.json](docs/v02/stage_e_scores.json).

Stage F trains the Stage D operator arms only on viscosities above the frozen `hard_ood` cut and scores that slice as out of distribution. These runs were made at package version `0.1.0`. The Stage A–E tables are not edited.

- [docs/v02/stage_f_ood_protocol.json](docs/v02/stage_f_ood_protocol.json) freezes the restricted train and validation subsets, the Stage C weight `1e-2`, `λ = 0`, and an unclipped 191-point viscosity grid before the test scores.
- Forecasts inside the training range barely move. On the out-of-distribution slice the one-step error is about 11 times the in-range error for data-only and about 12 times for hybrid. The viscosity inverse on that slice has mean relative error `0.23310978476634858` (data-only) and `0.21277812472713906` (hybrid), against `0.049487208469027544` and `0.03368266375533967` in Stage D, where those viscosities were inside the training range. The hybrid loss does not extrapolate better than data-only. Tables are in [STAGE_F_REPORT.md](STAGE_F_REPORT.md) and [docs/v02/stage_f_scores.json](docs/v02/stage_f_scores.json).

Stage D fits viscosity from the unchanged `sensors32_bursts` mask with the Stage B data-only FNO and the Stage C hybrid `1e-2` FNO. These runs were made at package version `0.1.0`. The Stage A–C tables are not edited.

- [docs/v02/stage_d_inverse_protocol.json](docs/v02/stage_d_inverse_protocol.json) freezes the mask, the 191-point grid on `[0.005, 0.1]`, seeds 0 through 4, the `0.5` relative-error rule, and the Stage A `hard_ood` cut before the test scores. The residual weight is chosen on validation `hard_ood` only. Both arms select `λ = 0`.
- On test `hard_ood`, data-only mean relative error is `0.049487208469027544 ± 0.0144232271670265` and hybrid `1e-2` is `0.03368266375533967 ± 0.0034734247381767768`, against `0.3494591802034973` and 9 failures for the closed-form residual least squares. Both operator arms have 0 failures on that slice. Tables are in [STAGE_D_REPORT.md](STAGE_D_REPORT.md) and [docs/v02/stage_d_scores.json](docs/v02/stage_d_scores.json).

Stage C trains residual and hybrid losses on the Stage A harder Burgers pilot. These runs were made at package version `0.1.0`. The data-only arm is the Stage B table and is not retrained. There is no learned inverse.

- [docs/v02/stage_c_train_protocol.json](docs/v02/stage_c_train_protocol.json) freezes seeds 0 through 4, the Stage B architecture, the Stage 4 hybrid weights `{1e-6, 1e-4, 1e-2}`, and the Stage A `hard_ood` cut before the Stage C test scores. The weight is chosen on validation only. The selected weight is `1e-2`.
- On `hard_ood`, the hybrid mean relative L2 is `5.215427637735405e-03 ± 6.736875927560861e-04`, lower than the Stage B data-only `7.049252763443842e-03 ± 3.032654479273341e-04` on every seed. Residual-only is also lower on that slice and is not lower on every seed for the full test. Tables are in [STAGE_C_REPORT.md](STAGE_C_REPORT.md) and [docs/v02/stage_c_scores.json](docs/v02/stage_c_scores.json).

Stage B trains the Stage 3 data-only 1D FNO on the Stage A harder Burgers pilot. These runs were made at package version `0.1.0`. The coordinate-PINN Burgers hook still raises `NotImplementedError`. There is no hybrid loss and no learned inverse in this stage.

- [docs/v02/stage_b_train_protocol.json](docs/v02/stage_b_train_protocol.json) freezes five seeds, the Stage 3 window and architecture, normalized data MSE, and the Stage A `hard_ood` cut before the test scores. The cut is not refit.
- `pinnforge fno` accepts `docs/stage_a/pilot_manifest.json` as well as the Stage 2 manifest. `fno slices` scores the full test split, `hard_ood`, and the complement. `fno aggregate` reports the mean and the sample standard deviation across the frozen seeds. The numbers are in [STAGE_B_REPORT.md](STAGE_B_REPORT.md) and [docs/v02/stage_b_scores.json](docs/v02/stage_b_scores.json).
- On these five seeds the `hard_ood` mean relative L2 is about twice the complement, and both slices beat the one-step persistence baseline. The Stage A `sensors32_bursts` inverse failures are unchanged.

Stage A adds a harder periodic Burgers pilot beside the Stage 2 dataset. These runs were made at package version `0.1.0`. That stage does not itself train an operator. The coordinate-PINN Burgers hook still raises `NotImplementedError`.

- `tanh_bandlimited` initial data (modes through 48, `tanh(3 p_hat)` projected and max-normalized) and viscosities in `[0.005, 0.10]`. Labels use `N = 1024`, `dt = 2.5e-4`. The preregistered relative-L2 gate is `1e-9`. Counts, hashes, and the convergence tables are in [STAGE_A_REPORT.md](STAGE_A_REPORT.md) and [docs/stage_a/](docs/stage_a/).
- [docs/v02/pilot_protocol.json](docs/v02/pilot_protocol.json) freezes that family before any operator training. `hard_ood` is the training-split quartile of `ν` (`0.027028120493367818` on this draw). Later stages score that slice on its own.
- The unchanged Stage 5 `sensors32_bursts` residual least squares, scored on the harder test split, has mean absolute error about 50 times the Stage 5 easy-pilot table and 9 failures of the `0.5` relative-error rule, all inside `hard_ood`. The record is [docs/v02/inverse_stress.json](docs/v02/inverse_stress.json). No operator is trained.
- Stage 2 commands keep their defaults: `python -m pinnforge.reference.numerical pilot`. The harder set is `hard-pilot`.

Stage 5 recovers each pilot instance's viscosity from a preregistered sparse sensor mask. These runs were made at package version `0.1.0`. The coordinate-PINN Burgers hook still raises `NotImplementedError`.

- `pinnforge fno inverse` minimizes the Stage 4 central Burgers residual in the scalar `ν`. The minimizer is the normal equation on 32 equispaced sensors and four short frame bursts. Test absolute and relative errors are in [STAGE5_REPORT.md](STAGE5_REPORT.md). No new Fourier layer is trained.

Stage 4 physics-informed losses on the Stage 3 FNO. These runs were made at package version `0.1.0`. The coordinate-PINN Burgers hook still raises `NotImplementedError`. There is no inverse-viscosity fit in that stage.

- `pinnforge fno --loss` selects normalized data MSE, a discrete Burgers residual, or a weighted sum. The residual uses the pilot spectral derivative and a central time difference. The checkpoint is still the best validation relative L2. Measured test numbers are in [STAGE4_REPORT.md](STAGE4_REPORT.md).

Stage 3 data-only Fourier neural operator. These runs were made at package version `0.1.0`. The coordinate-PINN Burgers hook still raises `NotImplementedError`. That baseline does not put a PDE residual in the loss. There is no inverse-viscosity fit.

- 1D FNO on temporal windows of the Stage 2 Burgers pilot, split by problem instance before windowing. `u` uses the training mean and standard deviation in `docs/stage2/pilot_manifest.json`. Viscosity is a constant input channel normalized on the training split only.
- Supervised mean squared error in normalized space, validation relative L2 for epoch selection, and test relative L2 in [STAGE3_REPORT.md](STAGE3_REPORT.md). Commands: `pinnforge fno` and `python -m pinnforge.operator`.

Stage 2 periodic Burgers reference. These runs were made at package version `0.1.0`. The coordinate-PINN Burgers hook still raises `NotImplementedError`.

- Dealiased Fourier spectral solver with ETDRK4 for `u_t + u u_x = ν u_xx` on `x ∈ [-1, 1]`, `t ∈ [0, 1]`, in `pinnforge.reference.numerical`.
- Cole–Hopf and finite-difference cross-checks, plus a convergence study under `docs/stage2/`.
- Seeded low-frequency initial conditions, viscosities in `[0.02, 0.10]`, and a 512/128/128 pilot split by problem instance. The manifest is committed. The trajectory arrays are regenerated with `python -m pinnforge.reference.numerical pilot`.

Stage 1 correctness. These runs were made at package version `0.1.0`.

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
