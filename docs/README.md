# PINNForge documentation map

The top-level [README](../README.md) has the headline results, the quickstart, and what is not claimed. This page points to every report, protocol, and score file. Numbers live in the reports and the JSON files, not here.

Every study stage froze a protocol file before its test scores were computed. The report explains the run. The score file is the record. `pinnforge claims check` ([Stage G](v02/stage_g_claims.md)) ties the README, the changelog, the status note, and the notes in `docs/v02/` to those score files through [claims.yaml](claims.yaml).

## v061 identifiable The Well shear_flow inverse

[v061/identifiability.md](v061/identifiability.md) is the validation-only full-field oracle sweep. No cell recovered both coefficients at every listed `(Re, Sc)`. The cheapest passing subset is the native grid and the `late_9` window on the Reynolds `10000` slice recorded in that note. [v061/report.md](v061/report.md) scores the inverse on that subset. The inverse fails the preregistered bars on the identifiable subset. The in-distribution median-rel bar, the OOD median-rel bar, the gap bar, the coverage bar, the hybrid-below-baselines bar, and the constant-predictor bar fail. The noise-robustness bar passes. Protocol, frozen before the test scores: [protocol.json](v061/protocol.json). Scores: [scores.json](v061/scores.json). The Well `shear_flow` is CC BY 4 (Ohana et al.). No v06 artifact was edited.

## v06 The Well shear_flow inverse

[v06/report.md](v06/report.md) scores a solver-free weak-form coefficient fit, a data-only FNO inverse, and an FNO-bracket hybrid with split-conformal intervals on a named subset of The Well `shear_flow`. The inverse fails the preregistered bars. The in-distribution median-rel bar and the OOD median-rel bar fail. The OOD-over-in-dist gap bar, the interval coverage bar, the hybrid-below-baselines bar, and the noise-robustness bar pass. Protocol, frozen before the test scores: [protocol.json](v06/protocol.json). Download provenance: [download_manifest.json](v06/download_manifest.json). Validation: [validation.json](v06/validation.json). Scores: [scores.json](v06/scores.json). Reproducibility: [reproducibility.md](v06/reproducibility.md). Raw HDF5 is not committed. The Well `shear_flow` is CC BY 4 (Ohana et al.). No v052 or earlier artifact was edited.

## v0.2 viscosity inverse study

All of these use the Stage A harder periodic Burgers pilot and its frozen `hard_ood` slice.

| stage | question | report | protocol | scores |
| --- | --- | --- | --- | --- |
| A | Harder pilot, `hard_ood` cut, closed-form stress | [STAGE_A_REPORT.md](../STAGE_A_REPORT.md) | [pilot_protocol.json](v02/pilot_protocol.json), [stage_a/](stage_a/) | [inverse_stress.json](v02/inverse_stress.json) |
| B | Data-loss FNO forecast, five seeds | [STAGE_B_REPORT.md](../STAGE_B_REPORT.md) | [stage_b_train_protocol.json](v02/stage_b_train_protocol.json) | [stage_b_scores.json](v02/stage_b_scores.json) |
| C | Residual and hybrid physics losses | [STAGE_C_REPORT.md](../STAGE_C_REPORT.md) | [stage_c_train_protocol.json](v02/stage_c_train_protocol.json), [stage_c_weight_selection.json](v02/stage_c_weight_selection.json) | [stage_c_scores.json](v02/stage_c_scores.json) |
| D | FNO-conditioned viscosity inverse on `sensors32_bursts` | [STAGE_D_REPORT.md](../STAGE_D_REPORT.md) | [stage_d_inverse_protocol.json](v02/stage_d_inverse_protocol.json), [stage_d_objective_selection.json](v02/stage_d_objective_selection.json) | [stage_d_scores.json](v02/stage_d_scores.json) |
| E | Noise and sensor sparsity stress | [STAGE_E_REPORT.md](../STAGE_E_REPORT.md) | [stage_e_stress_protocol.json](v02/stage_e_stress_protocol.json) | [stage_e_scores.json](v02/stage_e_scores.json), [stage_e/](v02/stage_e/) curves |
| F | Out-of-distribution viscosity | [STAGE_F_REPORT.md](../STAGE_F_REPORT.md) | [stage_f_ood_protocol.json](v02/stage_f_ood_protocol.json) | [stage_f_scores.json](v02/stage_f_scores.json) |
| G | Claim-to-evidence checker | [stage_g_claims.md](v02/stage_g_claims.md) | [claims.yaml](claims.yaml) | the score files above |

## Research baselines

[research/strong_baselines.md](research/strong_baselines.md) compares the Stage D FNO inverse with a weak-form identifier and two solver-in-the-loop fits on the Stage D mask, plus the Stage E noise and sparsity cells. Protocol: [strong_baselines_protocol.json](research/strong_baselines_protocol.json). Scores: [strong_baselines_scores.json](research/strong_baselines_scores.json). Every number in that note is a leaf of those two files.

## v0.5.2 Navier-Stokes WSINDy fix

[v052/report.md](v052/report.md) rescores the v0.5.1 density grid for Navier-Stokes only, after breaking the Kolmogorov-period lock on WSINDy bump centers. Navier-Stokes first reaches the frozen bar at `32x32`. The `64x64` noise-0 control passes. Burgers is not rescored; it remains `64x64` from v0.5.1. Protocol, frozen before the test scores: [protocol.json](v052/protocol.json). Diagnosis: [diagnosis.md](v052/diagnosis.md). Validation: [validation.json](v052/validation.json). Scores: [scores.json](v052/scores.json). Reproducibility: [reproducibility.md](v052/reproducibility.md). No v051 or v05 artifact was edited.

## v0.5.1 sensor-density study

[v051/report.md](v051/report.md) sweeps WSINDy over `8x8`, `16x16`, `32x32`, and full-field `64x64` probes. Burgers first reaches the frozen bar at `64x64`. Navier-Stokes reaches it at none. The `64x64` noise-0 control passes on Burgers and fails on Navier-Stokes (method or implementation failure, not a sparsity finding). Protocol, frozen before the test scores: [protocol.json](v051/protocol.json). Validation: [validation.json](v051/validation.json). Scores: [scores.json](v051/scores.json). Reproducibility: [reproducibility.md](v051/reproducibility.md). No v05 artifact was edited.

## v0.5 2D PDE term discovery

[v05/report.md](v05/report.md) scores WSINDy term discovery on the existing 2D Burgers and vorticity-form Navier-Stokes generators. Both problems fail the preregistered support-recovery bar. Protocol, frozen before the test scores: [protocol.json](v05/protocol.json). Validation choices: [validation.json](v05/validation.json). Scores: [scores.json](v05/scores.json). Reproducibility: [reproducibility.md](v05/reproducibility.md). Every number in the scored note is a leaf of those JSON files. No v04 or v041 artifact was edited.

## v0.4.1 2D viscosity inverse

[v041/report.md](v041/report.md) is the follow-up with a 64-instance calibration split and an FNO-located conformal bracket plus a short local solver search. Both problems fail the preregistered bars. The hard_ood gap bar and the coverage bar pass. The in-distribution error bar, the warm-versus-cold error bar, and the time bar fail. Protocol, frozen before the test scores: [protocol.json](v041/protocol.json). Validation choices: [validation.json](v041/validation.json). Scores: [scores.json](v041/scores.json). Reproducibility: [reproducibility.md](v041/reproducibility.md). Every number in the scored note is a leaf of those JSON files. No v0.4 artifact was edited.

## v0.4 2D viscosity inverse

[v04/report.md](v04/report.md) scores the 2D Burgers and vorticity-form Navier-Stokes viscosity inverse. Both problems fail the preregistered bars. The time bar passes. The in-distribution error bar, the warm-versus-cold error bar, and the coverage bar fail. Protocol, frozen before the test scores: [protocol.json](v04/protocol.json). Validation choices: [validation.json](v04/validation.json). Scores: [scores.json](v04/scores.json). Reproducibility: [reproducibility.md](v04/reproducibility.md). Every number in the scored note is a leaf of those JSON files.

## v0.3 warm-started solver fit

[v03/warmstart.md](v03/warmstart.md) starts the sensor-lift solver fit from an FNO first guess, caps the refinement solves, and adds split-conformal intervals and an abstain flag calibrated on validation. On `hard_ood` the FNO hybrid warm start passes all three preregistered bars (error, wall clock against the cold search re-timed in the same run, and full-test coverage). Protocol, frozen before the test scores: [warmstart_protocol.json](v03/warmstart_protocol.json). Validation choices: [warmstart_validation.json](v03/warmstart_validation.json). Scores: [warmstart_scores.json](v03/warmstart_scores.json). Every number in that note is a leaf of those three files.

## Demo

[demo/result.md](demo/result.md) and [demo/index.html](demo/index.html) are the committed run of `pinnforge demo burgers-inverse`, a small smooth Burgers sample that is easier than `hard_ood`. The record is [demo/result.json](demo/result.json). Plots: [viscosity_scatter.png](demo/viscosity_scatter.png), [relative_error.png](demo/relative_error.png), [field_sensors.png](demo/field_sensors.png).

## v0.1 sandbox and Stages 1–5

| stage | topic | report |
| --- | --- | --- |
| Days 1–10 | coordinate-MLP PINN sandbox, API, freeze at `v0.1.0` | [daily/](daily/), [PROJECT_STATUS.md](../PROJECT_STATUS.md) |
| 1 | boundary penalties, Poisson references, evaluation splits | [STAGE1_REPORT.md](../STAGE1_REPORT.md) |
| 2 | periodic Burgers Fourier reference and pilot | [STAGE2_REPORT.md](../STAGE2_REPORT.md), [stage2/](stage2/) |
| 3 | data-only FNO on Burgers windows | [STAGE3_REPORT.md](../STAGE3_REPORT.md) |
| 4 | residual and hybrid losses on that FNO | [STAGE4_REPORT.md](../STAGE4_REPORT.md) |
| 5 | closed-form viscosity from sparse sensors | [STAGE5_REPORT.md](../STAGE5_REPORT.md) |

## Package

- [architecture.md](architecture.md): modules, optional extras, the localhost API, and the path sandbox.
- [CHANGELOG.md](../CHANGELOG.md): release history.
- `pinnforge --help` lists the commands. `pinnforge fno --help` lists the stage study commands.
