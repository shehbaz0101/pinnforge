# PINNForge documentation map

The top-level [README](../README.md) has the headline results, the quickstart, and what is not claimed. This page points to every report, protocol, and score file. Numbers live in the reports and the JSON files, not here.

Every study stage froze a protocol file before its test scores were computed. The report explains the run. The score file is the record. `pinnforge claims check` ([Stage G](v02/stage_g_claims.md)) ties the README, the changelog, the status note, and the notes in `docs/v02/` to those score files through [claims.yaml](claims.yaml).

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
