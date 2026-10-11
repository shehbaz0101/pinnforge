# v062 protocol (frozen before test scores)

This note is the human reading of [protocol.json](protocol.json). That JSON is the contract. It was committed after [solver_validation.md](solver_validation.md) and before any v062 The Well test split was scored. It holds no v062 test number. It does not edit [docs/v061/](../v061/) or earlier study trees.

Training fields come from `pinnforge.v062.solver`. Test fields are official The Well `shear_flow` files (CC BY 4).

## Question

If a data-only FNO and a solver-in-the-loop search are trained on continuously sampled Schmidt from a The Well-matched shear-flow solver, can they recover `kappa` on official The Well files at Reynolds `10000` against the frozen v06 bars plus a constant-predictor gate?

## Compute reduction (stated before scoring)

A native `256` by `512` integrate to `t = 16.8` is too slow on one free CPU. Training trajectories are generated at `64` by `128`. On those generated fields the `late_9` full-field oracle recovered both coefficients below the identifiability gate `0.02`. The Well test stays on the native `256` by `512` identifiable setting. SITL searches `kappa` only, `8` golden-section steps, window `0.8`, `dt = 0.005`.

## Training

Reynolds `10000`. Schmidt is log-uniform on `[0.08, 1.3]`. `32` train, `64` cal (at least `60`), `16` val. The IC family is the Dedalus `n_shear`, `n_blobs`, `width` grid.

## Test

Official The Well test files at Reynolds `10000` and Schmidt `0.1`, `0.2`, `0.5`, `1`. Held-out Schmidt `2` is the OOD bar. Probe lattices `16` by `32` and `32` by `64`. Noise fractions `0`, `0.01`, `0.05`.

## Methods

- **wsindy.** Weak-form least-squares on the probe lattice.
- **fno_data.** Reused `FNO2d` on the lattice, trained on the generated sims.
- **sitl_cold.** Kappa-only golden-section search on the `64` by `128` solver, started from the training-median kappa.
- **hybrid.** FNO log-bracket plus SITL kappa clipped into that bracket, with split-conformal intervals on the `64` sim calibration instances.
- **oracle.** Full-field weak form. Ceiling only.
- **constant.** Training-set median of each coefficient.

## Frozen bars (not retuned)

| bar | rule |
| --- | --- |
| in-dist median rel | hybrid `<= 0.05` at noise `0.01` |
| OOD median rel | hybrid `<= 0.1` at noise `0.01` |
| OOD / in-dist | hybrid `<= 3` at noise `0.01` |
| interval coverage | hybrid in `[0.88, 0.92]` at noise `0.01` |
| hybrid vs baselines | hybrid OOD median rel is below WSINDy and below data-only FNO |
| noise `0.05` | hybrid in-dist median rel `<=` WSINDy on that slice |
| vs constant | hybrid in-dist median rel is below the training-median predictor |

Overall pass only if every bar passes.

## What is NOT claimed

- a result on Petrobras 3W
- 3D
- the full ensemble
- that Dedalus was run in this study
- that PySR was scored
- identifiability at Reynolds `50000` or higher
- a retuned bar after seeing test scores
- a change to any v061 or earlier number
