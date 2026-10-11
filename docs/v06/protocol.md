# v06 protocol (frozen before test scores)

This note is the human reading of [protocol.json](protocol.json). That JSON is the contract. It was committed before any v06 test split was scored. It holds no v06 test number. It does not edit [docs/v052/](../v052/) or earlier study trees.

The dataset is The Well `shear_flow` on Hugging Face (`polymathic-ai/shear_flow`), CC BY 4. Access is anonymous `huggingface_hub`. Raw HDF5 is never committed.

## Question

On a documented subset of The Well `shear_flow`, can a solver-free weak-form coefficient fit, a data-only FNO inverse, and an FNO-bracket hybrid with split-conformal intervals recover `nu = 1/Re` and `kappa = 1/(Re Sc)` from sparse noisy probes, against the frozen roadmap bars?

## Subset and grid

Three in-distribution `(Re, Sc)` files plus one held-out Reynolds as OOD. Official train files supply the train and calibration instances (`n_cal` is `64`, at least `60`). Official valid files choose the FNO checkpoint. Official test files are the in-distribution test split. OOD uses only the held-out valid and test files.

Native stored fields are `256` by `512`. They are downsampled to `32` by `64` by periodic Fourier truncation (the anti-alias). The probe lattice is `8` by `16` on that coarse grid. The time window is the `9` frames starting at index `80`. Noise fractions are `0`, `0.01`, and `0.05`.

## Methods

- **wsindy.** Weak-form least-squares for the two known coefficients. Test-function stride is the v052 `center_stride_fraction`. Pressure is a known probed field. Not STLSQ support discovery. Not a Dedalus solve.
- **fno_data.** Reused `FNO2d` maps the lifted lattice to `(log nu, log kappa)`.
- **hybrid.** FNO log-bracket plus WSINDy clipped into that bracket, with split-conformal intervals calibrated on the `64` calibration instances. No PDE solver search.
- **oracle.** Same weak-form fit on the clean full coarse field. Ceiling only.

## Frozen bars (not retuned)

| bar | rule |
| --- | --- |
| in-dist median rel | hybrid `<= 0.05` at noise `0.01` |
| OOD median rel | hybrid `<= 0.1` at noise `0.01` |
| OOD / in-dist | hybrid `<= 3` at noise `0.01` |
| interval coverage | hybrid in `[0.88, 0.92]` at noise `0.01` |
| hybrid vs baselines | hybrid OOD median rel is below WSINDy and below data-only FNO |
| noise `0.05` | hybrid in-dist median rel `<=` WSINDy on that slice |

Overall pass only if every bar passes.

## What is NOT claimed

- a result on Petrobras 3W
- 3D
- the full ensemble
- that a Dedalus solver was run
- that PySR was scored
- a retuned bar after seeing test scores
- a change to any v052, v051, v05, v041, or v04 number
