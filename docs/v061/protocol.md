# v061 protocol (frozen before test scores)

This note is the human reading of [protocol.json](protocol.json). That JSON is the contract. It was committed after [identifiability.md](identifiability.md) and before any v061 test split was scored. It holds no v061 test number. It does not edit [docs/v06/](../v06/) or earlier study trees.

The dataset is The Well `shear_flow` on Hugging Face (`polymathic-ai/shear_flow`), CC BY 4. Access is anonymous `huggingface_hub`. Raw HDF5 is never committed.

## Question

On the identifiable Reynolds `10000` subset of The Well `shear_flow`, with native `256` by `512` fields and the `late_9` window, can a solver-free weak-form coefficient fit, a data-only FNO inverse on two probe lattices, and an FNO-bracket hybrid with split-conformal intervals recover `kappa = 1/(Re Sc)` from sparse noisy probes, against the frozen v06 bars plus a constant-predictor gate?

## Restriction from the diagnostic

No swept cell recovered both coefficients at every listed `(Re, Sc)`. The cheapest CPU-tractable cell that does so on the largest subset is native `256` by `512`, window `late_9` (`frame_start` `160`), Reynolds `10000`, Schmidt `0.1`, `0.2`, `0.5`, and `1`. The continuous inverse is `kappa`. `nu` is constant on that slice. Reynolds `50000` and higher, and Schmidt `5`, miss the identifiability gate `0.02` even on the native grid. Held-out Schmidt is `2`. Held-out Reynolds is reported and is not a pass bar.

## Subset and sensors

Four in-distribution Schmidt values at Reynolds `10000`. Official train files supply train (`32`) and calibration (`64`, at least `60`). Official valid files choose the FNO checkpoint. Official test files are the in-distribution test split. OOD uses only the held-out Schmidt valid and test files.

The probe lattices are `16` by `32` and `32` by `64` on the native grid. The time window is the `9` frames starting at index `160`. Noise fractions are `0`, `0.01`, and `0.05`. The FNO reads the lattice; it is not lifted onto the native grid.

## Methods

- **wsindy.** Weak-form least-squares for the two known coefficients. Test-function stride is the v052 `center_stride_fraction`. Pressure is a known probed field.
- **fno_data.** Reused `FNO2d` maps the noisy lattice to `(log nu, log kappa)`.
- **hybrid.** FNO log-bracket plus WSINDy clipped into that bracket, with split-conformal intervals calibrated on the `64` calibration instances. No PDE solver search.
- **oracle.** Same weak-form fit on the clean full native field. Ceiling only.
- **constant.** Training-set median of each coefficient. A method must sit below this predictor in-distribution to count as passing.

## Frozen bars (not retuned)

| bar | rule |
| --- | --- |
| in-dist median rel | hybrid `<= 0.05` at noise `0.01` |
| OOD median rel | hybrid `<= 0.1` at noise `0.01` on the held-out Schmidt |
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
- that a Dedalus solver was run
- that PySR was scored
- identifiability at Reynolds `50000` or higher
- a retuned bar after seeing test scores
- a change to any v06, v052, v051, v05, v041, or v04 number
