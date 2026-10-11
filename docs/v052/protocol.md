# v052 protocol (frozen before test scores)

This note is the human reading of [protocol.json](protocol.json). That JSON is the contract. It was committed before any v052 test split was scored. It holds no v052 test number. It does not edit [docs/v051/](../v051/), [docs/v05/](../v05/), [docs/v041/](../v041/), or [docs/v04/](../v04/). The diagnosis that selected the registered fix is [diagnosis.md](diagnosis.md).

## Question

After breaking the Kolmogorov-period lock on WSINDy bump centers, at what sensor density does numpy WSINDy recover the exact vorticity-form Navier-Stokes support on the existing generator?

## Registered fix

`center_stride_fraction` is the value stored in protocol.json. The v051 value `0.25` placed every bump center one Kolmogorov period apart on `8x8`, `16x16`, `32x32`, and `64x64`, so `force_y`, `const`, and `force_x` were rank one. Kolmogorov forcing stays in the library as `force_y`. It is not subtracted as a known term. Burgers is not scored.

## Lattices

Equispaced probes on the frozen `64` grid: `8x8`, `16x16`, `32x32`, and full-field `64x64`. Bursts and noise fractions match v051 (`0`, `0.01`, `0.02`, `0.05`). Five seeds. The identifier is `wsindy_raw` only. The FNO-denoise arm is optional and is not run.

## Reported quantity (not retuned)

The smallest lattice whose seed exact-support rate at noise `0.01` is `>= 0.8`, for `ns2d`, or `none`. Thresholds are chosen on the validation split only. The library is the v05 library, including distractors.

## Positive control

`64x64` at noise `0` is the full-field control. If that cell is below `0.8` seed exact support, the reading is a method or implementation failure, not a sparsity finding.

## What is NOT claimed

- a result on The Well `shear_flow` or Petrobras 3W
- 3D
- that the FNO-denoise arm was scored
- that PySR was scored
- that Kolmogorov forcing was treated as a known right-hand-side term
- that Burgers was rescored
- a retuned bar after seeing test scores
- a change to any v051, v05, v041, or v04 number
