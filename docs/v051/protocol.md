# v0.5.1 protocol (frozen before test scores)

This note is the human reading of [protocol.json](protocol.json). That JSON is the contract. It was committed before any v0.5.1 test split was scored. It holds no v0.5.1 test number. It does not edit [docs/v05/](../v05/), [docs/v041/](../v041/), or [docs/v04/](../v04/).

## Question

At what sensor density does numpy WSINDy recover the exact PDE support on the existing 2D Burgers and vorticity-form Navier-Stokes generators?

## Lattices

Equispaced probes on the frozen 64-grid: `8x8`, `16x16`, `32x32`, and full-field `64x64`. Bursts and noise fractions match v0.5 (`0`, `0.01`, `0.02`, `0.05`). Five seeds. The identifier is `wsindy_raw` only. The FNO-denoise arm is optional and is not run.

## Reported quantity (not retuned)

The smallest lattice whose seed exact-support rate at noise `0.01` is `>= 0.8`, per problem, or `none`. Thresholds are chosen on the validation split only. The library is the v0.5 library, including distractors.

## Positive control

`64x64` at noise `0` is the full-field control. If that cell is below `0.8` seed exact support, the reading is a method or implementation failure, not a sparsity finding.

## What is NOT claimed

- a result on The Well `shear_flow` or Petrobras 3W
- 3D
- that the FNO-denoise arm was scored
- that PySR was scored
- a retuned bar after seeing test scores
- a change to any v05, v041, or v04 number
