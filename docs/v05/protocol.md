# v0.5 protocol (frozen before test scores)

This note is the human reading of [protocol.json](protocol.json). That JSON is the contract. It was committed before any v0.5 test split was scored. It holds no v0.5 test number. It does not edit [docs/v04/](../v04/) or [docs/v041/](../v041/).

## Question

With sparse noisy sensors of the existing 2D Burgers and vorticity-form Navier-Stokes generators, can weak-form sparse regression (WSINDy) recover the exact PDE support — the true terms and no extras — on raw probes and on fields denoised by a reused 2D FNO, against a frozen 80% seed-success bar at 1% noise?

## What is recovered

The unknown is the **support** of a frozen library, not only a scalar viscosity. Burgers truth is `{u q_x, v q_y, q_xx, q_yy}`. Navier-Stokes Kolmogorov truth adds the known forcing `-4 cos(4 y)`. Decaying Navier-Stokes drops that forcing. The library also holds distractors: `q²`, third derivatives, mixed derivatives, crossed advection, constants, and a wrong-axis forcing.

## Methods

- **wsindy_raw.** Compact-polynomial WSINDy on the 8 by 8 sensor lattice. Numpy only.
- **wsindy_fno.** The same identifier after a 2D FNO denoiser (the v0.4 `FNO2d` class, trained as noisy-lift → clean field, no viscosity channel).
- **PySR.** Not run. It needs Julia. That is recorded in the protocol, not after a failed score.

The STLSQ threshold is chosen on the validation split only. Five seeds. Noise fractions 0, 0.01, 0.02, 0.05. Initial conditions reuse the v0.4 band-limited families (`max_mode` 3 on Burgers, 4 on Navier-Stokes).

## Pass bar (not retuned)

Correct support in `>= 80%` of seeds at `1%` noise for at least one method on each problem. Navier-Stokes uses the Kolmogorov family for that bar.

## What is NOT claimed

- a result on The Well `shear_flow` or Petrobras 3W
- 3D
- that PySR was scored
- a retuned bar after seeing test scores
- a change to any v0.4 or v0.4.1 number
