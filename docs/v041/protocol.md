# v0.4.1 protocol (frozen before test scores)

This note is the human reading of [protocol.json](protocol.json). That JSON is the contract. It was committed before any v0.4.1 test split was scored. It holds no v0.4.1 test number. It does not edit [docs/v04/](../v04/).

The parent contract is [docs/v04/protocol.json](../v04/protocol.json), SHA-256 stored here as `parent_protocol_sha256`.

## Question

With a calibration split of at least 60 instances disjoint from train and test, can an FNO-located conformal bracket plus a short local solver search recover a scalar viscosity on the same 2D Burgers and vorticity-form Navier-Stokes problems, against the frozen v0.4 pass bars?

## Diagnosis (from frozen v0.4 scores, not this test)

Two v0.4 failures, cited from [docs/v04/scores.json](../v04/scores.json):

1. **Calibration.** `n_val` was 8. The split-conformal rank `ceil((n + 1)(1 - alpha))` is 9, so the quantile is infinite and coverage is 1, outside `[0.85, 0.95]`.
2. **Warm start.** The hybrid FNO inverse complement error was 0.5231465619444131 on Burgers and 0.7441866512832163 on Navier-Stokes. Gauss-Newton from that point reached only 0.168616089900106 and 0.3245006510224735, above 0.05 and far from the Burgers cold complement 0.004404494487213693.

Causes considered and not chosen as this protocol's lever: 24 training instances, width 12, modes 6, layers 2, 10 epochs; the 64-grid solver already passed its refinement gate; the 8 by 8 lift is faithful on Burgers and not on Navier-Stokes vorticity; a 21-point inverse grid. The chosen fix is the inverse, not a larger FNO.

## Chosen fix

- Calibration split `n_cal = 64` (`>= 60`), disjoint from train `24` and test `16`. The generator's `n_val` slot is that calibration split.
- FNO inverse locates a split-conformal log-bracket of `|log(ν / ν_FNO)|`.
- A short equispaced solver search inside that bracket is the reported point. No Gauss-Newton seed.
- Reported intervals use `|log(ν / ν̂)|` on the same 64 calibration rows, so the rank is finite.

## Pass bars (identical to v0.4; not retuned)

| bar | limit |
| --- | --- |
| in-distribution (`complement`) mean relative `ν` error | `<= 0.05` |
| `hard_ood` / complement gap ratio | `<= 3` |
| warm-start `hard_ood` error vs cold solver | `<= 1.5×` at `<= 0.25×` wall clock |
| full-test coverage of the nominal 0.9 interval | in `[0.85, 0.95]` |

All four must pass on Burgers hybrid warm start. The same bars are scored on Navier-Stokes and published as pass or fail without retuning.

## What is NOT claimed

- a result on The Well `shear_flow` or Petrobras 3W
- 3D or 3D+time
- that the warm start is more accurate than the cold search
- coverage conditional on viscosity, noise, or the training range
- that the abstain flag detects every large error
- a retuned `hard_ood` rule after seeing test scores
- a change to any v0.4 number
- that a larger FNO would have passed v0.4
