# v0.5 2D PDE term discovery

Research note. The contract is [protocol.json](protocol.json). Its SHA-256 is stored as `protocol_sha256` in [scores.json](scores.json). Hashes of the committed JSON files are in [reproducibility.md](reproducibility.md). Thresholds chosen on validation are [validation.json](validation.json). No v04 or v041 artifact was edited. No hyperparameter was chosen on the test split. PySR was not run; that skip is in the protocol.

## Question

With sparse noisy sensors of the existing 2D Burgers and vorticity-form Navier-Stokes generators, can weak-form sparse regression (WSINDy) recover the exact PDE support — the true terms and no extras — on raw probes and on fields denoised by a reused 2D FNO, against a frozen `0.8` seed-success bar at noise `0.01`?

## Pass / fail

**FAIL.** The frozen bar is exact support in `>= 0.8` of seeds at noise `0.01` for at least one of `wsindy_raw` and `wsindy_fno` on each problem. Navier-Stokes uses the Kolmogorov family. The bar was not retuned.

| problem | method | exact support @ 0.01 | bar |
| --- | --- | --- | --- |
| burgers2d | `wsindy_raw` | 0 | fail |
| burgers2d | `wsindy_fno` | 0 | fail |
| burgers2d | overall |  | fail |
| ns2d | `wsindy_raw` | 0 | fail |
| ns2d | `wsindy_fno` | 0 | fail |
| ns2d | overall |  | fail |

Both problems fail overall. burgers2d fails; ns2d fails.

## Every noise fraction

| problem | noise | method | seed exact | instance exact | coeff rel err | FNO rel L2 |
| --- | --- | --- | --- | --- | --- | --- |
| burgers2d | 0.00 | `wsindy_raw` | 0 | 0.3333333333333333 | 0.1722130637663681 | — |
| burgers2d | 0.00 | `wsindy_fno` | 0 | 0 | 5.157108860954214 | 0.1452485954123657 |
| burgers2d | 0.01 | `wsindy_raw` | 0 | 0.09999999999999999 | 0.4127669100527364 | — |
| burgers2d | 0.01 | `wsindy_fno` | 0 | 0 | 5.185687385591988 | 0.1453612504954881 |
| burgers2d | 0.02 | `wsindy_raw` | 0.2 | 0.01666666666666667 | 0.5597734290381388 | — |
| burgers2d | 0.02 | `wsindy_fno` | 0 | 0 | 5.185495887234345 | 0.1455646243679156 |
| burgers2d | 0.05 | `wsindy_raw` | 0 | 0 | 0.8466240661070547 | — |
| burgers2d | 0.05 | `wsindy_fno` | 0 | 0 | 5.197625408098108 | 0.1467140665431189 |
| ns2d | 0.00 | `wsindy_raw` | 0 | 0 | 0.5434178428069232 | — |
| ns2d | 0.00 | `wsindy_fno` | 0 | 0 | 17.23019991295643 | 0.1556945181880136 |
| ns2d | 0.01 | `wsindy_raw` | 0 | 0 | 0.746914782392211 | — |
| ns2d | 0.01 | `wsindy_fno` | 0 | 0 | 17.21117913181028 | 0.1557176110307147 |
| ns2d | 0.02 | `wsindy_raw` | 0 | 0 | 1.091022589172221 | — |
| ns2d | 0.02 | `wsindy_fno` | 0 | 0 | 17.15093495630007 | 0.1558155898124704 |
| ns2d | 0.05 | `wsindy_raw` | 0 | 0 | 2.224955722565634 | — |
| ns2d | 0.05 | `wsindy_fno` | 0 | 0 | 17.1579529073303 | 0.1565568149338024 |

## What is not claimed

- a result on The Well shear_flow or Petrobras 3W
- 3D
- that PySR was scored
- a retuned bar after seeing test scores
- a change to any v04 or v041 number

## Reproduce

```
python -m pinnforge.v05 generate
python -m pinnforge.v05 train
python -m pinnforge.v05 score --checkpoint artifacts/v05_pilot/score_cases.jsonl
python -m pinnforge.v05 report
```

CI runs `python -m pinnforge.v05 smoke` only. Pilot arrays and the library-matrix JSONL stay under `artifacts/v05_pilot/` and are not committed. Scoring is resumable: finished JSONL rows are skipped.

