# v051 sensor-density study

Research note. The contract is [protocol.json](protocol.json). Its SHA-256 is stored as `protocol_sha256` in [scores.json](scores.json). Hashes of the committed JSON files are in [reproducibility.md](reproducibility.md). Thresholds chosen on validation are [validation.json](validation.json). No v05, v041, or v04 artifact was edited. No hyperparameter was chosen on the test split. The FNO-denoise arm and PySR were not run; those skips are in the protocol.

## Question

At what sensor density does numpy WSINDy recover the exact PDE support on the existing 2D Burgers and vorticity-form Navier-Stokes generators?

## Reported quantity

Smallest lattice whose `wsindy_raw` seed exact-support rate at noise `0.01` is `>= 0.8`, or `none`. The bar was not retuned.

| problem | min lattice @ 0.01 | 64x64 noise 0 control | control reading |
| --- | --- | --- | --- |
| burgers2d | 64x64 | 1 | control passed |
| ns2d | none | 0 | method/implementation failure |

## Density by noise

| problem | lattice | noise | seed exact | instance exact | coeff rel err |
| --- | --- | --- | --- | --- | --- |
| burgers2d | 8x8 | 0.00 | 0 | 0.3333333333333333 | 0.1722130637663681 |
| burgers2d | 8x8 | 0.01 | 0 | 0.09999999999999999 | 0.4127669100527364 |
| burgers2d | 8x8 | 0.02 | 0.2 | 0.01666666666666667 | 0.5597734290381388 |
| burgers2d | 8x8 | 0.05 | 0 | 0 | 0.8466240661070547 |
| burgers2d | 16x16 | 0.00 | 1 | 0 | 0.4951634026446608 |
| burgers2d | 16x16 | 0.01 | 0.2 | 0.01666666666666667 | 5.293000734231838 |
| burgers2d | 16x16 | 0.02 | 0 | 0 | 9.215997281281483 |
| burgers2d | 16x16 | 0.05 | 0.2 | 0 | 21.9117026131553 |
| burgers2d | 32x32 | 0.00 | 1 | 1 | 0.0002496551126552383 |
| burgers2d | 32x32 | 0.01 | 0.4 | 0.5333333333333334 | 0.206358340636994 |
| burgers2d | 32x32 | 0.02 | 0.4 | 0.1833333333333333 | 0.5764961777469202 |
| burgers2d | 32x32 | 0.05 | 0.4 | 0.03333333333333333 | 1.379461120418692 |
| burgers2d | 64x64 | 0.00 | 1 | 1 | 0.0002520453187600139 |
| burgers2d | 64x64 | 0.01 | 1 | 0.9166666666666667 | 0.02571004789129059 |
| burgers2d | 64x64 | 0.02 | 0.4 | 0.7166666666666666 | 0.08967984080655926 |
| burgers2d | 64x64 | 0.05 | 0.2 | 0.2833333333333333 | 0.3276254706299899 |
| ns2d | 8x8 | 0.00 | 0 | 0 | 0.5434178428069232 |
| ns2d | 8x8 | 0.01 | 0 | 0 | 0.746914782392211 |
| ns2d | 8x8 | 0.02 | 0 | 0 | 1.091022589172221 |
| ns2d | 8x8 | 0.05 | 0 | 0 | 2.224955722565634 |
| ns2d | 16x16 | 0.00 | 0 | 0 | 0.4520029179092058 |
| ns2d | 16x16 | 0.01 | 0 | 0 | 9.183196940885345 |
| ns2d | 16x16 | 0.02 | 0 | 0 | 16.33717488997174 |
| ns2d | 16x16 | 0.05 | 0 | 0 | 50.02800933774664 |
| ns2d | 32x32 | 0.00 | 0 | 0 | 0.453114794924064 |
| ns2d | 32x32 | 0.01 | 0 | 0 | 0.6744537854879052 |
| ns2d | 32x32 | 0.02 | 0 | 0 | 1.453640101889495 |
| ns2d | 32x32 | 0.05 | 0 | 0 | 2.678638601888111 |
| ns2d | 64x64 | 0.00 | 0 | 0 | 0.4531543827729688 |
| ns2d | 64x64 | 0.01 | 0 | 0 | 0.4868617057023817 |
| ns2d | 64x64 | 0.02 | 0 | 0 | 0.5324306568749484 |
| ns2d | 64x64 | 0.05 | 0 | 0 | 0.8920102505501979 |

## What is not claimed

- a result on The Well shear_flow or Petrobras 3W
- 3D
- that the FNO-denoise arm was scored
- that PySR was scored
- a retuned bar after seeing test scores
- a change to any v05, v041, or v04 number

## Reproduce

```
python -m pinnforge.v051 generate
python -m pinnforge.v051 score --checkpoint artifacts/v051_pilot/score_cases.jsonl
python -m pinnforge.v051 report
```

CI runs `python -m pinnforge.v051 smoke` only. Pilot arrays and the library-matrix JSONL stay under `artifacts/v051_pilot/` and are not committed. Scoring is resumable: finished JSONL rows are skipped.

