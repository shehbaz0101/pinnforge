# v052 Navier-Stokes WSINDy fix

Research note. The contract is [protocol.json](protocol.json). Its SHA-256 is stored as `protocol_sha256` in [scores.json](scores.json). Hashes of the committed JSON files are in [reproducibility.md](reproducibility.md). Thresholds chosen on validation are [validation.json](validation.json). The diagnosis is [diagnosis.md](diagnosis.md). No v051, v05, v041, or v04 artifact was edited. Burgers was not rescored. No hyperparameter was chosen on the test split. The FNO-denoise arm and PySR were not run; those skips are in the protocol.

## Question

After breaking the Kolmogorov-period lock on WSINDy bump centers, at what sensor density does numpy WSINDy recover the exact vorticity-form Navier-Stokes support on the existing generator?

## Root cause

The v051 `64x64` noise-0 control failed because `center_stride_fraction` `0.25` placed every bump center one Kolmogorov period apart, so `force_y`, `const`, and `force_x` were rank one. The generator PDE and the library true support already matched. The registered fix is the incommensurate stride in protocol.json. Kolmogorov forcing stays in the library.

## Reported quantity

Smallest lattice whose `wsindy_raw` seed exact-support rate at noise `0.01` is `>= 0.8`, or `none`. The bar was not retuned. Burgers is the frozen v051 number.

| problem | min lattice @ 0.01 | 64x64 noise 0 control | control reading | source |
| --- | --- | --- | --- | --- |
| burgers2d | 64x64 | 1 | control passed | v051, not rescored |
| ns2d | 32x32 | 1 | control passed | this study |

## Density by noise (ns2d only)

| problem | lattice | noise | seed exact | instance exact | coeff rel err |
| --- | --- | --- | --- | --- | --- |
| ns2d | 8x8 | 0.00 | 0 | 0.5 | 0.09854547557600983 |
| ns2d | 8x8 | 0.01 | 0.2 | 0.06666666666666667 | 0.3004155333496733 |
| ns2d | 8x8 | 0.02 | 0 | 0 | 0.4092125681997053 |
| ns2d | 8x8 | 0.05 | 0 | 0 | 0.8460045349053036 |
| ns2d | 16x16 | 0.00 | 0 | 0 | 0.2597913870116459 |
| ns2d | 16x16 | 0.01 | 0.2 | 0 | 3.598851228028859 |
| ns2d | 16x16 | 0.02 | 0 | 0 | 7.51598897780787 |
| ns2d | 16x16 | 0.05 | 0 | 0 | 20.58529198537747 |
| ns2d | 32x32 | 0.00 | 0 | 1 | 0.0002688154048716325 |
| ns2d | 32x32 | 0.01 | 1 | 0.5 | 0.157086381570109 |
| ns2d | 32x32 | 0.02 | 0.8 | 0.2 | 0.2486281223916864 |
| ns2d | 32x32 | 0.05 | 0 | 0 | 0.4805877024704722 |
| ns2d | 64x64 | 0.00 | 1 | 1 | 0.0001556812261303891 |
| ns2d | 64x64 | 0.01 | 1 | 1 | 0.02261283064760737 |
| ns2d | 64x64 | 0.02 | 0.8 | 0.5666666666666667 | 0.1060666371842755 |
| ns2d | 64x64 | 0.05 | 1 | 0.03333333333333333 | 0.29958240761708 |

## What is not claimed

- a result on The Well shear_flow or Petrobras 3W
- 3D
- that the FNO-denoise arm was scored
- that PySR was scored
- that Kolmogorov forcing was treated as a known right-hand-side term
- that Burgers was rescored
- a retuned bar after seeing test scores
- a change to any v051, v05, v041, or v04 number

## Reproduce

```
python -m pinnforge.v052 generate
python -m pinnforge.v052 score --checkpoint artifacts/v052_pilot/score_cases.jsonl
python -m pinnforge.v052 report
```

CI runs `python -m pinnforge.v052 smoke` only. Pilot arrays and the library-matrix JSONL stay under `artifacts/v052_pilot/` and are not committed. Scoring is resumable: finished JSONL rows are skipped.

