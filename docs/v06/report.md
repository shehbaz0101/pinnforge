# v06 The Well shear_flow inverse

Research note. The contract is [protocol.json](protocol.json). Its SHA-256 is stored as `protocol_sha256` in [scores.json](scores.json). Hashes of the committed JSON files are in [reproducibility.md](reproducibility.md). No v052 or earlier artifact was edited. Raw HDF5 is not committed. The dataset is The Well `shear_flow` (CC BY 4).

## Question

On a documented subset of The Well shear_flow, can a solver-free weak-form coefficient fit, a data-only FNO inverse, and an FNO-bracket hybrid with split-conformal intervals recover nu = 1/Re and kappa = 1/(Re Sc) from sparse noisy probes, against the frozen roadmap bars?

## Result: FAIL

| bar | value | pass |
| --- | --- | --- |
| in-dist median rel | 1.088413403421 | fail |
| OOD median rel | 2.253780358576188 | fail |
| OOD / in-dist | 2.070702502828718 | pass |
| interval coverage | 0.8888888888888888 | pass |
| hybrid OOD below both baselines | pass | pass |
| noise 0.05 not worse than WSINDy | pass | pass |

## Methods by noise

| noise | slice | method | median rel nu | median rel kappa | combined | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | in_dist | wsindy | 0.6829332583226666 | 4.079093417030547 | 2.381013337676607 | — |
| 0.00 | in_dist | fno_data | 0.7265107016289786 | 18.19812441220054 | 9.462317556914758 | — |
| 0.00 | in_dist | hybrid | 0.6829332583226666 | 0.7230541843499877 | 0.7029937213363272 | 0.8888888888888888 |
| 0.00 | in_dist | oracle | 0.07888787979647335 | 1.067625756379083 | 0.5732568180877783 | — |
| 0.00 | ood | wsindy | 43.24793833810012 | 49.50117586066622 | 46.37455709938317 | — |
| 0.00 | ood | fno_data | 1.156621245295039 | 152.1140125273974 | 76.63531688634625 | — |
| 0.00 | ood | hybrid | 1.425873102400389 | 2.992375139701938 | 2.209124121051163 | 0.5833333333333334 |
| 0.00 | ood | oracle | 1.383504218247076 | 19.93786058502263 | 10.66068240163485 | — |
| 0.01 | in_dist | wsindy | 16.57492990693011 | 9.512239853190366 | 13.04358488006024 | — |
| 0.01 | in_dist | fno_data | 0.7264812746092999 | 18.1955815332647 | 9.461031403937 | — |
| 0.01 | in_dist | hybrid | 1.230427367276824 | 0.9463994395651772 | 1.088413403421 | 0.8888888888888888 |
| 0.01 | in_dist | oracle | 0.07888787979647335 | 1.067625756379083 | 0.5732568180877783 | — |
| 0.01 | ood | wsindy | 148.2090129818162 | 106.0765689384314 | 127.1427909601238 | — |
| 0.01 | ood | fno_data | 1.154524945833798 | 152.1553930743652 | 76.65495901009952 | — |
| 0.01 | ood | hybrid | 1.699556803121213 | 2.808003914031163 | 2.253780358576188 | 0.5833333333333334 |
| 0.01 | ood | oracle | 1.383504218247076 | 19.93786058502263 | 10.66068240163485 | — |
| 0.05 | in_dist | wsindy | 86.8921215718549 | 43.20399633009402 | 65.04805895097446 | — |
| 0.05 | in_dist | fno_data | 0.726370963083205 | 18.18538924990716 | 9.455880106495181 | — |
| 0.05 | in_dist | hybrid | 1.456748284681761 | 0.9665317487155836 | 1.211640016698672 | 0.8888888888888888 |
| 0.05 | in_dist | oracle | 0.07888787979647335 | 1.067625756379083 | 0.5732568180877783 | — |
| 0.05 | ood | wsindy | 471.5943152669055 | 360.6361412211977 | 416.1152282440517 | — |
| 0.05 | ood | fno_data | 1.146168756167102 | 152.3183519429951 | 76.73226034958111 | — |
| 0.05 | ood | hybrid | 1.911783487514528 | 7.635294733577902 | 4.773539110546215 | 0.5833333333333334 |
| 0.05 | ood | oracle | 1.383504218247076 | 19.93786058502263 | 10.66068240163485 | — |

## What is not claimed

- a result on Petrobras 3W
- 3D
- the full ensemble
- that a Dedalus solver was run
- that PySR was scored
- a retuned bar after seeing test scores
- a change to any v052, v051, v05, v041, or v04 number

## Attribution

The Well `shear_flow` is CC BY 4 (Ohana et al.). The generator is Dedalus (Burns et al.). [dataset page](https://polymathic-ai.org/the_well/datasets/shear_flow/).

## Reproduce

```
python -m pinnforge.v06 download
python -m pinnforge.v06 pack
python -m pinnforge.v06 score --checkpoint artifacts/v06_pilot/score_cases.jsonl
python -m pinnforge.v06 report
```

CI runs `python -m pinnforge.v06 smoke` only. It never downloads The Well.

