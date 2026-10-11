# v061 identifiable The Well shear_flow inverse

Research note. The contract is [protocol.json](protocol.json). Its SHA-256 is stored as `protocol_sha256` in [scores.json](scores.json). The validation-only oracle sweep is [identifiability.md](identifiability.md). No v06 or earlier artifact was edited. Raw HDF5 is not committed. The dataset is The Well `shear_flow` (CC BY 4).

## Question

On the identifiable Reynolds 10000 subset of The Well shear_flow, with native 256 by 512 fields and the late_9 window, can a solver-free weak-form coefficient fit, a data-only FNO inverse on two probe lattices, and an FNO-bracket hybrid with split-conformal intervals recover kappa = 1/(Re Sc) from sparse noisy probes, against the frozen v0.6 bars plus a constant-predictor gate?

## Result: FAIL

| bar | value | pass |
| --- | --- | --- |
| in-dist median rel | 0.9443049046645678 | fail |
| OOD median rel | 4.289229307568904 | fail |
| OOD / in-dist | 4.542208016056537 | fail |
| interval coverage | 0.8333333333333334 | fail |
| hybrid OOD below both baselines | fail | fail |
| noise 0.05 not worse than WSINDy | pass | pass |
| below constant predictor | fail | fail |

## Methods by noise

| noise | slice | method | median rel nu | median rel kappa | combined | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | in_dist | wsindy | 0.1657202507074217 | 0.1633859623700059 | 0.1645531065387139 | — |
| 0.00 | in_dist | fno_data | 0.6273401921583002 | 0.8331836232537325 | 0.7302619077060163 | — |
| 0.00 | in_dist | hybrid | 0.1781858159589718 | 0.149422270427809 | 0.1638040431933904 | 0.8333333333333334 |
| 0.00 | in_dist | oracle | 0.001168087597576003 | 0.0004897098401290064 | 0.0008288987188525045 | — |
| 0.00 | in_dist | constant | 0 | 0.6999999999999998 | 0.3499999999999999 | — |
| 0.00 | ood | wsindy | 0.06690570254922985 | 7.406998116023559 | 3.736951909286394 | — |
| 0.00 | ood | fno_data | 0.5091209915026333 | 6.757230187129776 | 3.633175589316205 | — |
| 0.00 | ood | hybrid | 0.06772363619812653 | 1.767349413377676 | 0.9175365247879013 | 0.5 |
| 0.00 | ood | oracle | 0.0005047211072245857 | 0.002101997470626264 | 0.001303359288925425 | — |
| 0.00 | ood | constant | 0 | 5.999999999999999 | 3 | — |
| 0.00 | ood_re | wsindy | 2.846714883084233 | 73.76298319639656 | 38.3048490397404 | — |
| 0.00 | ood_re | fno_data | 0.936197648195187 | 13.77611575533316 | 7.356156701764172 | — |
| 0.00 | ood_re | hybrid | 0.7419599159665784 | 7.569372838191008 | 4.155666377078792 | 0.125 |
| 0.00 | ood_re | oracle | 0.02549815686791706 | 0.08777455380914473 | 0.0566363553385309 | — |
| 0.00 | ood_re | constant | 4 | 16.5 | 10.25 | — |
| 0.01 | in_dist | wsindy | 6.381361591493661 | 2.727739317775508 | 4.554550454634584 | — |
| 0.01 | in_dist | fno_data | 0.6272378934649778 | 0.833628857444493 | 0.7304333754547353 | — |
| 0.01 | in_dist | hybrid | 0.9810476746438175 | 0.9075621346853182 | 0.9443049046645678 | 0.8333333333333334 |
| 0.01 | in_dist | oracle | 0.001168087597576003 | 0.0004897098401290064 | 0.0008288987188525045 | — |
| 0.01 | in_dist | constant | 0 | 0.6999999999999998 | 0.3499999999999999 | — |
| 0.01 | ood | wsindy | 21.57300042004834 | 19.52807671195871 | 20.55053856600353 | — |
| 0.01 | ood | fno_data | 0.5089622396488847 | 6.757076730843131 | 3.633019485246008 | — |
| 0.01 | ood | hybrid | 1.830079568238538 | 6.74837904689927 | 4.289229307568904 | 0.5 |
| 0.01 | ood | oracle | 0.0005047211072245857 | 0.002101997470626264 | 0.001303359288925425 | — |
| 0.01 | ood | constant | 0 | 5.999999999999999 | 3 | — |
| 0.01 | ood_re | wsindy | 86.45507732271753 | 78.00473957405148 | 82.22990844838451 | — |
| 0.01 | ood_re | fno_data | 0.9364317139699677 | 13.77990324158677 | 7.358167477778368 | — |
| 0.01 | ood_re | hybrid | 3.808050161982256 | 8.673781152939611 | 6.240915657460933 | 0.125 |
| 0.01 | ood_re | oracle | 0.02549815686791706 | 0.08777455380914473 | 0.0566363553385309 | — |
| 0.01 | ood_re | constant | 4 | 16.5 | 10.25 | — |
| 0.05 | in_dist | wsindy | 32.58199698304704 | 12.61904807641437 | 22.60052252973071 | — |
| 0.05 | in_dist | fno_data | 0.6268363921464273 | 0.835411352415558 | 0.7311238722809925 | — |
| 0.05 | in_dist | hybrid | 0.9832521244955792 | 1.420854114674755 | 1.202053119585167 | 0.8333333333333334 |
| 0.05 | in_dist | oracle | 0.001168087597576003 | 0.0004897098401290064 | 0.0008288987188525045 | — |
| 0.05 | in_dist | constant | 0 | 0.6999999999999998 | 0.3499999999999999 | — |
| 0.05 | ood | wsindy | 111.557945207764 | 74.12970084427637 | 92.8438230260202 | — |
| 0.05 | ood | fno_data | 0.5083401533660855 | 6.756491604056208 | 3.632415878711147 | — |
| 0.05 | ood | hybrid | 1.84061867234529 | 6.109726273853751 | 3.975172473099521 | 0.5 |
| 0.05 | ood | oracle | 0.0005047211072245857 | 0.002101997470626264 | 0.001303359288925425 | — |
| 0.05 | ood | constant | 0 | 5.999999999999999 | 3 | — |
| 0.05 | ood_re | wsindy | 447.8582192216986 | 255.2498375830698 | 351.5540284023842 | — |
| 0.05 | ood_re | fno_data | 0.9373180274213191 | 13.79886765849941 | 7.368092842960365 | — |
| 0.05 | ood_re | hybrid | 4.393371185310171 | 16.30306173244822 | 10.3482164588792 | 0.125 |
| 0.05 | ood_re | oracle | 0.02549815686791706 | 0.08777455380914473 | 0.0566363553385309 | — |
| 0.05 | ood_re | constant | 4 | 16.5 | 10.25 | — |

## What is not claimed

- a result on Petrobras 3W
- 3D
- the full ensemble
- that a Dedalus solver was run
- that PySR was scored
- identifiability at Reynolds 50000 or higher
- a retuned bar after seeing test scores
- a change to any v06, v052, v051, v05, v041, or v04 number

## Attribution

The Well `shear_flow` is CC BY 4 (Ohana et al.). The generator is Dedalus (Burns et al.). [dataset page](https://polymathic-ai.org/the_well/datasets/shear_flow/).

## Reproduce

```
python -m pinnforge.v061 download
python -m pinnforge.v061 pack
python -m pinnforge.v061 score --checkpoint artifacts/v061_pilot/score_cases.jsonl
python -m pinnforge.v061 report
```

CI runs `python -m pinnforge.v061 smoke` only. It never downloads The Well.

