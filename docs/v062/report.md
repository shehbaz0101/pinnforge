# v062 sim-train / The Well-test inverse

Research note. The contract is [protocol.json](protocol.json). Its SHA-256 is stored as `protocol_sha256` in [scores.json](scores.json). Solver validation is [solver_validation.md](solver_validation.md). No v061 or earlier artifact was edited. The dataset is The Well `shear_flow` (CC BY 4).

## Question

If a data-only FNO and a solver-in-the-loop search are trained on continuously sampled Sc from a The Well-matched shear-flow solver, can they recover kappa on official The Well files at Reynolds 10000 against the frozen v0.6 bars plus a constant-predictor gate?

## Result: FAIL

| bar | value | pass |
| --- | --- | --- |
| in-dist median rel | 1.250833866293762 | fail |
| OOD median rel | 9.315464060219249 | fail |
| OOD / in-dist | 7.447403137413522 | fail |
| interval coverage | 0.875 | fail |
| hybrid OOD below both baselines | fail | fail |
| noise 0.05 not worse than WSINDy | pass | pass |
| below constant predictor | fail | fail |

## Methods by noise

| noise | slice | method | median rel nu | median rel kappa | combined | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 | in_dist | wsindy | 0.1657202507074217 | 0.1633859623700059 | 0.1645531065387139 | — |
| 0.00 | in_dist | fno_data | 0.4554716681363045 | 0.9281804247041334 | 0.6918260464202189 | — |
| 0.00 | in_dist | hybrid | 0.290701322238535 | 1.448537593424907 | 0.8696194578317212 | 0.875 |
| 0.00 | in_dist | sitl_cold | 0 | 2.222037413350063 | 1.111018706675032 | — |
| 0.00 | in_dist | oracle | 0.001168087597576003 | 0.0004897098401290064 | 0.0008288987188525045 | — |
| 0.00 | in_dist | constant | 0 | 0.6484501610860745 | 0.3242250805430372 | — |
| 0.00 | ood | wsindy | 0.05894078232750418 | 7.406998116023559 | 3.732969449175531 | — |
| 0.00 | ood | fno_data | 0.4811157586553727 | 8.131905026506148 | 4.30651039258076 | — |
| 0.00 | ood | hybrid | 0.06398672765615997 | 17.41164236200036 | 8.737814544828261 | 0.1666666666666667 |
| 0.00 | ood | sitl_cold | 0 | 17.41164236200036 | 8.70582118100018 | — |
| 0.00 | ood | oracle | 0.0004336398492150079 | 0.001937963424801612 | 0.00118580163700831 | — |
| 0.00 | ood | constant | 0 | 3.687331185519006 | 1.843665592759503 | — |
| 0.01 | in_dist | wsindy | 7.105428935245492 | 3.538361543525133 | 5.321895239385313 | — |
| 0.01 | in_dist | fno_data | 0.4551537878331923 | 0.9280363318188112 | 0.6915950598260017 | — |
| 0.01 | in_dist | hybrid | 1.053219830322296 | 1.448447902265228 | 1.250833866293762 | 0.875 |
| 0.01 | in_dist | sitl_cold | 0 | 2.222037413350063 | 1.111018706675032 | — |
| 0.01 | in_dist | oracle | 0.001168087597576003 | 0.0004897098401290064 | 0.0008288987188525045 | — |
| 0.01 | in_dist | constant | 0 | 0.6484501610860745 | 0.3242250805430372 | — |
| 0.01 | ood | wsindy | 20.19508736635802 | 25.38494166527127 | 22.79001451581464 | — |
| 0.01 | ood | fno_data | 0.4811032548413081 | 8.137707429311332 | 4.30940534207632 | — |
| 0.01 | ood | hybrid | 1.219285758438137 | 17.41164236200036 | 9.315464060219249 | 0.1666666666666667 |
| 0.01 | ood | sitl_cold | 0 | 17.41164236200036 | 8.70582118100018 | — |
| 0.01 | ood | oracle | 0.0004336398492150079 | 0.001937963424801612 | 0.00118580163700831 | — |
| 0.01 | ood | constant | 0 | 3.687331185519006 | 1.843665592759503 | — |
| 0.05 | in_dist | wsindy | 33.88598712285669 | 16.31714903125925 | 25.10156807705797 | — |
| 0.05 | in_dist | fno_data | 0.4538833998647305 | 0.9274337004646103 | 0.6906585501646703 | — |
| 0.05 | in_dist | hybrid | 1.315342405222659 | 0.8333124829176903 | 1.074327444070175 | 0.875 |
| 0.05 | in_dist | sitl_cold | 0 | 0.8333124829176903 | 0.4166562414588452 | — |
| 0.05 | in_dist | oracle | 0.001168087597576003 | 0.0004897098401290064 | 0.0008288987188525045 | — |
| 0.05 | in_dist | constant | 0 | 0.6484501610860745 | 0.3242250805430372 | — |
| 0.05 | ood | wsindy | 100.9537557773814 | 124.1639546225511 | 112.5588551999662 | — |
| 0.05 | ood | fno_data | 0.4810669428050882 | 8.160617015451086 | 4.320841979128087 | — |
| 0.05 | ood | hybrid | 1.453975708388261 | 11.98634549462746 | 6.720160601507859 | 0.1666666666666667 |
| 0.05 | ood | sitl_cold | 0 | 14.74877938009512 | 7.374389690047559 | — |
| 0.05 | ood | oracle | 0.0004336398492150079 | 0.001937963424801612 | 0.00118580163700831 | — |
| 0.05 | ood | constant | 0 | 3.687331185519006 | 1.843665592759503 | — |

## What is not claimed

- a result on Petrobras 3W
- 3D
- the full ensemble
- that Dedalus was run in this study
- that PySR was scored
- identifiability at Reynolds 50000 or higher
- a retuned bar after seeing test scores
- a change to any v061 or earlier number

## Attribution

The Well `shear_flow` is CC BY 4 (Ohana et al.). The generator family is Dedalus (Burns et al.). [dataset page](https://polymathic-ai.org/the_well/datasets/shear_flow/).

## Reproduce

```
python -m pinnforge.v062 generate
python -m pinnforge.v062 pack-well
python -m pinnforge.v062 score --checkpoint artifacts/v062_pilot/score_cases.jsonl
python -m pinnforge.v062 report
```

CI runs `python -m pinnforge.v062 smoke` only.

