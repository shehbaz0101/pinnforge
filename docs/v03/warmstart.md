# Warm-started solver fit with calibrated intervals

Research note. It does not change any Stage D, E, F, or strong-baseline number. The contract is `docs/v03/warmstart_protocol.json`; it was committed before the test split was scored, and its SHA-256 is stored as `protocol_sha256` in `docs/v03/warmstart_scores.json`. Calibration and every hyperparameter come from `docs/v03/warmstart_validation.json`, which reads only the validation split.

## Question

Can an FNO first guess followed by a short solver-in-the-loop refinement reach the accuracy of the cold solver search on the frozen `hard_ood` slice at a fraction of its wall clock, with intervals whose empirical coverage is near nominal?

## Answer

The primary method (FNO hybrid first guess, then at most 2 solver evaluations, mean over 5 seeds) passes every preregistered bar.

| bar | measured | limit | result |
|---|---|---|---|
| `hard_ood` mean relative error | 0.01347636436363094 | <= 0.02069497348624076 | pass |
| `hard_ood` mean wall clock / cold mean wall clock, same run | 0.1314027993853286 | <= 0.25 | pass |
| full-test coverage of the nominal 0.9 interval | 0.8828125 | [0.85, 0.95] | pass |

Cold search re-timed in this run: `hard_ood` mean 15.65172069286673 s per instance and mean relative error 0.01379664899082717. Warm start mean 2.056679914239964 s. The re-run cold estimates are identical to the committed strong-baseline estimates on 128 of 128 test instances (largest absolute difference 0).

## Headline table on `hard_ood`

Seed-dependent rows are mean ± sample standard deviation over seeds. The cold search and the ensembles are one run each. Seconds include the FNO first guess and every refinement solve; model loading is excluded. Coverage is for the nominal interval. Abstain rate and accepted error use the validation threshold.

| method | mean rel error | failures | mean seconds | time / cold | coverage | abstain rate | error on accepted |
|---|---|---|---|---|---|---|---|
| solver, sensor-lift initial condition, cold search | 0.01379664899082717 | 0 | 15.65172069286673 | 1 | n/a | n/a | n/a |
| warm start, FNO hybrid loss first guess | 0.01347636436363094 ± 0.0002398017770186549 | 0 ± 0 | 2.056679914239964 ± 0.01109735325538703 | 0.1314027993853286 ± 0.0007090180992333099 | 0.9666666666666666 ± 1.241267076623637e-16 | 0.2666666666666667 ± 0 | 0.008106074353132467 ± 8.744751719865737e-05 |
| warm start, FNO data loss first guess | 0.01334164088084249 ± 0.00031308302245126 | 0 ± 0 | 2.057493486466713 ± 0.008409529817275684 | 0.1314547791160378 ± 0.0005372910737608741 | 0.9733333333333334 ± 0.01490711984999859 | 0.2666666666666667 ± 0 | 0.008093703156615742 ± 0.0001106992996267863 |
| warm start, median of the FNO hybrid loss seeds | 0.01339105657829486 | 0 | 3.283992072699766 | 0.2098166800405815 | 0.9666666666666667 | 0.2666666666666667 | 0.008092830296365731 |
| warm start, median of the FNO data loss seeds | 0.01372543778550932 | 0 | 3.27300956846672 | 0.2091149997302465 | 0.9666666666666667 | 0.2666666666666667 | 0.008175909921445047 |
| FNO first guess alone, FNO hybrid loss | 0.04205561399711315 ± 0.007390582608551378 | n/a | 0.3023443271533203 ± 0.001843031320906062 | n/a | n/a | n/a | n/a |
| FNO first guess alone, FNO data loss | 0.05470118594871485 ± 0.01084283098542982 | n/a | 0.3019869606133215 ± 0.004011391705517764 | n/a | n/a | n/a | n/a |

Published context from `docs/research/strong_baselines_scores.json`: cold sensor-lift solver 0.01379664899082717 at 18.72235205688281 s per instance, oracle initial condition 0.002512684392086404, FNO hybrid grid inverse 0.03368266375533967, FNO data grid inverse 0.04948720846902754. The first-guess rows above use a coarser grid than the published FNO inverse, so they are not the same estimator.

## Coverage and abstention by slice

| method | slice | n | mean rel error | coverage | mean log half-width | abstain rate | error on accepted | error on abstained |
|---|---|---|---|---|---|---|---|---|
| warm, FNO hybrid loss | `full_test` | 128 | 0.006744334396181523 ± 6.960039420331425e-05 | 0.8828125 ± 0 | 0.0148530422261387 ± 6.042779937616647e-05 | 0.08749999999999999 ± 0.003493856214843421 | 0.005107445137018699 ± 3.062386395507773e-05 | 0.0238144511133583 ± 0.00051479005708821 |
| warm, FNO hybrid loss | `hard_ood` | 30 | 0.01347636436363094 ± 0.0002398017770186549 | 0.9666666666666666 ± 1.241267076623637e-16 | 0.03544444577279494 ± 0.0003816467687548248 | 0.2666666666666667 ± 0 | 0.008106074353132467 ± 8.744751719865737e-05 | 0.02824466189250174 ± 0.0009371248075892856 |
| warm, FNO hybrid loss | `complement` | 98 | 0.004683508895941904 ± 3.651537173074321e-05 | 0.8571428571428571 ± 0 | 0.008549551344509238 ± 4.02673852212304e-05 | 0.0326530612244898 ± 0.004563404035713856 | 0.004411536616629563 ± 2.250625617102393e-05 | 0.01273613582220349 ± 3.170198598661011e-05 |
| warm, FNO data loss | `full_test` | 128 | 0.006733437572995568 ± 5.375081227923653e-05 | 0.884375 ± 0.003493856214843421 | 0.01493368735929763 ± 0.0002068790271173901 | 0.08749999999999999 ± 0.003493856214843421 | 0.005125459449458144 ± 3.442563128119225e-05 | 0.02350953167813793 ± 0.0007270582865657011 |
| warm, FNO data loss | `hard_ood` | 30 | 0.01334164088084249 ± 0.00031308302245126 | 0.9733333333333334 ± 0.01490711984999859 | 0.03573954020965507 ± 0.0008256656280661083 | 0.2666666666666667 ± 0 | 0.008093703156615742 ± 0.0001106992996267863 | 0.02777346962246604 ± 0.0009883066338078241 |
| warm, FNO data loss | `complement` | 98 | 0.004710518193042428 ± 4.502654993208009e-05 | 0.8571428571428571 ± 0 | 0.008564548731637183 ± 1.990231369202104e-05 | 0.0326530612244898 ± 0.004563404035713856 | 0.004436555988159082 ± 5.202603138540347e-05 | 0.01281998476220072 ± 5.786677943098918e-05 |
| warm, median of FNO hybrid loss seeds | `full_test` | 128 | 0.006712986282422067 | 0.8828125 | 0.01485112562052119 | 0.0859375 | 0.005106042209675446 | 0.02380502778345431 |
| warm, median of FNO hybrid loss seeds | `hard_ood` | 30 | 0.01339105657829486 | 0.9666666666666667 | 0.03541584473146332 | 0.2666666666666667 | 0.008092830296365731 | 0.02796117885359996 |
| warm, median of FNO hybrid loss seeds | `complement` | 98 | 0.004668679048991621 | 0.8571428571428571 | 0.008555803443702161 | 0.03061224489795918 | 0.004414364968547169 | 0.01272195826306593 |
| warm, median of FNO data loss seeds | `full_test` | 128 | 0.006830004069758287 | 0.8828125 | 0.01472613898676599 | 0.0859375 | 0.005161078077661414 | 0.02458130780387958 |
| warm, median of FNO data loss seeds | `hard_ood` | 30 | 0.01372543778550932 | 0.9666666666666667 | 0.03492429482189748 | 0.2666666666666667 | 0.008175909921445047 | 0.02898663941168608 |
| warm, median of FNO data loss seeds | `complement` | 98 | 0.004719157013916134 | 0.8571428571428571 | 0.008543030057644112 | 0.03061224489795918 | 0.004462906492785203 | 0.0128337568497289 |

## Paired comparison with the cold search

Same instances, same worker, same run. Delta is warm minus cold relative error; negative means the warm start has lower error. Counts are instances per seed, averaged over seeds.

| method | slice | mean delta | warm lower | warm higher | tie | time ratio of means | max instance time ratio | mean abs log gap to cold estimate |
|---|---|---|---|---|---|---|---|---|
| warm, FNO hybrid loss | `full_test` | -6.825056490967105e-05 ± 6.960039420331471e-05 | 68.2 ± 3.03315017762062 | 59.8 ± 3.03315017762062 | 0 ± 0 | 0.1312151096281456 ± 0.0001468479625189112 | 0.1528970651873602 ± 0.006142150539386128 | 0.0009736011245989626 ± 3.834331818233634e-05 |
| warm, FNO hybrid loss | `hard_ood` | -0.0003202846271962301 ± 0.0002398017770186551 | 15 ± 1.224744871391589 | 15 ± 1.224744871391589 | 0 ± 0 | 0.1314027993853286 ± 0.0007090180992333099 | 0.1440011213481037 ± 0.003849922726232089 | 0.00222744541094394 ± 0.0001316035686989135 |
| warm, FNO hybrid loss | `complement` | 8.902719463765386e-06 ± 3.651537173074333e-05 | 53.2 ± 2.387467277262665 | 44.8 ± 2.387467277262665 | 0 ± 0 | 0.1311576196192404 ± 0.0002521704395777525 | 0.1522822419698932 ± 0.006913805965223238 | 0.0005897712410239694 ± 1.540470003305447e-05 |
| warm, FNO data loss | `full_test` | -7.91473880956262e-05 ± 5.375081227923661e-05 | 66.59999999999999 ± 1.949358868961793 | 61.4 ± 1.949358868961793 | 0 ± 0 | 0.1312425010780903 ± 6.070608766277355e-05 | 0.1495724635939536 ± 0.005303905474395551 | 0.001014508862163384 ± 6.771764312768793e-05 |
| warm, FNO data loss | `hard_ood` | -0.0004550081099846839 ± 0.0003130830224512601 | 15 ± 1.224744871391589 | 15 ± 1.224744871391589 | 0 ± 0 | 0.1314547791160378 ± 0.0005372910737608741 | 0.1463311833009679 ± 0.00556021262652485 | 0.00238934789633516 ± 0.0002410058594146871 |
| warm, FNO data loss | `complement` | 3.591201656428945e-05 ± 4.502654993208002e-05 | 51.6 ± 2.509980079602227 | 46.4 ± 2.509980079602227 | 0 ± 0 | 0.1311774795954311 ± 0.0001126357059457949 | 0.1475686550341382 ± 0.005686965229359951 | 0.0005936397700699828 ± 1.815604029049854e-05 |

Cold search by slice in this run:

| slice | n | mean rel error | failures | mean seconds | mean solver evaluations |
|---|---|---|---|---|---|
| `full_test` | 128 | 0.006812584961091193 | 0 | 15.64464182871883 | 17.84375 |
| `hard_ood` | 30 | 0.01379664899082717 | 0 | 15.65172069286673 | 17.73333333333333 |
| `complement` | 98 | 0.004674606176478139 | 0 | 15.64247482948988 | 17.87755102040816 |

## Secondary: Stage E observation noise on `hard_ood`

Additive Gaussian noise with the Stage E seeds, scaled by the clean-field standard deviation. Calibration and abstain thresholds are the clean validation ones; nothing is refit for noise. The cold search is re-run on the same noisy fields.

| noise fraction | cold mean rel error | published cold | warm, FNO hybrid | warm, FNO data | warm time / cold | coverage, FNO hybrid | abstain rate, FNO hybrid |
|---|---|---|---|---|---|---|---|
| 0 | 0.01379664899082717 | 0.01379664899082717 | 0.01347636436363094 ± 0.0002398017770186549 | 0.01334164088084249 ± 0.00031308302245126 | 0.1314027993853286 ± 0.0007090180992333099 | 0.9666666666666666 ± 1.241267076623637e-16 | 0.2666666666666667 ± 0 |
| 0.01 | 0.01442659423227493 | n/a | 0.01275244352837155 ± 0.0004203603431534397 | 0.01261870570593494 ± 7.417917031598084e-05 | 0.1233904709658874 ± 0.0003536885946399709 | 1 ± 0 | 0.4 ± 0 |
| 0.02 | 0.01420591117118668 | 0.01420591117118668 | 0.01819891780744707 ± 0.003188372089190953 | 0.01555069064325268 ± 0.001934804408757096 | 0.1233509488551651 ± 0.000455881508918364 | 1 ± 0 | 0.9066666666666666 ± 0.02788866755113585 |
| 0.05 | 0.01822853475529066 | 0.01822853475529066 | 0.149360067939267 ± 0.04085717042607805 | 0.1243926982190979 ± 0.05614338618686938 | 0.1263371461306297 ± 0.0004220148718303889 | 0.8066666666666666 ± 0.02788866755113585 | 1 ± 0 |

The FNO hybrid warm start has higher `hard_ood` mean relative error than the cold search on the same noisy fields at noise fractions 0.02 and 0.05; the time ratio stays near its clean value because the solve count is capped. Its interval coverage falls below 0.85 at noise fraction 0.05 (abstain rate 1).

## Secondary: Stage F restricted-range operators

First guesses come from operators trained only on `nu > threshold_nu`, so `hard_ood` is outside their training range. Intervals and abstain thresholds are calibrated on validation instances above the threshold only. The refinement and the cold search are unchanged.

| method | slice | first guess mean rel error | warm mean rel error | failures | time / cold | coverage | abstain rate | error on accepted |
|---|---|---|---|---|---|---|---|---|
| warm, restricted FNO data loss | `hard_ood` | 0.2369619726633709 ± 0.03206243710730067 | 0.02760850438664737 ± 0.00406103734570976 | 0 ± 0 | 0.1315169975991654 ± 0.0007157226324120327 | 0.6333333333333333 ± 0.02357022603955158 | 0.6933333333333332 ± 0.05477225575051664 | 0.004438587551571537 ± 0.0002766328581215442 |
| warm, restricted FNO data loss | `complement` | 0.02997335624640215 ± 0.002282969021705749 | 0.004691987009926278 ± 3.89061308220215e-05 | 0 ± 0 | 0.1310554402487793 ± 0.0003293860195339026 | 0.8367346938775511 ± 0 | 0.1102040816326531 ± 0.004563404035713856 | 0.004369347029342355 ± 3.51422668718807e-05 |
| warm, restricted FNO hybrid loss | `hard_ood` | 0.2172758798964253 ± 0.02832158177806241 | 0.02560100457562003 ± 0.004656779011521679 | 0 ± 0 | 0.1319461764580656 ± 0.0007605405279342839 | 0.6266666666666667 ± 0.02788866755113584 | 0.5999999999999999 ± 0.05270462766947298 | 0.005643812901178432 ± 0.0008899443573611682 |
| warm, restricted FNO hybrid loss | `complement` | 0.02924030729278533 ± 0.001213744955860869 | 0.004705797669201777 ± 4.710257187074702e-05 | 0 ± 0 | 0.1310859524659902 ± 0.0003366319679464716 | 0.8367346938775511 ± 0 | 0.1081632653061225 ± 0.005589005688828225 | 0.004366780829656847 ± 2.8807184037891e-05 |
| warm, median of restricted FNO data loss seeds | `hard_ood` | 0.2415316295527753 | 0.0252077121550805 | 0 | 0.2100652967396442 | 0.7 | 0.7 | 0.004079300077381573 |
| warm, median of restricted FNO data loss seeds | `complement` | 0.0272006162502925 | 0.004670109393307312 | 0 | 0.2089371836626738 | 0.8367346938775511 | 0.1122448979591837 | 0.004359174207268847 |
| warm, median of restricted FNO hybrid loss seeds | `hard_ood` | 0.2198544844865819 | 0.02316104347675894 | 0 | 0.2087689787949416 | 0.6666666666666666 | 0.6 | 0.00511107544805249 |
| warm, median of restricted FNO hybrid loss seeds | `complement` | 0.02703416972942706 | 0.004682547283129617 | 0 | 0.2086658836843751 | 0.8367346938775511 | 0.1122448979591837 | 0.004358498195747258 |

Published Stage F restricted FNO grid inverse on the out-of-range slice: data loss 0.2331097847663486, hybrid loss 0.2127781247271391. Cold search on the same `hard_ood` instances: 0.01379664899082717.

With the restricted FNO data loss first guess, the `hard_ood` warm-start error is above the error bar limit and its coverage is below 0.85. With the restricted FNO hybrid loss first guess, the `hard_ood` warm-start error is above the error bar limit and its coverage is below 0.85. These rows are secondary and were not held to the bars.

## Validation-only choices

Solver-evaluation cap chosen on validation `hard_ood` for the primary method: 2. Validation cold mean wall clock 16.27786641370363 s.

| cap | validation mean rel error | validation time / cold |
|---|---|---|
| 2 | 0.01208286303516941 | 0.1158376315341465 |
| 3 | 0.01200866804475037 | 0.1682238449928145 |
| 4 | 0.01200891201019723 | 0.1785819452119398 |
| 5 | 0.01200711300295837 | 0.1793841218996646 |
| 6 | 0.01200711300295837 | 0.1793841218996646 |

| family | method kind | interval scale | abstain statistic |
|---|---|---|---|
| full | seed | `gauss_newton` | `log_standard_error` |
| full | ensemble | `gauss_newton` | `log_standard_error` |
| restricted | seed | `gauss_newton` | `log_standard_error` |
| restricted | ensemble | `gauss_newton` | `log_standard_error` |

## What is NOT claimed

- a retuned hard_ood threshold, sensor mask, solver search, or Stage A through F protocol
- a change to any previously reported number
- that the warm start is more accurate than the cold search; the bar is accuracy within a factor of the cold search
- speed-ups on other machines, other worker counts, GPUs, or other solvers
- coverage conditional on viscosity, on noise level, or outside the training range
- that the abstain flag detects every large error
- any result for viscosity ranges, equations, or observation masks outside this pilot

## Limitations

- One scalar viscosity, one-dimensional periodic Burgers, and the Stage A pilot. This is not a multi-equation benchmark.
- The warm start minimizes the same sensor-lift objective as the cold search. It cannot be more accurate than that objective allows except by stopping at a different point; the oracle initial condition is the ceiling for both.
- Wall clock is one CPU machine with the protocol worker count. Ratios, not absolute seconds, are the object of the time bar.
- The networks were retrained for this study with the frozen commands. They are not bitwise copies of the published Stage B, C, and F networks.
- Validation timing ran while other jobs shared the machine. Only the cap choice read validation time.
- Coverage is marginal over instances and calibrated on one validation split of the same pilot. It is not conditional coverage for a given viscosity.

## Reproduce

Checkpoints are retrained with the frozen Stage B, C, and F commands listed in the protocol; they are not in git. Their SHA-256 values are in the protocol. Then:

```bash
python -m pinnforge.operator warmstart validate
python -m pinnforge.operator warmstart test
python -m pinnforge.operator warmstart report
python -m pinnforge.claims check
```

Python, NumPy, and torch versions are stored as `python`, `numpy`, and `torch` in the scores JSON.
