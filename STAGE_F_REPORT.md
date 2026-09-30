# Stage F — out-of-distribution viscosity

Restricted-range 1D Fourier neural operators on the Stage A harder Burgers pilot. Training and validation use only instances with `ν` above the frozen `hard_ood` threshold. The test split is then sliced into in-range, out-of-distribution (`ν` at or below that threshold), and `ν < 0.02`. Five seeds, two arms: data-only and the Stage C hybrid weight `1e-2`. The version string stays `0.1.0`. This run used Python 3.12.3, NumPy 2.4.4, and the CPU torch wheel 2.14.1+cpu.

The question was frozen before any test score from this stage. Stages B–D trained on the full harder-pilot range `[0.005, 0.10]`, so `hard_ood` was inside the training support. This stage asks whether an operator trained only on the easier viscosities extrapolates to low `ν`, for one-step and rollout prediction and for the Stage D operator-conditioned viscosity inverse, and whether the physics-informed hybrid loss extrapolates better than data-only. The hybrid weight was not chosen again. A negative answer is a result.

The contract is [docs/v02/stage_f_ood_protocol.json](docs/v02/stage_f_ood_protocol.json). Its SHA-256 is `dc12223986d762050802c144c7c7770116084c11a14fba342e49e75d7a6c4b6f`. That file was committed before the test split was scored, and it was not edited afterward. `hard_ood` is still `ν <= 0.027028120493367818` from [docs/v02/pilot_protocol.json](docs/v02/pilot_protocol.json). That quartile was not refit. The published Stage B, C, and D numbers are copied into [docs/v02/stage_f_scores.json](docs/v02/stage_f_scores.json) by hash. They were not recomputed and not edited.

## Setup

| item | value |
| --- | --- |
| architecture | 1D FNO, width 32, 16 modes, 4 layers |
| window | 8 input frames, 8 output frames, stride 8 |
| optimizer | Adam, learning rate `0.001`, constant, 30 epochs, batch 32 |
| seeds | 0, 1, 2, 3, 4 |
| checkpoint | lowest validation mean relative L2 on the in-range validation subset. Ties would keep the earliest epoch |
| train subset | `ν > 0.027028120493367818`, 384 instances, 128 excluded |
| train `ν` | min `0.02704009846210967`, max `0.09988823841997412` |
| validation subset | same rule, 101 instances, 27 excluded. Instances at or below the cut are not loaded |
| normalization | population mean and standard deviation, `ddof = 0`, restricted training instances only |
| data-only | normalized data MSE |
| hybrid | data MSE + `1e-2` × mean of `R²`, the Stage C weight, not reselected |
| test slices | full test 128, in-range 98, OOD 30, `ν < 0.02` 20 (a subset of OOD) |
| inverse | `λ = 0` sensor MSE on `sensors32_bursts`, not reselected |
| `ν` search | `linspace(0.005, 0.1, 191)`, step `0.0005`, both endpoints included, smallest `ν` on a tie |
| search clip | not clipped to the training range. 45 grid points lie below `train_nu_min`, 1 lies above `train_nu_max` |

`u_mean = -6.93012725267756e-19`, `u_std = 0.3472358145478557`, `nu_mean = 0.0637698128512489`, `nu_std = 0.02106505306567045`. Validation, test, and the excluded low-viscosity training instances are not in those four numbers.

Mean fit time after the windows were loaded was about 466 s for data-only and about 680 s for hybrid. Those clocks are not a hardware benchmark.

## Commands

```bash
python -m pinnforge.operator train \
  --protocol docs/v02/stage_f_ood_protocol.json \
  --pilot artifacts/burgers_hard_pilot \
  --manifest docs/stage_a/pilot_manifest.json \
  --seed 0 \
  --output runs/stage_f/data_only/seed_0 \
  --loss data
```

Hybrid runs use `--loss hybrid --residual-weight 1e-2` and write `runs/stage_f/hybrid_1e-2/seed_<seed>`. Repeat for seeds 1–4. `runs/` is gitignored.

```bash
python -m pinnforge.operator ood-slices \
  --protocol docs/v02/stage_f_ood_protocol.json \
  --checkpoint runs/stage_f/data_only/seed_0/checkpoint.pt \
  --output runs/stage_f/data_only/seed_0/slices.json

python -m pinnforge.operator ood-inverse \
  --protocol docs/v02/stage_f_ood_protocol.json \
  --checkpoint runs/stage_f/data_only/seed_0/checkpoint.pt \
  --output runs/stage_f/data_only/seed_0/inverse.json

python -m pinnforge.operator ood-aggregate \
  --protocol docs/v02/stage_f_ood_protocol.json \
  --arm data_only \
  --inputs runs/stage_f/data_only/seed_{0,1,2,3,4}/slices.json \
  --output runs/stage_f/data_only_aggregate.json

python -m pinnforge.operator stage-f-scores \
  --protocol docs/v02/stage_f_ood_protocol.json \
  --data-only-forward runs/stage_f/data_only_aggregate.json \
  --hybrid-forward runs/stage_f/hybrid_aggregate.json \
  --inverse runs/stage_f/data_only/seed_{0,1,2,3,4}/inverse.json \
           runs/stage_f/hybrid_1e-2/seed_{0,1,2,3,4}/inverse.json \
  --output docs/v02/stage_f_scores.json
```

The same `ood-slices`, `ood-inverse`, and `ood-aggregate` commands with `--arm hybrid_1e-2` cover the second arm. `stage-f-scores` refuses to overwrite the protocol. It checks the SHA-256 of the published Stage B, C, and D files before it copies their means.

## Checkpoint selection

Validation only, in-range subset only. Mean ± sample standard deviation (`ddof = 1`) of the selected epoch and of the validation mean relative L2.

| arm | selected epochs | validation mean relative L2 |
| --- | --- | ---: |
| data-only | 30, 27, 29, 30, 27 | 0.003632184639037897 ± 0.0005296678324067348 |
| hybrid `1e-2` | 25, 29, 29, 26, 25 | 0.003087439909936613 ± 0.00020617841998121195 |

Selected-epoch means are 28.6 ± 1.51657508881031 and 26.8 ± 2.04939015319192. Epoch 30 was selected for two data-only seeds and for no hybrid seed. No seed was dropped. Rollout error and inverse error were not used to pick the checkpoint. The OOD slice was not used to pick it either.

## Forward prediction

Primary metric: mean per-window relative L2 on denormalized `u`. One scalar per seed, then the arithmetic mean and the sample standard deviation. Windows are not pooled across seeds. The full-range columns are the published Stage B data-only table and the published Stage C hybrid `1e-2` table. In those stages `hard_ood` was in distribution. Here the same 30 test instances are out of distribution. `in-range` in this table is the Stage B/C complement.

| slice | n | restricted data-only | restricted hybrid `1e-2` | full-range data-only | full-range hybrid `1e-2` | one-step persistence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full test | 128 | 0.011576874005482485 ± 0.0005267011050376607 | 0.010967086328074149 ± 0.00027688144710178835 | 0.004298970125314308 ± 0.0003608651258270467 | 0.00324308298751049 ± 0.0003669287417174703 | 0.08456701056846332 |
| in-range | 98 | 0.0035031846024788406 ± 0.0005104856137032302 | 0.002999427171440285 ± 0.00016621134666117572 | 0.0034570468687440416 ± 0.00038656370146409214 | 0.002639304012951843 ± 0.00030849661952913005 | 0.08050939647178759 |
| OOD | 30 | 0.03795092605529439 ± 0.0012229562365480783 | 0.036994772906411434 ± 0.0012438061139733567 | 0.007049252763443842 ± 0.0003032654479273341 | 0.005215427637735405 ± 0.0006736875927560861 | 0.09782188328427069 |
| `ν < 0.02` | 20 | 0.052887317691553845 ± 0.0016553652903734555 | 0.05191112595861432 ± 0.0015655595158504325 | not published | not published | 0.10557523367006487 |

The in-range slice barely moves. Restricted data-only is 1.013 times the Stage B complement mean. Restricted hybrid is 1.136 times the Stage C complement mean. Holding out the low viscosities does not spoil forecasts inside the training range.

The OOD slice does not extrapolate. Restricted data-only is 5.384 times the Stage B `hard_ood` mean. Restricted hybrid is 7.093 times the Stage C `hard_ood` mean. Stage B's in-distribution gap, `hard_ood` over complement, was 2.039. The true out-of-distribution gap is several times larger.

OOD divided by in-range, ratio of the two means:

| arm | one-step ratio of means | one-step ratio per seed | rollout ratio of means |
| --- | ---: | ---: | ---: |
| data-only | 10.833264689631386 | 10.996216251621068 ± 1.4441280752525938 | 7.117035537397621 |
| hybrid `1e-2` | 12.333946047653837 | 12.370522120458043 ± 0.9301832117799445 | 7.397994580440332 |

Per seed, the data-only one-step ratios are 11.207, 12.636, 8.816, 11.795, and 10.528. The hybrid ratios are 13.798, 11.672, 12.085, 11.534, and 12.763. Every seed of both arms has an OOD mean above its in-range mean. The smallest ratio in the ten runs is 8.816.

Hybrid minus data-only, paired by seed, on the OOD one-step mean:

`+0.0016577663119434968`, `-0.0037593027116601674`, `-0.0017397304088457924`, `+0.0009647198491254047`, `-0.0019042187849777278`.

Paired mean `-0.0009561531488829572`, paired sample standard deviation `0.002230067110320764`. Seeds 0 and 3 are higher for hybrid. The paired mean is smaller than the paired spread. The hybrid OOD gap is larger than the data-only gap because the in-range slice improved more than the OOD slice. The physics loss at the frozen weight `1e-2` does not extrapolate better than data-only. That is the negative result this stage was allowed to report.

Against one-step persistence, the OOD mean is 0.388 of the baseline for data-only and 0.378 for hybrid. Persistence is about 2.6 times larger. Stage B's in-distribution `hard_ood` mean was 0.072 of the same baseline. On `ν < 0.02` the restricted means are 0.501 and 0.492 of persistence. The network still beats persistence. The margin is much smaller than it was when those viscosities were inside the training range.

| slice | data-only median | data-only pooled | hybrid median | hybrid pooled |
| --- | ---: | ---: | ---: | ---: |
| OOD | 0.026779668485104168 ± 0.0013960356840425155 | 0.05617389134012172 ± 0.0015090742988427305 | 0.02581263806708594 ± 0.0014440337543729712 | 0.05489363701476663 ± 0.0014251519115037114 |
| `ν < 0.02` | 0.05070573760953677 ± 0.0025343209228215107 | 0.06564532573872793 ± 0.0017536424079178724 | 0.049930417087538724 ± 0.0013572441237650348 | 0.06420929365937653 ± 0.0016383943884832881 |

The median sits below the mean, and the pooled score sits above the mean. A few windows carry more of the squared error than a typical window. The same pattern is in the Stage C hybrid table.

### Rollout

Secondary. Not used for selection. Mean ± sample standard deviation of the per-instance relative L2 on frames 8 through 95, after feeding predictions back as inputs.

| slice | restricted data-only | restricted hybrid `1e-2` | full-range data-only | full-range hybrid `1e-2` | open-loop persistence |
| --- | ---: | ---: | ---: | ---: | ---: |
| full test | 0.02960523085218663 ± 0.0030642854400767987 | 0.02751131976383908 ± 0.0007484083678859277 | 0.015117106214026083 ± 0.001852906885827387 | 0.012333156149629956 ± 0.0019938922697132697 | 0.976635185504515 |
| in-range | 0.012164799139420143 ± 0.00301234296290178 | 0.01100659723479001 ± 0.0012710675242996003 | 0.01183115853201427 ± 0.001866801285523518 | 0.009666639862762874 ± 0.0020385656321904467 | 0.9918664859829069 |
| OOD | 0.08657730778055715 ± 0.004345595791179338 | 0.08142674669206604 ± 0.0042005288164289984 | 0.025851201975264682 ± 0.0018221690825946362 | 0.021043776020062428 ± 0.002621035045734292 | 0.9268796039417675 |
| `ν < 0.02` | 0.1162210054192416 ± 0.005206101455699562 | 0.10997394330446793 ± 0.0054055433357322455 | not published | not published | 0.9361300919531456 |

Restricted data-only rollout on OOD is 3.349 times the Stage B `hard_ood` rollout. Restricted hybrid rollout is 3.869 times the Stage C `hard_ood` rollout. Open-loop persistence holds one early frame across 0.88 time units, so a score near 1 is the trivial forecast. The rollout stays under that floor. The informative comparison is the full-range operator, and the restricted operators lose it.

Paired hybrid minus data-only on the OOD rollout mean: `-0.002587982588796278`, `-0.007712772574022073`, `-0.010643155550991781`, `+0.0040440602049183205`, `-0.008852954933563806`. Paired mean `-0.005150561088491124`, paired sample standard deviation `0.005948720684030061`. Seed 3 is higher for hybrid. Four seeds are lower. The paired mean is inside the paired spread.

The gap is already present on the first predicted block, which still sees the true input. Data-only step 0 is `0.06204814412459732` on OOD and `0.007767742526398884` on in-range. Hybrid step 0 is `0.05888098651320141` on OOD and `0.0061480384103606315` on in-range. The error then rises and levels off. It does not jump to the persistence floor. The step table is rounded. The unrounded curves are in the score file.

| step | `t` of the block | data-only OOD | hybrid OOD |
| ---: | --- | ---: | ---: |
| 0 | 0.08–0.15 | 6.205e-02 | 5.888e-02 |
| 1 | 0.16–0.23 | 8.630e-02 | 8.165e-02 |
| 2 | 0.24–0.31 | 9.312e-02 | 8.731e-02 |
| 5 | 0.48–0.55 | 9.296e-02 | 8.668e-02 |
| 10 | 0.88–0.95 | 8.720e-02 | 8.288e-02 |

### Where the forward error sits

The five worst OOD instances by one-step mean relative L2 are the same five ids on every seed of both arms: `179`, `338`, `619`, `530`, and `185`. The order of the first three is the same on every seed. `179` is first on every seed. It has the smallest test viscosity in the manifest, `ν = 0.00659586199923937`.

| instance | `ν` | data-only five-seed mean | hybrid five-seed mean |
| ---: | ---: | ---: | ---: |
| 179 | 0.00659586199923937 | 0.10074261038672197 ± 0.0027139659575281328 | 0.09888218099392557 |
| 338 | 0.007689406142113342 | 0.09060234228341174 ± 0.0020203478984204375 | 0.08940451819268234 |
| 619 | 0.007207675915965656 | 0.08883877059676921 ± 0.0019111040646515645 | 0.08756179757466598 |
| 530 | 0.007820793287511035 | 0.07809154508005446 ± 0.0017266938042388585 | 0.07700138462028257 |
| 185 | 0.006878265076856729 | 0.07736412340727068 ± 0.0019161654393986362 | 0.07608180477020139 |

Stage B, where these viscosities were in distribution, listed `179`, `619`, `338`, `185`, and `205` as the worst `hard_ood` ids, and quoted seed 0 on `179` at `0.013307020107418`. The Stage F five-seed mean on `179` is 7.57 times that seed-0 window. `530` replaces `205` in the worst five. The errors are about 0.08 to 0.10, not the 0.01 of the in-distribution run, and they still beat one-step persistence on that instance (`0.12241994740816874` in the Stage B table). The threshold was not moved.

## Inverse, `sensors32_bursts`

The mask is the Stage 5 pattern used in Stage D. 32 equispaced sensors, four bursts of five frames. It was not retuned. `λ = 0` is sensor mean square alone, the weight Stage D selected on validation. It was not selected again, and it was not selected on OOD data.

The search grid is the Stage D grid, `linspace(0.005, 0.1, 191)`. It is not clipped to `[0.02704009846210967, 0.09988823841997412]`. Forty-five grid points, from `0.005` through `0.027`, sit strictly below every training viscosity. One grid point, `0.1`, sits above `train_nu_max`. A minimizer can land outside the training range, and on this test split it does.

Every one of the 30 OOD instances, on every seed, of both arms, has `ν̂` below `train_nu_min`. The count is `30.00 ± 0.00`. The inverse is not stuck inside the training interval. It extrapolates the parameter, and it extrapolates low.

Mean relative viscosity error. The full-range columns are the published Stage D `λ = 0` table. The closed-form column is the Stage A `sensors32_bursts` least squares, copied from [docs/v02/inverse_stress.json](docs/v02/inverse_stress.json).

| slice | n | restricted data-only | restricted hybrid `1e-2` | full-range data-only | full-range hybrid `1e-2` | closed-form LS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full test | 128 | 0.06528957212499967 ± 0.008302079949584722 | 0.05880278932089433 ± 0.006565364875645695 | 0.021866330905899147 ± 0.008819091358584748 | 0.015381956640685085 ± 0.0027924678283891245 | 0.1171354466174288 |
| in-range | 98 | 0.013916037642954071 ± 0.004146691295472162 | 0.011667482563880647 ± 0.002542793382231917 | 0.01341096022330882 ± 0.007957033491083166 | 0.009779699360688785 ± 0.002898987551908216 | 0.04601593633597921 |
| OOD | 30 | 0.23310978476634858 ± 0.02949963302953843 | 0.21277812472713906 ± 0.028889898365513873 | 0.049487208469027544 ± 0.0144232271670265 | 0.03368266375533967 ± 0.0034734247381767768 | 0.3494591802034973 |
| `ν < 0.02` | 20 | 0.32400305722662354 ± 0.033977569784910615 | 0.3003392455495894 ± 0.03807874771902955 | 0.06339223281191389 ± 0.015643388668796922 | 0.044268482159261324 ± 0.004262785004741288 | 0.46966295632108307 |

In-range inverse error stays near the full-range complement: 0.0139 versus 0.0134 for data-only, 0.0117 versus 0.0098 for hybrid. Neither restricted arm has a `0.5`-rule failure on the in-range slice, on any seed. A few in-range estimates fall one or more grid steps below `train_nu_min` (2.6 ± 0.89 data-only, 1.6 ± 0.55 hybrid). Those instances sit just above the cut. They are not the OOD failure.

On OOD, restricted data-only is 4.710 times the Stage D data-only mean, and restricted hybrid is 6.317 times the Stage D hybrid mean. Stage D had zero `0.5`-rule failures on every slice. The restricted operators do not.

| arm | OOD failures per seed | OOD mean failure count | in-range failures |
| --- | --- | ---: | ---: |
| data-only | 3, 2, 2, 1, 1 | 1.8 ± 0.8366600265340756 | 0 |
| hybrid `1e-2` | 4, 1, 1, 2, 1 | 1.8 ± 1.3038404810405297 | 0 |

The closed form has 9 failures, all inside OOD and inside `ν < 0.02`. Both restricted arms beat that slice mean on every seed. Data-only minus the closed-form OOD mean, per seed: `-0.0829344458082229`, `-0.09666518129035526`, `-0.1115389116268383`, `-0.15707243604433943`, `-0.13353600241598834`. The closed-form mean is 1.499 times the restricted data-only mean and 1.642 times the restricted hybrid mean. Beating the closed form here is a smaller margin than Stage D, where the data-only mean was about 7.1 times smaller than the closed form and the hybrid mean about 10.4 times smaller. Neither arm approaches the dense residual least squares on OOD, which remains `0.0013208386837906374`: about 176 times smaller than restricted data-only and about 161 times smaller than restricted hybrid.

Hybrid minus data-only on the OOD inverse mean, paired by seed:

`-0.006030769016622306`, `-0.03906099208220523`, `-0.051928666109973265`, `+0.01665395834621064`, `-0.02129183133345741`.

Paired mean `-0.020331660039209516`, paired sample standard deviation `0.027030224063623712`. Seed 3 is worse for hybrid. The failure count does not drop. The hybrid loss does not repair the out-of-distribution inverse.

The one-step sensor curve, scored with true input windows rather than a rollout, is already in the same range. Oracle OOD mean relative error is `0.24310127746647342 ± 0.019443089632228502` for data-only and `0.25690672957688493 ± 0.019436143748460203` for hybrid. Feeding predictions back is not the main source of the viscosity error. The one-step map at low `ν` is.

### Failure examples

The smallest viscosities collapse to the bottom of the search grid. Data-only lands on `ν̂ = 0.005` for all five seeds on instances `179`, `185`, `338`, `530`, and `619`, and for four seeds on `655` and `743`. Hybrid does the same on `185`, `338`, `530`, `619`, and `655` for all five seeds, and on `179` for four. Their true viscosities are `0.0066` to `0.0088`, so the relative error of the grid floor is about 0.24 to 0.43. That undershoot stays under the `0.5` rule. It is still a boundary collapse: the argmin is the lowest value the search is allowed to return.

The `0.5`-rule failures are a different set, slightly higher in viscosity, where the same undershoot is a larger fraction of `ν`.

| instance | `ν` | arm | seeds failing | mean rel | `ν̂` across seeds | LS rel |
| ---: | ---: | --- | ---: | ---: | --- | ---: |
| 83 | 0.011835399149504187 | data-only | 4 | 0.5352923944072974 | 0.0055, 0.005, 0.005, 0.0055, 0.0065 | 0.5240345713792666 |
| 594 | 0.013600622209521913 | data-only | 2 | 0.4412020359863261 | 0.0065, 0.0075, 0.006, 0.0095, 0.0085 | 0.08746448239728127 |
| 391 | 0.010141552360801248 | hybrid | 3 | 0.45767671414305405 | 0.005, 0.0055, 0.007, 0.005, 0.005 | 0.5328612255755112 |
| 83 | 0.011835399149504187 | hybrid | 3 | 0.44235087328875683 | 0.0055, 0.005, 0.0055, 0.0075, 0.0095 | 0.5240345713792666 |
| 62 | 0.012683399819147168 | hybrid | 1 | 0.40867589865945736 | 0.006, 0.008, 0.008, 0.007, 0.0085 | 0.32940113457238773 |

Instance `594` is the clear loss to the closed form. Least squares is already accurate there (`0.087`), and both the data-only estimates above and the hybrid estimates sit between `0.006` and `0.0095`. Instance `83` fails for both the operator and the closed form. Stage D's full-range operators stayed under `0.5` on every one of the nine closed-form failures. These restricted operators do not.

## What is not claimed

- `hard_ood` was not refit. The Stage A, B, C, and D protocols were not retuned. No previously reported number was edited. The full-range columns above are copies.
- The hybrid weight stays `1e-2` from [docs/v02/stage_c_weight_selection.json](docs/v02/stage_c_weight_selection.json). It was not chosen on OOD data, on test data, or on this restricted training run. `λ` stays `0`. It was not chosen again.
- The viscosity search is not clipped to the restricted training range. Returning a value below `train_nu_min` is allowed. This stage does not claim that those values are accurate.
- Hybrid does not extrapolate better than data-only. The OOD one-step paired difference changes sign, and the OOD gap ratio is larger for hybrid. A lower mean on three of five forward seeds, or four of five inverse seeds, is not a win.
- Beating the closed form on this mask is not the dense ceiling, and it is not the Stage D result. Dense OOD mean relative error remains `0.0013208386837906374`. Stage D's zero `0.5`-rule failures were for operators trained on the full range.
- `hard_ood` in Stages B and C was not an out-of-distribution slice. The ratio 2.039 in Stage B is an in-distribution difficulty gap. The ratio 10.83 here is the extrapolation gap.
- No architecture, window, or learning-rate search. Sixteen Fourier modes do not cover the 48-mode initial band. This run does not measure how much of the OOD error that truncation causes.
- One seed is not the result. The tables are the five-seed mean and the sample standard deviation. No seed was dropped.
- Rollout error and inverse error were not used to pick the checkpoint. The OOD slice was not used to pick it.
- The version string is still `0.1.0`. This is not a `0.2.0` release.

## Tests

```bash
python -m ruff check .
python -m pytest -q --tb=line
```

`tests/test_stage_f.py` checks that the committed protocol keeps the frozen cut, the restricted train and validation counts, the Stage C weight, `λ = 0`, and an unclipped 191-point grid with 45 points below the training minimum. It checks that a protocol file which already holds scores is rejected. It checks that the committed score file still matches the protocol hash, still copies the Stage B `hard_ood` mean, the Stage C hybrid `hard_ood` mean, the Stage D data-only `hard_ood` inverse mean, and the 9 closed-form OOD failures, and that the restricted OOD forward and inverse means sit above those full-range means and below the closed form. The paired hybrid-minus-data OOD one-step values are required to change sign. Curve fitting is not imported by that file.
