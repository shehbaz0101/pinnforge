# Stage D report

Operator-conditioned sparse viscosity inverse on the Stage A harder Burgers pilot, on `feat/v02-stage-d-operator-inverse`, based on `ee3754b0894e085cc4b2cc9ba0d4d6b33060d8e6` (Stage C). The version string stays `0.1.0`. These runs used Python 3.12.3, NumPy 2.4.4, and the CPU torch wheel `2.14.0+cpu`.

The question was frozen before any test score. Stage A showed that the Stage 5 closed-form residual least squares on `sensors32_bursts` fails the `0.5` relative-error rule on 9 of 30 `hard_ood` test instances, while the dense residual least squares does not. Those failures line up with aliasing of `u_xx` on the 32-point sensor grid. This stage asks whether the trained Fourier neural operators from Stages B and C can recover `ν` from that same mask more accurately than the closed-form estimator, especially on `hard_ood`.

They can, on this pilot and these five seeds. A negative result would have been reported the same way. The dense stencil is still tighter than either operator.

## Contract

[docs/v02/stage_d_inverse_protocol.json](docs/v02/stage_d_inverse_protocol.json) was committed before the validation curves were reduced and before any test curve was written. It is not a score table.

- Mask: `sensors32_bursts`, the Stage 5 pattern. 32 equispaced sensors on `[-1, 1)`, four bursts of five frames starting at frames 0, 24, 48, and 72, `Δt = 0.01`. It was not retuned.
- Failure rule: `ν̂ <= 0` or relative error `> 0.5`.
- `hard_ood`: `ν <= 0.027028120493367818`, the Stage A training quartile. Not refit. The complement is `ν` above that cut. `below_stage2_floor` is `ν < 0.02`.
- Search: `linspace(0.005, 0.1, 191)`, both endpoints included. The step is `0.0005`. The minimizer is the grid argmin. A tie keeps the smaller `ν`.
- Arms: Stage B data-only, and Stage C hybrid with residual weight `1e-2`. Seeds 0 through 4. No other hybrid weight.
- Objective weight `λ ∈ {0, 1, 10}`, chosen on validation `hard_ood` only, separately per arm, by the mean across seeds of the mean relative viscosity error. Ties keep the smaller `λ`. `λ = 0` is always scored. A positive `λ` that loses validation is not scored on test.

The Stage A record [docs/v02/inverse_stress.json](docs/v02/inverse_stress.json) is the baseline. `stage-d-scores` recomputes the closed-form tables and refuses to write a file if those published fields move.

## Estimator

`ν` is the unknown scalar channel of a checkpoint that was already trained. It is normalized by the training-split mean and population standard deviation stored in the checkpoint. The network weights are not updated.

The input window does not depend on `ν`. Frames 0 through 4 are the opening burst, lifted from the 32 sensors onto the trajectory grid by a real FFT zero-pad. The coarse Nyquist coefficient is halved before padding. Frames 5, 6, and 7 are a degree-2 Newton extrapolation of lifted frames 2, 3, and 4, with coefficients `(1, -3, 3)`, `(3, -8, 6)`, and `(6, -15, 10)`. That fill is not a Burgers solution.

The FNO then rolls forward in blocks of 8 frames. Each predicted block is the next input. The rollout stops at the block that contains frame 76. The primary objective is the mean square error between that rollout and the masked sensors on the bursts that start at frames 24, 48, and 72. Samples off the mask are not read for the fit.

For `λ > 0` the objective is

    sensor_mse(ν) / sensor_mse(ν_train) + λ * residual_mse(ν) / residual_mse(ν_train),

where `ν_train` is the training-split mean viscosity and `residual_mse` is the Stage 4 central Burgers residual on `sensors32_bursts`. `λ = 0` is the sensor mean square alone. The same argmin is obtained if that sensor curve is divided by the positive training-mean anchor.

An oracle one-step ceiling is stored and is not used to choose `λ`. Each input window is the true full field, and the loss is still only the masked sensors on that one predicted block. It separates operator error from the sparse initial window. It is not a sparse estimator.

`alias_rel_l2` is the Stage A diagnostic: at the saved frame of maximum `||u_x||_∞`, the relative discrete L2 between spectral `u_xx` on the 32 sensors and the full-grid spectral `u_xx` restricted to those nodes. It is computed after the fit, from the stored field, and it does not enter the objective.

## Operators

Checkpoints are not in git. Both arms were retrained from the frozen protocols on `artifacts/burgers_hard_pilot`, with field hashes checked against [docs/stage_a/pilot_manifest.json](docs/stage_a/pilot_manifest.json).

Data-only uses [docs/v02/stage_b_train_protocol.json](docs/v02/stage_b_train_protocol.json): 30 epochs, batch 32, learning rate `0.001`, width 32, 16 modes, 4 layers, window 8/8/8, normalized data MSE, best validation relative L2. Selected epochs are 28, 28, 29, 26, 25, the same epochs as the Stage B table. Seed 0 validation relative L2 matches the Stage B value `0.00398207875714179`. The other four seeds differ by at most `8.375410048894541e-05` (seed 2).

Hybrid uses [docs/v02/stage_c_train_protocol.json](docs/v02/stage_c_train_protocol.json) at residual weight `0.01`, scope `with_input`, physical space, `Δt = 0.01`. Selected epochs are 29, 19, 28, 22, 30. The Stage C table is 29, 30, 28, 22, 30. Seed 1 is the mismatch: this retrain selects epoch 19 at validation relative L2 `0.00376379891486624`, against the published epoch 30 at `0.00277956952462546`. The other hybrid seeds stay within `1.821631212578119e-04` of the published validation relative L2. CPU torch was not bitwise stable on that hybrid seed. The Stage B and Stage C forecast tables were not edited. Stage D scores these retrained checkpoints, the same files the validation selection read.

## Validation weight

[docs/v02/stage_d_objective_selection.json](docs/v02/stage_d_objective_selection.json) was committed before the test curves. The metric is the mean, across seeds, of the mean relative error on validation `hard_ood`.

| arm | `λ = 0` | `λ = 1` | `λ = 10` | selected |
| --- | ---: | ---: | ---: | ---: |
| data-only | 0.045295874993332444 ± 0.020827868216910086 | 0.11821208664800702 ± 0.016969948545543626 | 0.2678035678835615 ± 0.005280913676815287 | 0 |
| hybrid `1e-2` | 0.033815910631525885 ± 0.006119802718807989 | 0.09717104954310071 ± 0.008568149096362113 | 0.26384730102459675 ± 0.0029030652787909337 | 0 |

`λ = 0` wins on both arms. `λ = 1` and `λ = 10` are worse on every seed. Adding the aliased residual pulls the fit back toward the closed-form failure mode. Those two weights are not scored on test.

## Commands

```bash
python -m pinnforge.operator train \
  --protocol docs/v02/stage_b_train_protocol.json \
  --pilot artifacts/burgers_hard_pilot \
  --manifest docs/stage_a/pilot_manifest.json \
  --output runs/stage_d/train/data_only/seed_0 \
  --epochs 30 --batch-size 32 --lr 0.001 \
  --width 32 --modes 16 --layers 4 \
  --input-frames 8 --output-frames 8 --stride 8 \
  --seed 0 --loss data

python -m pinnforge.operator train \
  --protocol docs/v02/stage_c_train_protocol.json \
  --pilot artifacts/burgers_hard_pilot \
  --manifest docs/stage_a/pilot_manifest.json \
  --output runs/stage_d/train/hybrid_1e-2/seed_0 \
  --epochs 30 --batch-size 32 --lr 0.001 \
  --width 32 --modes 16 --layers 4 \
  --input-frames 8 --output-frames 8 --stride 8 \
  --seed 0 --loss hybrid --residual-weight 0.01 \
  --residual-scope with_input --residual-space physical --dt 0.01

python -m pinnforge.operator operator-inverse \
  --protocol docs/v02/stage_d_inverse_protocol.json \
  --pilot artifacts/burgers_hard_pilot \
  --manifest docs/stage_a/pilot_manifest.json \
  --checkpoint runs/stage_d/train/data_only/seed_0/checkpoint.pt \
  --arm data_only --split val \
  --output runs/stage_d/val/data_only_seed_0.json

python -m pinnforge.operator select-objective \
  --protocol docs/v02/stage_d_inverse_protocol.json \
  --inputs runs/stage_d/val/data_only_seed_0.json \
  --output docs/v02/stage_d_objective_selection.json

python -m pinnforge.operator stage-d-scores \
  --protocol docs/v02/stage_d_inverse_protocol.json \
  --selection docs/v02/stage_d_objective_selection.json \
  --inputs runs/stage_d/test/data_only_seed_0.json \
  --output docs/v02/stage_d_scores.json
```

The committed selection and score files use all ten curves, seeds 0 through 4 of both arms. `select-objective` rejects a test curve. `stage-d-scores` reads the test split only.

## Baselines

Same 128 test instances. Closed-form numbers match [docs/v02/inverse_stress.json](docs/v02/inverse_stress.json) on every field that file publishes for these slices. "Worse than baseline" counts instances whose absolute error exceeds the training-mean constant. Correlation is Pearson correlation of `ν̂` with `ν`. The training-mean correlation is numerical dust, and it is undefined on `ν < 0.02` because `ν̂` does not vary.

| estimator | slice | n | MAE | mean rel | median rel | max rel | failures | correlation | worse than baseline |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sensors32_bursts` | full test | 128 | 0.0030159391726451274 | 0.1171354466174288 | 0.047671230713532645 | 0.9497873814639219 | 9 | 0.9975830065148409 | 2 |
| `sensors32_bursts` | `hard_ood` | 30 | 0.003878099063123072 | 0.3494591802034973 | 0.16533814561780605 | 0.9497873814639219 | 9 | 0.9575721159355071 | 0 |
| `sensors32_bursts` | complement | 98 | 0.0027520126755600427 | 0.04601593633597921 | 0.04141721701903255 | 0.1965295204205419 | 0 | 0.9974501288808931 | 2 |
| `sensors32_bursts` | `ν < 0.02` | 20 | 0.004485516759132618 | 0.46966295632108307 | 0.49552948190684665 | 0.9497873814639219 | 9 | 0.8254909890240939 | 0 |
| dense | full test | 128 | 0.0011637263530198242 | 0.01585009767787492 | 0.013821166946874056 | 0.04757213144149604 | 0 | 0.9998551835376941 | 2 |
| dense | `hard_ood` | 30 | 2.677073678153862e-05 | 0.0013208386837906374 | 0.0007840488827616291 | 0.0037678982358051883 | 0 | 0.9999956055739502 | 0 |
| dense | complement | 98 | 0.001511773990643789 | 0.020297830023002762 | 0.018565062059468465 | 0.04757213144149604 | 0 | 0.999783111424388 | 2 |
| dense | `ν < 0.02` | 20 | 9.650514684458573e-06 | 0.0007512237585760504 | 0.0006320713582617869 | 0.0027730587363552655 | 0 | 0.9999942863393853 | 0 |
| training mean | full test | 128 | 0.024858320887062635 | 0.96585356303953 | 0.39846528013115035 | 6.862441207259799 | 44 | ~0 | 0 |
| training mean | `hard_ood` | 30 | 0.035865859332177535 | 2.98108138535235 | 2.641231506763628 | 6.862441207259799 | 30 | ~0 | 0 |
| training mean | complement | 98 | 0.02148866626100706 | 0.3489470868213198 | 0.33066439405321574 | 0.9181815323955073 | 14 | ~0 | 0 |
| training mean | `ν < 0.02` | 20 | 0.040069343664762 | 3.904257958224511 | 3.4905430874135455 | 6.862441207259799 | 20 | null | 0 |

All 9 sparse failures sit in `hard_ood` and in `ν < 0.02`. The complement has none. The dense stencil has none on any slice.

## Operator results

Mean ± sample standard deviation across seeds 0 through 4 (`ddof = 1`). No seed was dropped. The scored objective is `λ = 0`.

### Data-only

| slice | n | MAE | mean rel | median rel | max rel | failures | correlation | worse than baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full test | 128 | 0.000773951890090851 ± 0.0003991633349422973 | 0.021866330905899147 ± 0.008819091358584748 | 0.013965623882809353 ± 0.00768733741208179 | 0.2922084272467235 ± 0.054177649429743346 | 0 ± 0 | 0.9997624518437667 ± 8.259595625306263e-05 | 1.2 ± 1.0954451150103321 |
| `hard_ood` | 30 | 0.0005861990341131721 ± 0.00023918023191149342 | 0.049487208469027544 ± 0.0144232271670265 | 0.0315188393346046 ± 0.01560588973425255 | 0.2922084272467235 ± 0.054177649429743346 | 0 ± 0 | 0.9962642672136852 ± 0.001268293900285726 | 0 ± 0 |
| complement | 98 | 0.0008314272541656506 ± 0.00045556876861729617 | 0.01341096022330882 ± 0.007957033491083166 | 0.011595975581466073 ± 0.007554098931445906 | 0.03755442593267517 ± 0.013800193924355221 | 0 ± 0 | 0.9996318531207325 ± 0.0001507529185686737 | 1.2 ± 1.0954451150103321 |
| `ν < 0.02` | 20 | 0.0006183303587231603 ± 0.0001759049683277322 | 0.06339223281191389 ± 0.015643388668796922 | 0.04015646987123901 ± 0.014389968416492732 | 0.2922084272467235 ± 0.054177649429743346 | 0 ± 0 | 0.9853977104393021 ± 0.004894666063483187 | 0 ± 0 |

### Hybrid `1e-2`

| slice | n | MAE | mean rel | median rel | max rel | failures | correlation | worse than baseline |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full test | 128 | 0.0005431187111602913 ± 0.0001300995210490847 | 0.015381956640685085 ± 0.0027924678283891245 | 0.00944136395667491 ± 0.0016932344212610805 | 0.18655281827982423 ± 0.053622511975932895 | 0 ± 0 | 0.9998097324167515 ± 0.0001171266310466537 | 0.8 ± 0.8366600265340756 |
| `hard_ood` | 30 | 0.0003750451927194688 ± 3.859311625385581e-05 | 0.03368266375533967 ± 0.0034734247381767768 | 0.019081007438454485 ± 0.002576182136905238 | 0.18655281827982423 ± 0.053622511975932895 | 0 ± 0 | 0.9979143182207665 ± 0.000680201109281932 | 0 ± 0 |
| complement | 98 | 0.0005945697882340124 ± 0.00016350615410051138 | 0.009779699360688785 ± 0.002898987551908216 | 0.00820332249500937 ± 0.0017978530082805588 | 0.03912937883190184 ± 0.020186186182230264 | 0 ± 0 | 0.999690804021221 ± 0.00022398852438651893 | 0.8 ± 0.8366600265340756 |
| `ν < 0.02` | 20 | 0.0004108898280481193 ± 3.7448412049836056e-05 | 0.044268482159261324 ± 0.004262785004741288 | 0.02967669765325361 ± 0.0061476051675649675 | 0.18655281827982423 ± 0.053622511975932895 | 0 ± 0 | 0.9933189478056381 ± 0.0009346345540773512 | 0 ± 0 |

On `hard_ood`, the data-only mean relative error is about 7.1 times smaller than the closed-form mask (`0.3494591802034973`), and the hybrid mean is about 10.4 times smaller. Both stay above the dense ceiling (`0.0013208386837906374`): about 37 times that ceiling for data-only and about 25 times for hybrid. Neither arm has a `0.5`-rule failure on any slice, on any seed.

The few instances worse than the training-mean constant are in the complement, next to that constant: 408 and 590, which Stage A already flagged for the closed-form mask, and 587. One grid step of `0.0005` is the same size as the distance from those viscosities to the training mean. None of them is a `0.5`-rule failure, and none is in `hard_ood`.

## Paired comparison on `hard_ood`

Each seed is compared, instance by instance, with the closed-form `sensors32_bursts` estimate. Delta is operator relative error minus closed-form relative error. A negative delta is a lower error for the operator.

| arm | mean rel − LS | instances lower | instances higher | failures |
| --- | ---: | ---: | ---: | ---: |
| data-only | -0.2999719717344699 ± 0.014423227167026476 | 29.4 ± 0.5477225575051661 | 0.6 ± 0.5477225575051662 | 0 ± 0 |
| hybrid `1e-2` | -0.3157765164481578 ± 0.003473424738176789 | 29.2 ± 0.4472135954999579 | 0.8 ± 0.44721359549995804 | 0 ± 0 |

Per seed, `hard_ood` mean relative error. The closed-form value on this slice is `0.3494591802034973` for every row.

| seed | data-only | hybrid `1e-2` | data-only lower / higher | hybrid lower / higher |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.05279371356956865 | 0.034637756095587585 | 29 / 1 | 30 / 0 |
| 1 | 0.058113644644793364 | 0.0322982674630476 | 29 / 1 | 29 / 1 |
| 2 | 0.06626046901346347 | 0.03910361776610336 | 29 / 1 | 29 / 1 |
| 3 | 0.029966696708239884 | 0.029848007961202478 | 30 / 0 | 29 / 1 |
| 4 | 0.04030151840907237 | 0.03252566949075731 | 30 / 0 | 29 / 1 |

Every seed of both arms is below the closed-form slice mean. The weakest operator seed is data-only seed 2 at `0.06626046901346347`, still about 5.3 times smaller than the closed-form mean. At the same seed index, the hybrid mean is lower than the data-only mean on all five seeds. That comparison uses these retrains. It is not a new hybrid-weight search.

The only `hard_ood` instance on which any operator seed loses to the closed form is instance `389`, `ν = 0.0198438261937898`, alias `0.4088110371918945`. The closed form is already accurate there: `ν̂ = 0.020043654512950804`, relative error `0.010070049858809124`. Data-only seeds 0, 1, and 2 land at `0.0205` or `0.022`. Hybrid seeds 1 through 4 land at `0.0195`, and seed 0 at `0.02`. The losses are one to four grid steps. None of them fails the `0.5` rule.

## The nine closed-form failures

All nine `sensors32_bursts` failures, with the mean relative error of each operator arm across the five seeds. Every seed of both arms stays under `0.5` on every one of these instances. The largest single-seed relative error among them is data-only on instance `619`, `0.3062951139458773`.

| instance | `ν` | alias | LS rel | LS failure | data-only mean rel | hybrid mean rel |
| ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 530 | 0.007820793287511035 | 0.42900313273292634 | 0.9497873814639219 | yes | 0.0977085865090548 | 0.053804425208756904 |
| 743 | 0.0083563697313075 | 0.3856680362668411 | 0.9365582569236526 | yes | 0.06505579410348097 | 0.03424670884814036 |
| 338 | 0.007689406142113342 | 0.733078525743896 | 0.8945644358983865 | yes | 0.04078906478229074 | 0.08965661708744999 |
| 619 | 0.007207675915965656 | 0.9081332472259828 | 0.8600432285811773 | yes | 0.12828067748319652 | 0.1005324820410316 |
| 655 | 0.008799276647784498 | 0.8386900051585175 | 0.8264779460832664 | yes | 0.04089601270207875 | 0.031771473935995435 |
| 185 | 0.006878265076856729 | 0.8636400249562893 | 0.7136988908141626 | yes | 0.06915305095928528 | 0.1567641061820608 |
| 375 | 0.011375559120315824 | 0.5181249591576774 | 0.5728204481644777 | yes | 0.03293592199273841 | 0.050556796196159035 |
| 391 | 0.010141552360801248 | 0.7299963656673164 | 0.5328612255755112 | yes | 0.08464657171389177 | 0.022512379174090373 |
| 83 | 0.011835399149504187 | 0.3235601430854653 | 0.5240345713792666 | yes | 0.03657841739264865 | 0.028129188200054056 |

## Where the operator error sits

Worst `hard_ood` instances by mean relative error across seeds. `ν̂` is the five seeds.

Data-only:

| instance | `ν` | alias | LS rel | mean rel | `ν̂` |
| ---: | ---: | ---: | ---: | ---: | --- |
| 179 | 0.00659586199923937 | 0.867979357173466 | 0.4982505498433595 | 0.28868675557193507 | 0.0085, 0.0085, 0.009, 0.008, 0.0085 |
| 619 | 0.007207675915965656 | 0.9081332472259828 | 0.8600432285811773 | 0.12828067748319652 | 0.0065, 0.005, 0.007, 0.0075, 0.006 |
| 530 | 0.007820793287511035 | 0.42900313273292634 | 0.9497873814639219 | 0.09770858650905481 | 0.0085, 0.0065, 0.008, 0.007, 0.007 |
| 391 | 0.010141552360801248 | 0.7299963656673164 | 0.5328612255755112 | 0.08464657171389177 | 0.0115, 0.0105, 0.012, 0.0105, 0.0105 |
| 185 | 0.006878265076856729 | 0.8636400249562893 | 0.7136988908141626 | 0.06915305095928528 | 0.0055, 0.0065, 0.007, 0.007, 0.0065 |

Hybrid `1e-2`:

| instance | `ν` | alias | LS rel | mean rel | `ν̂` |
| ---: | ---: | ---: | ---: | ---: | --- |
| 185 | 0.006878265076856729 | 0.8636400249562893 | 0.7136988908141626 | 0.1567641061820608 | 0.0055, 0.006, 0.005, 0.006, 0.0065 |
| 179 | 0.00659586199923937 | 0.867979357173466 | 0.4982505498433595 | 0.12191552838027275 | 0.0075, 0.0075, 0.007, 0.0075, 0.0075 |
| 619 | 0.007207675915965656 | 0.9081332472259828 | 0.8600432285811773 | 0.1005324820410316 | 0.006, 0.007, 0.0065, 0.0075, 0.006 |
| 338 | 0.007689406142113342 | 0.733078525743896 | 0.8945644358983865 | 0.08965661708744999 | 0.007, 0.0065, 0.007, 0.0075, 0.007 |
| 530 | 0.007820793287511035 | 0.42900313273292634 | 0.9497873814639219 | 0.053804425208756904 | 0.0075, 0.0075, 0.007, 0.0075, 0.0075 |

Instance `179` is the smallest test viscosity. Stage C already listed it as the worst hybrid forecast when `ν` is given. Here it is the worst data-only inverse and the second-worst hybrid inverse. The estimates sit a few grid steps above the truth (`0.008` to `0.009` against `0.00659586199923937`). Relative error `0.29` is not one bin: the bin width `0.0005` is a relative error of about `0.076` at this viscosity. The fit is biased high, and it still clears the `0.5` rule. The closed form on this instance is `0.4982505498433595`, just under the rule, so it is not one of the nine failures.

## Why the operator helps, and where it stops

The closed form inverts a residual whose diffusion term is `u_xx` on the 32-sensor grid. Stage A measured that second derivative as aliased even when the field itself has little energy above mode 16. The FNO does not invert that stencil. It matches later sensor bursts to a rollout whose spatial operators were learned on the full `N = 256` label grid, with 16 Fourier modes. High-mode aliasing in the residual normal equation is not an input to this fit.

On `hard_ood`, Pearson correlation of `alias_rel_l2` with operator relative error is `0.4315478426975135 ± 0.14201189148940227` for data-only and `0.5712060912620449 ± 0.0850094909514551` for hybrid. Aliasing still tracks the remaining error. Correlation of the same alias with the paired delta (operator minus closed form) is `-0.43596999794919933 ± 0.01936362085700128` and `-0.44816935853746553 ± 0.016891312731690797`. The gain over the closed form is larger where the sensor second derivative is more aliased.

The sensor objective is not flat. On `hard_ood`, the median of `(max − min) / min` of the sensor mean-square curve is `358.4338065710661 ± 71.029642732571` for data-only and `545.0504753952839 ± 319.8441938995013` for hybrid. The 191-point grid is resolving a peaked mismatch, not a plateau. That is also why a positive `λ` hurts: the residual term is the aliased stencil, and validation `hard_ood` gets worse as soon as it is added.

The oracle one-step ceiling, true full-field windows, `λ = 0`, does not remove the remaining error.

| arm | sparse rollout `hard_ood` mean rel | oracle one-step `hard_ood` mean rel |
| --- | ---: | ---: |
| data-only | 0.049487208469027544 ± 0.0144232271670265 | 0.044655927540203824 ± 0.020248534272213757 |
| hybrid `1e-2` | 0.03368266375533967 ± 0.0034734247381767768 | 0.03500070786514242 ± 0.005692905310658921 |

Data-only improves slightly when the true window replaces the quadratic fill. Hybrid does not. Both oracles still have 0 failures, and both stay an order of magnitude above the dense residual least squares. The quadratic opening is not the dominant error. What remains is the operator: 16 modes, a `0.0005` viscosity grid, and a forecast that Stage B and Stage C already showed is hardest at low `ν`. The operator is accurate enough to stay under the `0.5` rule on the instances where the aliased residual least squares is not, and it is not accurate enough to match observations of the full field.

## What is not claimed

- The `hard_ood` cut, the sensor mask, and the Stage A, B, and C protocols were not retuned. The published Stage A inverse table and the Stage B and Stage C forecast tables were not edited.
- These checkpoints are retrains of the frozen protocols. Hybrid seed 1 does not reproduce the Stage C selected epoch. The Stage D numbers are this retrain, not a claim that the original Stage C weight file was recovered byte for byte.
- The sparse fit does not read the field off `sensors32_bursts`. The alias diagnostic does, and it is not part of the objective.
- The three-frame quadratic fill is not a Burgers solution. The oracle ceiling is not a sparse estimator and was not used to choose `λ`.
- `λ = 1` and `λ = 10` were not scored on test. Validation rejected both.
- No learned inverse network, no new architecture, and no new hybrid weight. Sixteen Fourier modes still do not cover the 48-mode initial band.
- One seed is not the result. The tables are the five-seed mean and sample standard deviation.
- Beating the closed form on this mask is not the dense ceiling. Dense `hard_ood` mean relative error remains `0.0013208386837906374`.
- The grid step `0.0005` is a resolution floor. Relative error of one bin is about `0.076` at `ν = 0.0066` and about `0.019` at the `hard_ood` threshold.
- The version string is still `0.1.0`. This is not a `0.2.0` release.

## Tests

```bash
python -m ruff check .
python -m pytest -q --tb=line
```

`tests/test_stage_d.py` checks that the committed protocol keeps the Stage 5 mask, the 191-point grid, and validation-only `λ` selection, that the quadratic window ignores samples off the mask, that a test curve is rejected by `select-objective`, and that the committed score file still reports 9 closed-form `hard_ood` failures and a lower operator mean on that slice. Curve evaluation is marked `ml`.
