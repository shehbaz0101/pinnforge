# Stage C report

Residual and hybrid losses on the Stage B 1D Fourier neural operator, five seeds, scored on the full test split and on the preregistered `hard_ood` slice. The version string stays `0.1.0`. This run used Python 3.12.3, NumPy 2.4.4, and the CPU torch wheel 2.14.0+cpu. The data-only arm is the Stage B table. It was not retrained, and its numbers were not edited.

The training contract was written to [docs/v02/stage_c_train_protocol.json](docs/v02/stage_c_train_protocol.json) before any test score from this stage. Its SHA-256 is `1631d7d60e83ea1bdecf1d63de550f952f54f835385cf66786e563a643cd9708`. Each run manifest records that hash. The file was not edited after the test evaluation. `hard_ood` is still `ν <= 0.027028120493367818` from [docs/v02/pilot_protocol.json](docs/v02/pilot_protocol.json). That quartile was not refit.

The hybrid weight was chosen from validation manifests only, then written to [docs/v02/stage_c_weight_selection.json](docs/v02/stage_c_weight_selection.json), and only then were test slices scored. The selected weight is `1e-2`. Weights `1e-6` and `1e-4` were not scored on test.

`pinnforge.reference.burgers.reference_solution` still raises `NotImplementedError`. `pinnforge eval` and `pinnforge demo --equation burgers` still score residual metrics. They do not load these windows. The Stage 2 pilot path is unchanged: omit `--protocol` and point `--pilot` and `--manifest` at `artifacts/burgers_pilot` and `docs/stage2/pilot_manifest.json`. A Stage B protocol file still rejects a residual or hybrid loss. No inverse model was trained.

## Task

The operator, the window, and the pilot are the Stage B task. Each trajectory is a periodic solution of

    u_t + u u_x = ν u_xx,    x ∈ [-1, 1],    t ∈ [0, 1],

saved at `N = 1024` and `save_dt = 0.01`. The map is

    (u(·, t), …, u(·, t + 7Δ), ν)  ↦  (u(·, t + 8Δ), …, u(·, t + 15Δ))

with `Δ = 0.01`, stride 8, and 11 windows per trajectory. `ν` is a known input channel. It is not a predicted field.

The loss arms are the Stage 4 three:

| mode | optimized scalar |
| --- | --- |
| `data` | mean squared error of normalized `u` (Stage B reference, not retrained) |
| `residual` | mean of `R²` |
| `hybrid` | data MSE + `residual_weight` × mean of `R²` |

`R` is the Stage 4 stencil: spectral `u_x` and `u_xx` with the Nyquist multiplier set to 0, and a second-order central difference in time at `Δt = 0.01`. The scope is `with_input`. The field is physical `u`. `target_interior` and the normalized-space residual were not arms.

## Protocol

| item | locked value |
| --- | --- |
| seeds | 0, 1, 2, 3, 4 |
| epochs | 30, plus epoch 0 before any step |
| early stopping | none |
| checkpoint | lowest validation mean relative L2. Ties would keep the earliest epoch |
| test in the loss, the epoch choice, or the weight choice | no |
| optimizer | Adam, learning rate 0.001, constant |
| batch size | 32 |
| width, modes, layers | 32, 16, 4 |
| parameters | 138248 |
| windows | 8 input, 8 target, stride 8 |
| hybrid weights | `1e-6`, `1e-4`, `1e-2` |
| weight rule | lowest arithmetic mean, across seeds, of the selected-epoch validation mean relative L2. A tie would keep the smaller weight |
| device | cpu |
| primary metric | mean per-window relative L2 on denormalized `u` |
| aggregation | arithmetic mean and sample standard deviation (`ddof = 1`) of one scalar per seed |
| `hard_ood` | `ν <= 0.027028120493367818`, not refit |
| test counts | 128 full, 30 `hard_ood`, 98 complement |

The architecture matches [docs/v02/stage_b_train_protocol.json](docs/v02/stage_b_train_protocol.json). Sixteen Fourier modes still do not cover the 48-mode initial band. That was not changed after these scores.

At epoch 0 of residual seed 0, before any test score, the training data MSE is `0.9029178454946877` and the training residual MSE is `88.30589384496004`. Weight `1e-2` puts those two hybrid terms on the same order at initialization, `1e-4` is data-leaning, and `1e-6` is almost data-only. That is the same spacing Stage 4 used. The grid was not rebalanced after this observation.

## Pilot

`artifacts/burgers_hard_pilot/` is gitignored. This workspace regenerated it with the Stage A command. The solver-config SHA-256 was `955dc2f9687ae7131737e37c0b6dc2f790c70ebfcc2b8e67ffca30f29de28bd6`, the same value as [STAGE_A_REPORT.md](STAGE_A_REPORT.md) and [STAGE_B_REPORT.md](STAGE_B_REPORT.md).

```bash
python -m pinnforge.reference.numerical hard-pilot --output artifacts/burgers_hard_pilot
```

Training read train and validation only. Slice scoring read the test split after the weight file existed. Loaded fields were checked against `field_sha256` in [docs/stage_a/pilot_manifest.json](docs/stage_a/pilot_manifest.json).

## Commands

Shared training flags: `--epochs 30 --batch-size 32 --lr 0.001 --width 32 --modes 16 --layers 4 --input-frames 8 --output-frames 8 --stride 8`, residual scope `with_input`, space `physical`, `--dt 0.01`. `runs/` is gitignored.

```bash
python -m pinnforge.operator train \
  --protocol docs/v02/stage_c_train_protocol.json \
  --pilot artifacts/burgers_hard_pilot \
  --manifest docs/stage_a/pilot_manifest.json \
  --output runs/stage_c/residual/seed_0 \
  --epochs 30 --batch-size 32 --lr 0.001 \
  --width 32 --modes 16 --layers 4 \
  --input-frames 8 --output-frames 8 --stride 8 \
  --seed 0 --loss residual \
  --residual-scope with_input --residual-space physical --dt 0.01
```

Repeat for seeds 1–4. Hybrid runs use `--loss hybrid --residual-weight` equal to `1e-6`, `1e-4`, or `1e-2`, and write `runs/stage_c/hybrid_w1em6`, `hybrid_w1em4`, or `hybrid_w1em2`. `--loss data` with this protocol is rejected. The Stage B command remains the data-only baseline.

```bash
python -m pinnforge.operator select-hybrid \
  --protocol docs/v02/stage_c_train_protocol.json \
  --inputs runs/stage_c/hybrid_w1em6/seed_{0,1,2,3,4}/manifest.json \
          runs/stage_c/hybrid_w1em4/seed_{0,1,2,3,4}/manifest.json \
          runs/stage_c/hybrid_w1em2/seed_{0,1,2,3,4}/manifest.json \
  --output docs/v02/stage_c_weight_selection.json
```

Test scoring, after that file:

```bash
python -m pinnforge.operator slices \
  --pilot artifacts/burgers_hard_pilot \
  --manifest docs/stage_a/pilot_manifest.json \
  --protocol docs/v02/pilot_protocol.json \
  --train-protocol docs/v02/stage_c_train_protocol.json \
  --weight-selection docs/v02/stage_c_weight_selection.json \
  --checkpoint runs/stage_c/hybrid_w1em2/seed_0/checkpoint.pt \
  --batch-size 32 \
  --output runs/stage_c/hybrid_w1em2/seed_0/slices.json

python -m pinnforge.operator aggregate \
  --protocol docs/v02/stage_c_train_protocol.json \
  --weight-selection docs/v02/stage_c_weight_selection.json \
  --inputs runs/stage_c/hybrid_w1em2/seed_{0,1,2,3,4}/slices.json \
  --output runs/stage_c/hybrid_aggregate.json

python -m pinnforge.operator stage-c-scores \
  --protocol docs/v02/stage_c_train_protocol.json \
  --selection docs/v02/stage_c_weight_selection.json \
  --residual runs/stage_c/residual_aggregate.json \
  --hybrid runs/stage_c/hybrid_aggregate.json \
  --output docs/v02/stage_c_scores.json
```

The residual arm uses `runs/stage_c/residual/seed_<seed>` in the same `slices` and `aggregate` commands. A hybrid checkpoint whose weight is not the selected `1e-2` is rejected before the test split is scored.

## Weight choice

Validation only. Mean ± sample standard deviation of the selected-epoch validation mean relative L2.

| weight | mean validation relative L2 | selected epochs |
| ---: | ---: | --- |
| 1e-6 | 3.867427993099177e-03 ± 4.237471672307178e-04 | 28, 28, 29, 24, 29 |
| 1e-4 | 4.224597748415203e-03 ± 7.173968558788702e-04 | 25, 19, 29, 30, 25 |
| 1e-2 | 3.227673250584623e-03 ± 3.549295286876422e-04 | 29, 30, 28, 22, 30 |

The selected weight is `1e-2`. It is the minimum of these three means. It is not a search outside the grid.

For reference, the Stage B data-only validation mean relative L2 was `4.272110578045965e-03 ± 3.891333760268412e-04`. That number was already published. It was not used to pick the weight. The weight rule uses only the three hybrid rows above.

Mean fit time after the windows were loaded was about 509 s for residual, 508 s for weight `1e-6`, 535 s for `1e-4`, and 532 s for `1e-2`. Those clocks are not a hardware benchmark.

## Test scores

Mean ± sample standard deviation across seeds 0–4. The data-only column is copied from [docs/v02/stage_b_scores.json](docs/v02/stage_b_scores.json). Persistence does not depend on the seed.

| slice | data-only (Stage B) | residual | hybrid, weight 1e-2 | one-step persistence |
| --- | ---: | ---: | ---: | ---: |
| full test | 4.298970125314308e-03 ± 3.608651258270467e-04 | 3.463482404205462e-03 ± 5.112531513455785e-04 | 3.243082987510490e-03 ± 3.669287417174703e-04 | 8.456701056846332e-02 |
| `hard_ood` | 7.049252763443842e-03 ± 3.032654479273341e-04 | 5.704581248709277e-03 ± 7.767311465741109e-04 | 5.215427637735405e-03 ± 6.736875927560861e-04 | 9.782188328427069e-02 |
| complement | 3.457046868744042e-03 ± 3.865637014640921e-04 | 2.777431737520620e-03 ± 4.484855473780170e-04 | 2.639304012951843e-03 ± 3.084966195291301e-04 | 8.050939647178759e-02 |

On `hard_ood`, the hybrid mean is lower than the Stage B data-only mean. The difference of means is `-1.833825125708437e-03`. Paired by seed, hybrid minus data-only is

    -1.500748739815682e-03, -2.465292886100502e-03, -1.902777473182949e-03,
    -9.777930807938423e-04, -2.322513448649208e-03

with paired mean `-1.833825125708437e-03` and paired sample standard deviation `6.095188627875928e-04`. All five paired differences are negative. The hybrid `hard_ood` mean is `0.740` of the Stage B `hard_ood` mean.

Residual-only is also lower on `hard_ood` on every seed. Its difference of means is `-1.344671514734564e-03`. Seed 1 moves only `-4.115110761624286e-05`, which is much smaller than the other four seeds. Residual-only is not lower on every slice: on seed 1 the full-test paired difference is `+2.461749226855816e-04` and the complement paired difference is `+3.341318707371613e-04`. The residual mean on those two slices is still lower because the other seeds decrease. Hybrid is lower on all five seeds on the full test, on `hard_ood`, and on the complement.

The `hard_ood` mean remains about twice the complement mean: `1.976` for the hybrid and `2.054` for residual-only. Per seed the hybrid ratio is `2.187`, `2.009`, `1.918`, `2.086`, and `1.705` (mean `1.981 ± 0.183`). The physics loss did not remove the slice gap. The threshold was not moved.

Against one-step persistence, the hybrid `hard_ood` mean is `0.0533` of the baseline (persistence is 18.8 times larger). The Stage B data-only ratio on that slice was `0.0721` (13.9 times). The network still beats persistence. The margin is larger than in Stage B and still smaller than on the complement.

| slice | hybrid median relative L2 | hybrid pooled relative L2 | hybrid normalized MSE |
| --- | ---: | ---: | ---: |
| full test | 2.534177431902276e-03 ± 3.302368466128896e-04 | 4.725507684886816e-03 ± 5.230213263057131e-04 | 2.038944015863404e-05 ± 4.480666150577851e-06 |
| `hard_ood` | 4.399655929008957e-03 ± 6.609987481720335e-04 | 6.795731220229420e-03 ± 7.828275136574823e-04 | 6.153183468257717e-05 ± 1.414669367477271e-05 |
| complement | 2.311013999817171e-03 ± 3.425452872768354e-04 | 3.148516912977359e-03 ± 3.761943469251242e-04 | 7.794829590080019e-06 ± 1.807009369723238e-06 |

The median sits below the mean, and the pooled score sits above the mean, on every hybrid slice. A few windows still carry more of the squared error than a typical window.

## Per seed, `hard_ood`

| seed | data-only | residual | hybrid `1e-2` | residual epoch | hybrid epoch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 6.781261847096676e-03 | 5.306205240325827e-03 | 5.280513107280994e-03 | 30 | 29 |
| 1 | 6.962370482083190e-03 | 6.921219374466947e-03 | 4.497077595982688e-03 | 21 | 30 |
| 2 | 7.566978064983732e-03 | 5.722147548666018e-03 | 5.664200591800783e-03 | 29 | 28 |
| 3 | 7.027877414485406e-03 | 5.747492482390194e-03 | 6.050084333691564e-03 | 30 | 22 |
| 4 | 6.907776008570202e-03 | 4.825841597697399e-03 | 4.585262559920994e-03 | 29 | 30 |

No seed was dropped. Residual seed 1 is the high value on the full test and on the complement, and it is included.

## Residual magnitude

Mean of `|R|` on the `with_input` physical stencil, the same nodes the residual and hybrid arms trained on. The label floor is the mean `|R|` of the saved targets. It does not depend on the seed. Data-only test `|R|` is not in the Stage B aggregate, and this stage did not rescore that arm.

| slice | residual mean `|R|` | hybrid mean `|R|` | label floor |
| --- | ---: | ---: | ---: |
| full test | 3.291199933887241e-02 ± 7.994968405918375e-03 | 3.443932713303720e-02 ± 4.520179428157432e-03 | 8.702858719761340e-04 |
| `hard_ood` | 6.012024116303222e-02 ± 1.097023191229286e-02 | 6.357257280733616e-02 ± 7.870694371407128e-03 | 1.180897762176484e-03 |
| complement | 2.458294571923167e-02 ± 7.156958780393426e-03 | 2.552098662049670e-02 ± 3.710863986910654e-03 | 7.752005994658227e-04 |

Residual-only has the lower test mean `|R|`. Hybrid has the lower test mean relative L2. Both means are still about 40 times the full-test label floor and about 54 times the `hard_ood` label floor for the hybrid. The Stage A solver gate is relative L2 `1e-9`. The hybrid `hard_ood` mean is about `5.2e6` times that gate.

## Where the error sits

On the hybrid checkpoints, instance `179` is the worst `hard_ood` case on every seed. Its viscosity is `0.00659586199923937`, the smallest test viscosity in the manifest. On hybrid seed 0 its one-step mean relative L2 is `0.011297072023292599`, against one-step persistence `0.12241994740816874`. Instances `619`, `338`, and `185` are in the worst five on every hybrid seed. Those three are the same Stage A `sensors32_bursts` failures that Stage B already listed. The physics loss did not move the worst forecasts off the lowest viscosities.

The complement is still not uniformly easy, and `hard_ood` is still not the list of the individually worst windows in every other sense. The slice mean is higher on `hard_ood` on every seed of both physics arms. The threshold was not edited.

## Rollout

Secondary. Not used to choose the epoch or the weight. Mean ± sample standard deviation of the per-instance relative L2 on frames 8 through 95.

| slice | residual rollout | hybrid rollout | open-loop persistence |
| --- | ---: | ---: | ---: |
| full test | 1.378911513010553e-02 ± 1.913753604548061e-03 | 1.233315614962996e-02 ± 1.993892269713270e-03 | 9.766351855045150e-01 |
| `hard_ood` | 2.574377333219913e-02 ± 2.548976260538788e-03 | 2.104377602006243e-02 ± 2.621035045734292e-03 | 9.268796039417675e-01 |
| complement | 1.012952588456668e-02 ± 1.780431097018807e-03 | 9.666639862762874e-03 ± 2.038565632190447e-03 | 9.918664859829069e-01 |

The hybrid `hard_ood` rollout mean is about 4.03 times its one-step mean. Error still accumulates. The open-loop persistence score near 1 is a weak baseline on this horizon. Step curves are in [docs/v02/stage_c_scores.json](docs/v02/stage_c_scores.json).

## Inverse stress

No inverse model was trained. The Stage A `sensors32_bursts` record is unchanged: nine failures of the `0.5` relative-error rule, all inside `hard_ood`, and none in the complement. Forecasting `u` with `ν` given, with or without a residual term, does not estimate `ν`.

## What is not claimed

- The hybrid mean on `hard_ood` is lower than the Stage B data-only mean, on every seed in this set. That is not a claim about other grids, other widths, or other pilots. Only `{1e-6, 1e-4, 1e-2}` was trained, and only `1e-2` was scored on test.
- Residual-only is not uniformly better than data-only. Seed 1 is worse on the full test and on the complement.
- The `hard_ood` slice remains about twice as hard as the complement. The loss did not close that gap, and the threshold was not retuned to chase it.
- No learned inverse, and no change to the Stage A sensor mask or to the Stage B data-only numbers.
- No architecture search. Sixteen Fourier modes do not cover the 48-mode initial band. This run does not measure how much of the remaining error that truncation causes.
- `target_interior` and the normalized-space residual were not measured.
- The test error does not meet the `1e-9` label gate, and the predicted residual does not meet the central-difference floor of the labels.
- The open-loop persistence score near 1 is not evidence that the rollout is accurate. Rollout was not used to pick the checkpoint or the weight.
- The version string is still `0.1.0`. This is not a `0.2.0` release.

## Tests

```bash
python -m ruff check .
python -m pytest -q --tb=line
```

`tests/test_stage_c.py` checks that the committed protocol matches the Stage B architecture and the Stage 4 weight grid, that a data-only train call is rejected, that the weight rule uses validation means and breaks ties toward the smaller weight, and that a hybrid aggregate whose weight is not the selected one is rejected. In this workspace, after the slice scores were written: ruff reported `All checks passed!`, and pytest reported `373 passed, 6 skipped`.
