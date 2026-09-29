# Stage B report

Data-only 1D Fourier neural operator on the Stage A harder Burgers pilot, five seeds, scored on the full test split and on the preregistered `hard_ood` slice. The version string stays `0.1.0`. This run used Python 3.12.3, NumPy 2.4.4, and the CPU torch wheel 2.14.0+cpu. The loss is normalized data MSE. It is not a physics-informed operator and it is not an inverse-viscosity fit.

The training contract was written to [docs/v02/stage_b_train_protocol.json](docs/v02/stage_b_train_protocol.json) before any test score from this stage. Its SHA-256 is `51d4bb05ca5ae6096c22187c131d6fa52d8e025ac9b7bddfa876485615ae200a`. Each run manifest records that hash. The file was not edited after the test evaluation. `hard_ood` is still `ν <= 0.027028120493367818` from [docs/v02/pilot_protocol.json](docs/v02/pilot_protocol.json). That quartile was not refit.

`pinnforge.reference.burgers.reference_solution` still raises `NotImplementedError`. `pinnforge eval` and `pinnforge demo --equation burgers` still score residual metrics. They do not load these windows. The Stage 2 pilot path is unchanged: omit `--protocol` and point `--pilot` and `--manifest` at `artifacts/burgers_pilot` and `docs/stage2/pilot_manifest.json`.

## Task

Each harder-pilot trajectory is a periodic solution of

    u_t + u u_x = ν u_xx,    x ∈ [-1, 1],    t ∈ [0, 1],

saved at `N = 1024` and `save_dt = 0.01` (101 frames). The operator learns the Stage 3 map

    (u(·, t), …, u(·, t + 7Δ), ν)  ↦  (u(·, t + 8Δ), …, u(·, t + 15Δ))

with `Δ = 0.01`. Eight input frames and eight target frames is a lead of 0.08. Stride 8 means successive input windows on one trajectory do not overlap. On 101 frames the starts are `0, 8, …, 80` (11 windows). The last target frame used is index 95 (`t = 0.95`). Frames 96 through 100 are shorter than one window and are dropped.

The window, width, mode count, depth, batch size, learning rate, and epoch budget are the Stage 3 settings. They were not searched again on this pilot. The spectral layers keep 16 Fourier modes. The harder initial condition is band-limited at mode 48. That mismatch was known before the test scores and was not corrected afterward.

## Protocol

| item | locked value |
| --- | --- |
| loss | normalized data MSE (`--loss data`) |
| early stopping | none. All 30 epochs run |
| checkpoint | lowest validation mean relative L2. Ties would keep the earliest epoch |
| test in the loss or the selection | no |
| seeds | 0, 1, 2, 3, 4 |
| epochs | 30, plus epoch 0 before any step |
| optimizer | Adam, learning rate 0.001, constant |
| batch size | 32 |
| width, modes, layers | 32, 16, 4 |
| parameters | 138248 |
| windows | 8 input, 8 target, stride 8 |
| device | cpu |
| primary metric | mean per-window relative L2 on denormalized `u` |
| aggregation | arithmetic mean and sample standard deviation (`ddof = 1`) of one scalar per seed. Windows are not pooled across seeds. No seed is dropped |
| `hard_ood` | `ν <= 0.027028120493367818`, training quartile, not refit |
| complement | the other instances in the scored split (`ν > threshold`) |
| test counts | 128 full, 30 `hard_ood`, 98 complement. These counts were read from the manifest before training |

Relative L2 is the Stage 3 ratio `||prediction − target|| / ||target||` on the flattened window. `u` uses the training mean and population standard deviation in [docs/stage_a/pilot_manifest.json](docs/stage_a/pilot_manifest.json): mean `-4.623541922284834e-19`, std `0.37254954772040344`. Viscosity enters as one extra channel, normalized by the training-split mean `0.05185957718021863` and population std `0.027733196224510124`. Validation and test viscosities do not enter those two numbers.

The one-step baseline repeats the last input frame across the eight target frames. It is not trained. The same windows are used for the network and the baseline. Because every instance has 11 windows, the mean over windows equals the mean of the per-instance window means.

An autoregressive rollout is recorded and is not used to choose the epoch. The true frames 0 through 7 are the first input. Each predicted block is then the next input. The open-loop persistence baseline holds frame 7 for the whole predicted horizon (frames 8 through 95). That horizon baseline is weak on Burgers, because the field moves. The primary comparison is the one-step score against the one-step persistence baseline.

## Pilot

`artifacts/burgers_hard_pilot/` is gitignored. This workspace regenerated it with the Stage A command. The solver-config SHA-256 was `955dc2f9687ae7131737e37c0b6dc2f790c70ebfcc2b8e67ffca30f29de28bd6`, the same value as [STAGE_A_REPORT.md](STAGE_A_REPORT.md).

```bash
python -m pinnforge.reference.numerical hard-pilot --output artifacts/burgers_hard_pilot
```

Training read train and validation only. Slice scoring read the test split. Every loaded file was checked against `field_sha256` in the committed manifest. All 512 train, 128 validation, and 128 test fields matched. Counts after windowing: 5632 train windows, 1408 validation windows, 1408 test windows.

## Commands

```bash
python -m pinnforge.operator train \
  --protocol docs/v02/stage_b_train_protocol.json \
  --pilot artifacts/burgers_hard_pilot \
  --manifest docs/stage_a/pilot_manifest.json \
  --output runs/stage_b_fno/seed_0 \
  --epochs 30 --batch-size 32 --lr 0.001 \
  --width 32 --modes 16 --layers 4 \
  --input-frames 8 --output-frames 8 --stride 8 \
  --seed 0 --loss data
```

Repeat with `--seed 1`, `2`, `3`, and `4`, and with `--output runs/stage_b_fno/seed_<seed>`. `--protocol` rejects a different width, mode count, window, epoch budget, seed, loss, or pilot path. `runs/` is gitignored. The checkpoints are not part of the commit.

```bash
python -m pinnforge.operator slices \
  --pilot artifacts/burgers_hard_pilot \
  --manifest docs/stage_a/pilot_manifest.json \
  --protocol docs/v02/pilot_protocol.json \
  --train-protocol docs/v02/stage_b_train_protocol.json \
  --checkpoint runs/stage_b_fno/seed_0/checkpoint.pt \
  --batch-size 32 \
  --output runs/stage_b_fno/seed_0/slices.json

python -m pinnforge.operator aggregate \
  --protocol docs/v02/stage_b_train_protocol.json \
  --inputs runs/stage_b_fno/seed_{0,1,2,3,4}/slices.json \
  --output docs/v02/stage_b_scores.json
```

`pinnforge fno train`, `pinnforge fno slices`, and `pinnforge fno aggregate` call the same commands. The aggregate file is [docs/v02/stage_b_scores.json](docs/v02/stage_b_scores.json).

## Seeds

`wall_clock_seconds` is the fit after the windows are loaded. It is not a hardware benchmark. Seed 0 shared the machine with another seed for most of the run, which is why its clock is longer. The weights do not depend on that clock.

| seed | selected epoch | validation mean relative L2 | full test | `hard_ood` | complement | fit seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 28 | 3.98207875714179e-03 | 4.037062210534456e-03 | 6.781261847096676e-03 | 3.197001097301125e-03 | 2074.10 |
| 1 | 28 | 3.897175648928502e-03 | 3.981297744231881e-03 | 6.96237048208319e-03 | 3.068724457134541e-03 | 939.17 |
| 2 | 29 | 4.8875304865287515e-03 | 4.891482743580779e-03 | 7.566978064983732e-03 | 4.072453563559467e-03 | 766.72 |
| 3 | 26 | 4.319636697915502e-03 | 4.303186504526355e-03 | 7.027877414485406e-03 | 3.469097450457259e-03 | 560.42 |
| 4 | 25 | 4.274131299715281e-03 | 4.281821423698064e-03 | 6.907776008570202e-03 | 3.4779577752678185e-03 | 560.75 |

Validation relative L2 is not monotone in the epoch. Epoch 30 was not selected for any seed. Seed 2 is the high value on validation, on the full test, and on the complement. It stays in the mean. No seed hit a non-finite loss, and none was replaced.

Selected-epoch mean across seeds: 27.2 ± 1.6431676725154984. Validation mean relative L2: 4.272110578045965e-03 ± 3.891333760268412e-04.

## Test scores

Mean ± sample standard deviation across the five seeds. Persistence does not depend on the seed. Its sample standard deviation on the one-step scores is 0.

| slice | instances | windows | mean relative L2 | one-step persistence |
| --- | ---: | ---: | ---: | ---: |
| full test | 128 | 1408 | 4.298970125314308e-03 ± 3.608651258270467e-04 | 8.456701056846332e-02 |
| `hard_ood` | 30 | 330 | 7.049252763443842e-03 ± 3.032654479273341e-04 | 9.782188328427069e-02 |
| complement | 98 | 1078 | 3.457046868744042e-03 ± 3.865637014640921e-04 | 8.050939647178759e-02 |

The ratio of the `hard_ood` mean to the complement mean is 2.039. Per seed, that ratio is 2.121, 2.269, 1.858, 2.026, and 1.986 (mean 2.052 ± 0.154). The gap is in the same direction on every seed. It was not used to choose the model or the threshold.

Against one-step persistence, the full-test mean is 0.0508 of the baseline (persistence is 19.7 times larger), the `hard_ood` mean is 0.0721 of its baseline (13.9 times), and the complement mean is 0.0429 of its baseline (23.3 times). The network beats persistence on both slices. The margin is smaller on `hard_ood`.

On every seed, all 1408 test windows have relative L2 below the one-step persistence score for that window. The largest model-to-persistence ratio on a single window is 0.220, on seed 1. There is no test window in this run that loses to persistence. That is not a claim about a longer horizon or a different baseline.

| slice | median relative L2 | pooled relative L2 | normalized MSE |
| --- | ---: | ---: | ---: |
| full test | 3.272448955762990e-03 ± 4.613228924667977e-04 | 6.355987459278239e-03 ± 1.945319206669872e-04 | 3.655647630349448e-05 ± 2.275867775350810e-06 |
| `hard_ood` | 5.935098578901166e-03 ± 5.379087153609289e-04 | 8.990321295806170e-03 ± 1.230238933814690e-04 | 1.065753834216869e-04 ± 2.911643105729508e-06 |
| complement | 2.886673364041410e-03 ± 4.108704544444743e-04 | 4.399259509160329e-03 ± 3.497785160735466e-04 | 1.512211698159883e-05 ± 2.473791855935634e-06 |

The median sits below the mean, and the pooled score sits above the mean, on every slice. A few windows carry more of the squared error than a typical window. Normalized MSE is in the manifest-normalized `u` used for training. It is not the selection metric.

The Stage 3 easy-pilot test mean, one seed, was `0.002647639938017037` at `N = 256` with viscosities at least `0.02` and initial modes `1..4`. The full-test mean here is larger by a factor of about 1.62. That factor is not a paired comparison. The grid, the initial family, the viscosity floor, and the number of seeds all changed. The Stage 3 number is not a multi-seed result.

The Stage A label gate is relative L2 `1e-9` for the spectral solver against a finer integration. The full-test mean above is about `4.3e6` times that gate. It is not a solver error.

## Rollout

Secondary. Not used for selection. Mean ± sample standard deviation of the per-instance relative L2 on frames 8 through 95, after feeding predictions back as inputs.

| slice | rollout mean relative L2 | open-loop persistence |
| --- | ---: | ---: |
| full test | 1.511710621402608e-02 ± 1.852906885827387e-03 | 9.766351855045150e-01 |
| `hard_ood` | 2.585120197526468e-02 ± 1.822169082594636e-03 | 9.268796039417675e-01 |
| complement | 1.183115853201427e-02 ± 1.866801285523518e-03 | 9.918664859829069e-01 |

Open-loop persistence holds one early frame across 0.88 time units, so a score near 1 is the expected trivial forecast. It is a weak baseline. The informative comparison is the rollout against the one-step score: about 3.52 times the one-step full-test mean, 3.67 times on `hard_ood`, and 3.42 times on the complement. Error accumulates. It does not jump to the persistence floor.

The five-seed mean of the per-step relative L2 on `hard_ood` increases at every step. Step 0 is the block at `t = 0.08` through `0.15` (this step still sees the true input). Step 10 is `t = 0.88` through `0.95`. The step table is rounded. The unrounded curves are in the aggregate file.

| step | `t` of the block | `hard_ood` mean ± std | complement mean ± std |
| ---: | --- | ---: | ---: |
| 0 | 0.08–0.15 | 1.285495e-02 ± 2.311e-04 | 7.758063e-03 ± 3.524e-04 |
| 1 | 0.16–0.23 | 2.004241e-02 ± 1.954e-03 | 9.147699e-03 ± 8.863e-04 |
| 2 | 0.24–0.31 | 2.383586e-02 ± 2.827e-03 | 9.967699e-03 ± 1.400e-03 |
| 3 | 0.32–0.39 | 2.609532e-02 ± 2.542e-03 | 1.080306e-02 ± 1.724e-03 |
| 4 | 0.40–0.47 | 2.758842e-02 ± 2.296e-03 | 1.176758e-02 ± 1.977e-03 |
| 5 | 0.48–0.55 | 2.885667e-02 ± 2.370e-03 | 1.280214e-02 ± 2.242e-03 |
| 6 | 0.56–0.63 | 2.990703e-02 ± 2.704e-03 | 1.387594e-02 ± 2.565e-03 |
| 7 | 0.64–0.71 | 3.090413e-02 ± 3.078e-03 | 1.498399e-02 ± 2.968e-03 |
| 8 | 0.72–0.79 | 3.206356e-02 ± 3.293e-03 | 1.613138e-02 ± 3.448e-03 |
| 9 | 0.80–0.87 | 3.346180e-02 ± 3.389e-03 | 1.732404e-02 ± 3.995e-03 |
| 10 | 0.88–0.95 | 3.470331e-02 ± 3.496e-03 | 1.856417e-02 ± 4.594e-03 |

The seed spread also grows toward the end of the horizon. The full curves are in the aggregate file. The first rollout step matches the start-0 supervised window; the scorer rejects a checkpoint if those two fields disagree by more than `1e-5` in physical units.

## Where the error sits

The five worst `hard_ood` instances by one-step mean relative L2 are the same five ids on every seed: `179`, `619`, `338`, `185`, and `205`, with `179` first on every seed. Instance `179` has the smallest test viscosity in the manifest, `ν = 0.00659586199923937`. Seed 0, which is not the worst seed:

| instance | `ν` | one-step mean relative L2 | one-step persistence |
| ---: | ---: | ---: | ---: |
| 179 | 0.00659586199923937 | 0.013307020107418 | 0.12241994740816874 |
| 619 | 0.007207675915965656 | 0.01129369255393809 | 0.12361661551074345 |
| 338 | 0.007689406142113342 | 0.011081465887574114 | 0.1331177246602292 |
| 185 | 0.006878265076856729 | 0.010945816784277014 | 0.12891280066503621 |
| 205 | 0.011486023251651835 | 0.01009197925020616 | 0.11322923235224948 |

`619`, `338`, and `185` are three of the nine Stage A `sensors32_bursts` failures. The one-step errors here are about 0.01, not a collapse, and they still beat persistence. The rollout ranking is not stable: the worst five rollout instances are a different set on each seed. Several Stage A inverse-failure ids appear in some of those lists. That overlap is a description of these five seeds. It is not a ranking of the inverse problem, and it was not used to change the threshold.

The complement is not uniformly easy. Instances `55` (`ν = 0.08545530975129856`), `385` (`ν = 0.03321160766611633`), and `511` (`ν = 0.09412760255951814`) are in the worst five complement windows on every seed. On seed 0, instance `55` has one-step mean relative L2 `0.008084616015775732`, which is above the `hard_ood` mean. High viscosity does not by itself make a window easy, and `hard_ood` is not the set of the individually worst forecasts. The slice mean is still higher on `hard_ood` on every seed. The threshold was not moved to chase that.

## Inverse stress

No inverse model was trained. The Stage 5 pattern `sensors32_bursts` was not retuned. Every field loaded for training or for the slice scores matched `field_sha256` in the Stage A manifest, so the trajectories are the ones already scored in [docs/v02/inverse_stress.json](docs/v02/inverse_stress.json).

On that record, `sensors32_bursts` over the 128 test instances has mean absolute error `0.0030159391726451274` and mean relative error `0.1171354466174288`, about 49.5 and 88.1 times the Stage 5 easy-pilot table. Nine instances fail the `0.5` relative-error rule. All nine have `ν <= 0.027028120493367818`. The complement has zero failures. Dense residual least squares on the full grid does not cross that rule, including on `hard_ood`.

The data-only FNO does not remove that stress. It never sees the inverse objective. Forecasting `u` with `ν` given is a different task from recovering `ν` from 32 sensors.

## What is not claimed

- No hybrid loss, no residual loss, and no learned inverse. Those are later stages. The checkpoint is data MSE only.
- No retuning of `hard_ood`, of the Stage A protocol, of the sensor mask, or of the Stage 3 width, modes, depth, window, epoch budget, or learning rate. Sixteen Fourier modes do not cover the 48-mode initial band. This run does not measure how much of the error that truncation causes.
- No single-seed result. Seed 2 is worse than the mean on the full test and on the complement, and it is included. The sample standard deviation uses `ddof = 1` on five seeds.
- No claim that the one-step error, or the rollout error, meets the `1e-9` label gate.
- No claim that the factor of about 1.62 against the Stage 3 easy-pilot number is a controlled ablation. That number is one seed on a different grid and a different ensemble.
- The open-loop persistence score near 1 is not evidence that the rollout is accurate. The one-step persistence baseline is the primary comparison. The rollout is about 3.5 times the one-step error.
- Beating persistence on every test window in these five seeds is not a claim about other horizons, other baselines, or other initial families.
- `hard_ood` raises the slice mean. It is not a list of the worst individual windows. Complement instance `55` is a counterexample, and the threshold was not edited to remove it.
- The inverse failures remain the Stage A sensor-mask failures. Dense least squares still stays under the `0.5` rule. This network does not estimate `ν`.
- The coordinate-PINN Burgers reference is still `NotImplementedError`. These files do not make `pinnforge eval` return a field error.
- The version string is still `0.1.0`. This is not a `0.2.0` release.

## Tests

```bash
python -m ruff check .
python -m pytest -q --tb=line
```

In this workspace, after the slice command and the protocol checks were in place: ruff reported `All checks passed!`, and pytest reported `363 passed, 6 skipped`. The six skips are the no-torch cases in `tests/test_ml_import.py`, which are skipped because torch is installed. `tests/test_stage_b.py` checks that the committed protocol is data-only, that its threshold matches the Stage A protocol, that a protocol file containing scores is rejected, that the harder manifest loads, and that the manifest slice counts match the locked rule.
