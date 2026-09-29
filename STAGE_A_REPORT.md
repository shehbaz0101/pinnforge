# Stage A report

Harder periodic Burgers pilot for v0.2, on `feat/v02-stage-a-harder-pilot`, based on `c607c8e515e7f2f582c87a5bb50a48c9996ddd2d` (Stage 5). The version string stays `0.1.0`. These runs used Python 3.12.3 and NumPy 2.4.4. They extend the Stage 2 reference with a second dataset. They do not train an operator and they do not replace the Stage 2 pilot. The Stage 5 closed-form residual estimator is scored on this dataset as a ceiling/floor reference. That observation pattern is not retuned.

`pinnforge eval` and `pinnforge demo --equation burgers` still score residual metrics. `pinnforge.reference.burgers.reference_solution` still raises `NotImplementedError`. `pinnforge fno` still reads `docs/stage2/pilot_manifest.json`. The harder trajectories are produced by `python -m pinnforge.reference.numerical hard-pilot`.

## Task

The equation, the domain, and the solver stack are the Stage 2 problem:

    u_t + u u_x = ν u_xx,    x ∈ [-1, 1],    t ∈ [0, 1],

periodic in `x`, length `L = 2`. The method is the dealiased (3/2) Fourier spectral discretization with ETDRK4 in float64, documented in [STAGE2_REPORT.md](STAGE2_REPORT.md). SHA-256 of `solver.py`, `initial.py`, `dataset.py`, and `checks.py` match the Stage 2 manifest, so those four files are the same bytes as that pilot.

What changes is the ensemble. Stage 2 draws modes `1..4` with amplitudes `Uniform(-1, 1) / m` and `ν ~ Uniform(0.02, 0.10)`, and stores labels at `N = 256`, `dt = 1e-3`. On that family, `ν = 0.005` missed the `1e-8` gate. This pilot draws a steeper, wider-band initial condition, draws `ν` down to `0.005`, and stores labels on a finer grid that meets a tighter gate.

The time horizon stays `[0, 1]`. On the steep probe below, `||u||_∞` over the saved frames equals the value at `t = 0`, so the explicit-advection step guide does not get tighter as the front forms. The binding constraint is spatial resolution at low viscosity, which the label grid is chosen to meet. A longer interval was not required for that gate.

## Initial conditions

The family is `tanh_bandlimited`. The generator is PCG64 from `SeedSequence([master_seed, instance_id, 0x48415244])`. The salt is the ASCII bytes of `HARD` and is not the Stage 2 salt `0x42555247`. For modes `m = 1 .. 8`, in that order,

    a_m = Uniform(-1, 1) / sqrt(m),
    b_m = Uniform(-1, 1) / sqrt(m).

Let `p` be that trigonometric polynomial. On a fixed canonical grid of 8192 nodes,

    p_hat = (p - mean(p)) / max|p - mean(p)|,
    s = tanh(3 * p_hat).

`s` is projected onto Fourier modes `|m| <= 48`, the mean mode is set to 0, and the result is divided by its maximum absolute value on the canonical grid. The stored field is that trigonometric polynomial. A solver grid needs `N/2 > 48`. A draw whose canonical peak is below `1e-8` is redrawn, up to 8 attempts. Viscosity is drawn only after a successful field:

    ν ~ Uniform(0.005, 0.10).

The pilot master seed and split seed are both `20260929`.

### Sharpness against the Stage 2 family

Same instance ids `0 .. 63`, each family's own generator, slope and energy on `N = 256`. High-mode energy is the fraction of Fourier energy in `|m| > 4`. Stage 2 initial data is exactly modes `1..4`.

| family | slope min | slope median | slope max | energy in `|m| > 4`, median |
| --- | ---: | ---: | ---: | ---: |
| `tanh_bandlimited` | 15.92465428951044 | 32.47562954227386 | 49.71872944994811 | 0.22656425949456738 |
| Stage 2, same ids | 3.6797268844417372 | 5.796893120289845 | 9.608788259972254 | 6.88176697651503e-32 |

The median slope is larger by a factor of about 5.60. The median harder field keeps about 23% of its energy above mode 4. The largest Stage 2 fraction above mode 4 in this pool is `1.2891202787728367e-31`. The three steepest harder draws are instance ids `2`, `60`, and `49`, with slopes `49.71872944994811`, `49.23961874096104`, and `49.080740421480954`.

## Label resolution and the step guide

For `||u||_∞ <= 1` and `N = 1024` the explicit-advection guide from Stage 2 is

    dt <= 0.5 / (||u||_∞ * k_max) = 0.0003114578142698539,

with `k_max = π (N/2 - 1)`. The label step is `dt = 2.5e-4`, which is inside that guide. `save_dt = 0.01` gives 101 frames, the same count as Stage 2. This is a stability guide. The accuracy statement is the convergence section.

`dt = 5e-4` at `N = 1024` is outside the guide. It stayed finite on the steep probe and is reported in the temporal table. It is not the label step.

## Convergence

Command:

```bash
python -m pinnforge.reference.numerical hard-convergence --output docs/stage_a
```

Raw rows are in [docs/stage_a/convergence.json](docs/stage_a/convergence.json). Plots: [spatial](docs/stage_a/spatial_convergence.svg), [temporal](docs/stage_a/temporal_convergence.svg), [invariants](docs/stage_a/invariants.svg).

Relative L2 is `||u - v||_2 / ||v||_2` on the shared samples. Spatial comparisons restrict the fine field by truncating its FFT onto the coarse modes. Viscosities in this section are forced. They are not the random `ν` stored for that instance id in the pilot. The reference trajectory is `N = 2048`, `dt = 1.25e-4`, unless a row says otherwise. Saved frames are every `0.01`.

The study took 57.34 seconds in this workspace.

### Spatial, steepest initial condition

Instance `2`, `ν = 0.005`, `dt = 1.25e-4`, against `N = 2048` at the same step. The `N = 2048` row is that field restricted onto itself, so it is roundoff of the truncation, not a second integration.

| N | final-time relative L2 | 101-frame relative L2 |
| ---: | ---: | ---: |
| 256 | 1.72645411855589e-09 | 2.8502216886891648e-05 |
| 512 | 1.6368366786255006e-15 | 1.2341909963380749e-08 |
| 1024 | 5.48579968885302e-16 | 6.009899138090332e-15 |
| 2048 | 2.2441229179989656e-16 | 2.368515830310347e-16 |

At `N = 256` the error is in the saved transient, not at the final time: the 101-frame figure is about `2.85e-5` while the final time is about `1.73e-9`. `N = 512` brings the final time to roundoff and leaves the 101-frame error at about `1.23e-8`. `N = 1024` matches `N = 2048` to about `6e-15` on the saved frames.

### Temporal

Instance `2`, `ν = 0.005`, `N = 1024`, against `dt = 1.25e-4` on the same grid.

| dt | inside the `N = 1024` step guide | final-time relative L2 | 101-frame relative L2 | final-time ratio to the next row |
| ---: | --- | ---: | ---: | ---: |
| 5e-4 | no | 2.820541244856929e-11 | 7.155102942468685e-10 | 15.43 |
| 2.5e-4 | yes | 1.827901599686692e-12 | 4.2657802648582265e-11 | |
| 1.25e-4 | yes | 2.395877234206081e-16 | 2.411697110487644e-16 | |

The first halving is close to the factor 16 of a fourth-order step (15.43 at the final time, 16.77 on the 101 frames). The next halving is at the floor of this comparison.

### Coarser grids against the fine run

Same initial condition, `ν = 0.005`, against `N = 2048`, `dt = 1.25e-4`.

| N | dt | final-time relative L2 | 101-frame relative L2 | passes the `1e-9` gate |
| ---: | ---: | ---: | ---: | --- |
| 256 | 1e-3 | 1.6634370423514708e-09 | 2.8501942111259604e-05 | no |
| 512 | 5e-4 | 2.8205936503266273e-11 | 1.236292642275466e-08 | no |

`N = 256`, `dt = 1e-3` is the Stage 2 label grid. Its 101-frame error matches the spatial row at `N = 256`, so the miss is the grid, not that particular time step. `N = 512` still misses `1e-9` on the saved frames (`1.24e-8`), which is why the labels use `N = 1024`.

### Label settings against the fine run

`N = 1024`, `dt = 2.5e-4`, `ν = 0.005`, against `N = 2048`, `dt = 1.25e-4`. Instance `0` is the first draw. Instances `2`, `60`, and `49` are the three steepest slopes in ids `0 .. 63`.

| instance | final-time relative L2 | 101-frame relative L2 |
| ---: | ---: | ---: |
| 0 | 2.117081824213112e-12 | 3.801930538735817e-11 |
| 2 | 1.828228731884847e-12 | 4.265780171940469e-11 |
| 60 | 3.3892227954420676e-12 | 3.588656385666115e-11 |
| 49 | 2.044907765158085e-12 | 3.386580823903016e-11 |

The largest 101-frame figure in this table is about `4.27e-11`.

### Viscosity probe

Instance `2`, label grid against the fine run.

| ν | final-time relative L2 | 101-frame relative L2 | inside Stage 2 `[0.02, 0.10]` | passes `1e-9` |
| ---: | ---: | ---: | --- | --- |
| 0.10 | 2.7696944872446936e-10 | 1.8455559810329018e-10 | yes | yes |
| 0.05 | 5.007981336015978e-11 | 3.5717942693580205e-11 | yes | yes |
| 0.02 | 5.384332235417548e-12 | 7.592680434851888e-12 | yes | yes |
| 0.01 | 1.0556704396781642e-12 | 7.256218409951246e-12 | no | yes |
| 0.005 | 1.828228731884847e-12 | 4.265780171940469e-11 | no | yes |

The largest relative error in this probe is the final-time value at `ν = 0.10`, about `2.77e-10`. That is still under `1e-9`. The coarse-grid failure that forces `N = 1024` is at `ν = 0.005`, where the saved-frame error on `N = 256` is about `2.85e-5`.

### Mean and energy

Rectangle-rule mean and energy, as in Stage 2. Steep instance `2` at `ν = 0.005`, label settings `N = 1024`, `dt = 2.5e-4`, samples every `0.01`:

| quantity | value |
| --- | ---: |
| max abs mean drift | 3.469446951953614e-17 |
| energy at t = 0 | 0.4743790244210311 |
| energy at t = 1 | 0.05061633506269626 |
| max energy increase | 0 |
| max abs at t = 0 | 0.9999637483356468 |
| max abs at t = 1 | 0.46900333481713924 |
| max abs over saved frames | 0.9999637483356468 |

The `N = 2048`, `dt = 1.25e-4` run has max abs mean drift `5.551115123125783e-17`, max energy increase `0`, and final energy `0.05061633506261706`. The saved maximum stays at the initial maximum in both runs.

Every generated pilot file is included in the manifest invariant summary, not a 12-file spot check. Across all 768 trajectories the max abs mean drift is `1.3877787807814457e-16` (instance `75`) and the max energy increase is `0`.

## Preregistered label tolerance

Written with this pilot, before any operator is trained on it.

A label at `N = 1024`, `dt = 2.5e-4`, `save_dt = 0.01`, float64, 3/2 dealiasing, ETDRK4, for this initial-condition family and `ν ∈ [0.005, 0.10]`, is acceptable when all of the following hold against an `N = 2048`, `dt = 1.25e-4` integration of the same initial data:

- final-time relative L2 ≤ `1e-9`
- space-time relative L2 on the 101 saved frames ≤ `1e-9`
- max abs mean drift ≤ `1e-12`
- max energy increase on the saved samples ≤ `1e-10`

`1e-9` is a tighter relative-L2 gate than the Stage 2 gate of `1e-8`. The measured label-versus-reference errors on the four instances above are at most `3.39e-12` and `4.27e-11`. The viscosity probe stays at or below `2.77e-10`. Mean drift and energy increase on the full generated set pass. This tolerance is the gate for these labels. It is not a twin `N = 2048` integration of every one of the 768 files.

## Pilot dataset

Estimate, from the field sizes:

```text
instances=768 times=101 n=1024 bytes_per_trajectory=827392 field_mebibytes=606.000
```

A batch of 8 trajectories at this resolution took about 5.88 seconds in a probe, which extrapolates to about 564 seconds for 768 trajectories.

Actual generation in this workspace: **571.79 seconds**. The directory `artifacts/burgers_hard_pilot/` is **608,577,567 bytes** (580.38 MiB), including npz wrappers and the manifest. Raw fields were estimated at 635,437,056 bytes. The arrays are listed in `.gitignore`. The manifest is committed at [docs/stage_a/pilot_manifest.json](docs/stage_a/pilot_manifest.json).

```bash
python -m pinnforge.reference.numerical hard-estimate
python -m pinnforge.reference.numerical hard-pilot --output artifacts/burgers_hard_pilot
```

The second command refuses to overwrite an existing directory. Defaults are the preregistered pilot: 512 / 128 / 128, `N = 1024`, `dt = 2.5e-4`, `save_dt = 0.01`, `t_final = 1`, seeds `20260929`. The Stage 2 command is unchanged:

```bash
python -m pinnforge.reference.numerical pilot --output artifacts/burgers_pilot --train 512 --val 128 --test 128
```

| item | value |
| --- | --- |
| counts | 512 train, 128 validation, 128 test |
| `N`, `dt`, `save_dt`, `t_final` | 1024, 0.00025, 0.01, 1 |
| master seed, split seed | 20260929, 20260929 |
| viscosity interval | `[0.005, 0.10]` |
| solver config SHA-256 | `955dc2f9687ae7131737e37c0b6dc2f790c70ebfcc2b8e67ffca30f29de28bd6` |
| pilot config SHA-256 | `c0b97e2be1373233be4286f99e280011efb8ca8e905dffa5abe1bbd3b35753a0` |
| training `u` mean | -4.623541922284834e-19 |
| training `u` std (ddof 0) | 0.37254954772040344 |
| training value count | 52953088 |
| drawn `ν` min, max, mean, std | 0.005035307689910299, 0.09988823841997412, 0.052235219499597675, 0.027786002046762036 |
| training `ν` mean, std | 0.05185957718021863, 0.027733196224510124 |
| test `ν` min, max | 0.00659586199923937, 0.0972891142416336 |

The realized minimum `ν` is below the Stage 2 floor of `0.02`, including in the test split (`0.00659586199923937`). The split is a Fisher–Yates permutation of instance ids `0 .. 767` from PCG64 and `SeedSequence([split_seed, 0x53504C54])`, the same algorithm as Stage 2 with this pilot's split seed. It is computed before integration. No windows are extracted. Stored arrays are raw solver output. The training mean and standard deviation above are fit on the training files only and are not applied to the arrays. Each instance records an npz SHA-256 and a `field_sha256` of the contiguous float64 field, plus the saved-frame mean drift and energy increase.

Code SHA-256 values in the manifest:

| file | SHA-256 |
| --- | --- |
| `solver.py` | `aa3acbe14a4474bd3069ff1a4a0e99b3ee38dc5fa66cc596210ca0a051d5095b` |
| `initial.py` | `0a8b2f910bcb71d1be3993eb810cd805e9bca97d23f63404749695ef983b9ba0` |
| `dataset.py` | `a2661ff8599ffc8f86d71daa4d58ebca84deb0e5741c4c166a194e2aa7db1cea` |
| `checks.py` | `0f124f1eacd1f558e606d29ea44ae7cdec2e7497bd36e33d5828bd121f5a5361` |
| `harder.py` | `798764d3e18ec2899df6453b65b69385fdfcc6fb53fb2239d83eeb7efcf97b19` |

The first four match [docs/stage2/pilot_manifest.json](docs/stage2/pilot_manifest.json).

## Preregistered protocol

[docs/v02/pilot_protocol.json](docs/v02/pilot_protocol.json) is the same object as `protocol` in [docs/stage_a/pilot_manifest.json](docs/stage_a/pilot_manifest.json). It was written before any operator training. `frozen_before_operator_training` is true and `retuned_after_inverse_measurement` is false.

The first locked family already made the Stage 5 estimator miss the easy-pilot table, so the viscosity floor, `β`, the 48-mode cutoff, the horizon, and the seeds were not changed after that measurement. `hard_ood` is the lowest quartile of the training viscosities. It is not a cutoff fit to the test errors.

| item | value |
| --- | --- |
| format | `pinnforge.burgers_hard_pilot_protocol.v1` |
| IC family | `tanh_bandlimited` |
| `ν` | uniform on `[0.005, 0.10]`, drawn after a successful field |
| seeds | master `20260929`, split `20260929`, IC salt `0x48415244`, split salt `0x53504C54`, PCG64 |
| splits | 512 / 128 / 128 problem instances |
| label gate | final and space-time relative L2 ≤ `1e-9`, mean drift ≤ `1e-12`, energy increase ≤ `1e-10`, at `N = 1024`, `dt = 2.5e-4` |
| horizon | `t ∈ [0, 1]`, periodic `x ∈ [-1, 1]` |
| `hard_ood` | `ν <= threshold_nu`, quantile `0.25`, method `linear`, fit on train only |
| `threshold_nu` | `0.027028120493367818` |
| training instances at or below the threshold | 128 of 512 |
| Stage 2 floor, excluded from that pilot | `0.02` |

Later stages must score `hard_ood` separately. They must not use `hard_ood`, validation, or the test split to choose a model or to refit the quantile.

## Inverse stress against the Stage 5 residual estimator

The estimator is `sensors32_bursts` (`PREREGISTERED_OBSERVATION` in `pinnforge.operator.inverse`): 32 equispaced sensors and four bursts of five saved frames, `Δt = 0.01`. `ν̂ = (a · b) / (b · b)` from the Stage 4 central residual. The failure rule is unchanged: `ν̂ <= 0` or relative error `> 0.5`. The baseline is the harder pilot's training-split mean, `0.05185957718021863`. Stage 5 numbers below are the easy-pilot test table in [STAGE5_REPORT.md](STAGE5_REPORT.md). They were not recomputed. The harder scores are [docs/v02/inverse_stress.json](docs/v02/inverse_stress.json), from

```bash
python -m pinnforge.reference.numerical hard-stress \
  --pilot artifacts/burgers_hard_pilot \
  --manifest docs/stage_a/pilot_manifest.json \
  --output docs/v02/inverse_stress.json
```

On all 128 harder test instances, `sensors32_bursts`:

| quantity | Stage 5 easy pilot | harder pilot | ratio |
| --- | ---: | ---: | ---: |
| mean absolute error | 6.091698430189056e-05 | 0.0030159391726451274 | 49.50900323132916 |
| mean relative error | 1.3303134076090104e-03 | 0.1171354466174288 | 88.051015608388 |
| max relative error | 1.8379471956251635e-02 | 0.9497873814639219 | |
| failures | 0 | 9 | |
| nonpositive `ν̂` | 0 | 0 | |
| correlation | 0.9999942242288621 | 0.9975830065148409 | |
| worse than baseline | 0 | 2 | |

The harder training-mean baseline has mean absolute error `0.024858320887062635` and mean relative error `0.96585356303953`. The sparse estimator still beats that constant on 126 of 128 test instances. It does not stay near the Stage 5 errors.

`hard_ood` on the test split is the 30 instances with `ν <= 0.027028120493367818`. The complement has 98. Twenty test instances have `ν < 0.02`, which Stage 2 never drew.

| slice | n | mean absolute error | mean relative error | max relative error | failures | mean-abs ratio vs Stage 5 | mean-rel ratio vs Stage 5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `hard_ood` | 30 | 0.003878099063123072 | 0.3494591802034973 | 0.9497873814639219 | 9 | 63.66203297103654 | 262.6893619238077 |
| complement | 98 | 0.0027520126755600427 | 0.04601593633597921 | 0.1965295204205419 | 0 | 45.17644310692894 | 34.59029734856559 |
| `ν < 0.02` | 20 | 0.004485516759132618 | 0.46966295632108307 | 0.9497873814639219 | 9 | 73.63327010581202 | 353.04684868599077 |

All 9 failures are inside both `hard_ood` and `ν < 0.02`. The complement has none. Its mean relative error is still about 35 times the Stage 5 table.

The dense residual least squares, every saved frame and every grid point, is the stencil ceiling. It does not cross the `0.5` rule.

| slice | n | mean absolute error | mean relative error | max relative error | failures | mean-abs ratio vs Stage 5 dense | mean-rel ratio vs Stage 5 dense |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all test | 128 | 0.0011637263530198242 | 0.01585009767787492 | 0.04757213144149604 | 0 | 41.87605531740426 | 32.15240892944848 |
| `hard_ood` | 30 | 2.677073678153862e-05 | 0.0013208386837906374 | 0.0037678982358051883 | 0 | 0.9633302979195169 | 2.67936175247375 |

Stage 5 dense mean absolute error is `2.778977972493421e-05` and mean relative error is `4.9296765640965002e-04`. On `hard_ood`, the full grid recovers `ν` about as tightly in absolute error as that easy-pilot dense table. The `0.5`-rule failures are the 32-sensor mask.

At the saved frame of maximum `||u_x||_∞`, the relative discrete L2 between spectral `u_xx` on those 32 sensors and the full-grid spectral `u_xx` restricted to the same nodes has test median `0.4936308953893701` and test max `0.9081332472259828`. The `hard_ood` median is `0.4619794527910751`, which is not higher than the test median. The nine failures have median `0.7299963656673164`, min `0.3235601430854653`, and max `0.9081332472259828`. On those same frames the energy fraction of `u` in modes `|m| > 16` stays between `0.0010222675615028708` and `0.007099447870041589`. The stored field is not broadband above mode 16. The second derivative on the 32-point sensor grid is.

### Failure examples

All nine `sensors32_bursts` failures. `ν̂` is above `ν` on every row. `t` is the saved time of maximum `||u_x||_∞`. `alias` is the sensor `u_xx` relative L2 defined above. Rows are sorted by relative error.

| instance | `ν` | `ν̂` | absolute error | relative error | `t` | `||u_x||_∞` | alias |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 530 | 0.007820793287511035 | 0.015248884065026759 | 0.007428090777515724 | 0.9497873814639219 | 0.1 | 42.78773275250532 | 0.42900313273292634 |
| 743 | 0.0083563697313075 | 0.016182596801070422 | 0.007826227069762923 | 0.9365582569236526 | 0.23 | 43.87666726169284 | 0.3856680362668411 |
| 338 | 0.007689406142113342 | 0.014568075410026552 | 0.00687866926791321 | 0.8945644358983865 | 0.12 | 47.28104982515455 | 0.733078525743896 |
| 619 | 0.007207675915965656 | 0.013406588781299553 | 0.006198912865333897 | 0.8600432285811773 | 0.08 | 59.956904983203685 | 0.9081332472259828 |
| 655 | 0.008799276647784498 | 0.01607168473866388 | 0.0072724080908793814 | 0.8264779460832664 | 0.08 | 39.1135666877222 | 0.8386900051585175 |
| 185 | 0.006878265076856729 | 0.011787275232935168 | 0.0049090101560784385 | 0.7136988908141626 | 0.97 | 29.80575071191261 | 0.8636400249562893 |
| 375 | 0.011375559120315824 | 0.017891711993736646 | 0.0065161528734208225 | 0.5728204481644777 | 0.05 | 33.71043214636371 | 0.5181249591576774 |
| 391 | 0.010141552360801248 | 0.01554559238101602 | 0.005404040020214771 | 0.5328612255755112 | 0.11 | 41.10103332378971 | 0.7299963656673164 |
| 83 | 0.011835399149504187 | 0.01803755746991715 | 0.006202158320412963 | 0.5240345713792666 | 0.03 | 32.46665679914736 | 0.3235601430854653 |

Two test instances are worse in absolute error than the training-mean baseline. Neither is a failure. Both sit near that mean, so the constant is already close.

| instance | `ν` | `ν̂` | absolute error | baseline absolute error | relative error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 408 | 0.052381762522667755 | 0.05531890375419753 | 0.0029371412315297787 | 0.000522185342449126 | 0.056071829012220735 |
| 590 | 0.05226664813017004 | 0.0535222770246144 | 0.0012556288944443605 | 0.00040707094995141163 | 0.024023520531051043 |

## Gates

| gate | result |
| --- | --- |
| Stage 2 solver files unchanged (SHA-256) | pass |
| Stage 2 pilot command defaults and config hash | pass (`1926d9a625136afca2af0d4dbc318cd77616ee0268ef80341e55c45daf156088`) |
| richer initial band than modes `1..4` | pass (median energy above mode 4 about 0.227) |
| steeper than the Stage 2 family on the same ids | pass (median slope about 5.60 times larger) |
| `ν` interval includes `0.005` and `0.10` | pass |
| label step inside the advection guide | pass |
| `N = 1024` / `2048` recorded, coarse `N = 256` and `512` miss `1e-9` | pass |
| halved time steps, order near 4 until the floor | pass |
| label versus fine run ≤ `1e-9` on the probed instances and viscosities | pass |
| mean drift and energy on all 768 files | pass |
| 512 / 128 / 128 manifest, seeds, field SHA-256s, `ν` stats | pass |
| protocol frozen, `hard_ood` = training quartile `0.027028120493367818` | pass |
| `sensors32_bursts` mean absolute and relative error > 10× the Stage 5 table | pass (about 49.5× and 88.1×) |
| `hard_ood` contains the `0.5`-rule failures; complement has none | pass (9 and 0) |
| dense least squares stays under the `0.5` rule | pass |
| FNO trained on this pilot | not in this stage |

## What this does not claim

- No Fourier neural operator is trained on these trajectories. Stage 2 through Stage 5 still refer to `docs/stage2/pilot_manifest.json`. The Stage 5 observation constants are unchanged.
- The inverse numbers are a closed-form residual least squares on a fixed mask. They are not a learned inverse, and they are not a search over sensor layouts.
- Nine of the 30 `hard_ood` test instances cross the `0.5` rule. Twenty-one do not. The slice is the training quartile, not a threshold chosen to maximize failures.
- The complement has no `0.5`-rule failure, and its mean relative error is still about 35 times the Stage 5 table. Stress is not confined to `hard_ood`.
- Dense least squares does not cross the `0.5` rule, including on `hard_ood`, where its mean relative error is `0.0013208386837906374`. The failures above are the 32-sensor mask. Full-test dense mean relative error is still about 32 times the Stage 5 dense table.
- `hard_ood` is not the highest-alias quartile. Its median sensor-`u_xx` alias is `0.4619794527910751`, below the test median `0.4936308953893701`. The nine failures have a higher alias median, `0.7299963656673164`.
- Energy of `u` above mode 16 on the failure frames is at most `0.007099447870041589`. That is not a shock on the `N = 1024` grid, and it is not a claim that the stored field is broadband past mode 16.
- The two worse-than-baseline instances, `408` and `590`, have `ν` near the training mean. They are not the low-viscosity failures.
- `1e-9` is the solver label gate. It is not a tolerance on `ν̂`. It was measured on instance `0`, the three steepest draws in ids `0 .. 63`, and a five-value viscosity probe on the steepest draw. It is not a second integration of all 768 labels. The invariant scan does cover all 768 files.
- No statement that `N = 1024` or `dt = 2.5e-4` is optimal outside this family, this viscosity interval, and `t ∈ [0, 1]`. `β = 3` and the 48-mode cutoff are part of the family definition. They were not searched again after the inverse measurement.
- `dt = 5e-4` at `N = 1024` is outside the step guide. One steep trajectory at `ν = 0.005` stayed finite. That step is not a label setting.
- The coordinate-PINN Burgers reference is still `NotImplementedError`. These files do not make `pinnforge eval` return a field error.
- The version string is still `0.1.0`. This is not a `0.2.0` release.
