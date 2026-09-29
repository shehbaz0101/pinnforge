# Stage A report

Harder periodic Burgers pilot for v0.2, on `feat/v02-stage-a-harder-pilot`, based on `c607c8e515e7f2f582c87a5bb50a48c9996ddd2d` (Stage 5). The version string stays `0.1.0`. These runs used Python 3.12.3 and NumPy 2.4.4. They extend the Stage 2 reference with a second dataset. They do not train an operator and they do not replace the Stage 2 pilot.

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
| `harder.py` | `41682d0d000421ac1cf1fef5c787d4cc85fe8d2db31e2002005a27e2fdb32474` |

The first four match [docs/stage2/pilot_manifest.json](docs/stage2/pilot_manifest.json).

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
| FNO or inverse viscosity on this pilot | not in this stage |

## What this does not claim

- No Fourier neural operator, no residual loss, and no inverse-viscosity fit on these trajectories. Stage 2 through Stage 5 still refer to `docs/stage2/pilot_manifest.json`.
- The `1e-9` gate was measured on instance `0`, the three steepest draws in ids `0 .. 63`, and a five-value viscosity probe on the steepest draw. It is not a second integration of all 768 labels. The invariant scan does cover all 768 files.
- No statement that `N = 1024` or `dt = 2.5e-4` is optimal outside this family, this viscosity interval, and `t ∈ [0, 1]`. `β = 3` and the 48-mode cutoff are part of the family definition, not a search over sharper profiles.
- `dt = 5e-4` at `N = 1024` is outside the step guide. One steep trajectory at `ν = 0.005` stayed finite. That step is not a label setting.
- The coordinate-PINN Burgers reference is still `NotImplementedError`. These files do not make `pinnforge eval` return a field error.
- The version string is still `0.1.0`. This is not a `0.2.0` release.
