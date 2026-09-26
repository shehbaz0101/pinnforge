# Stage 2 report

Periodic viscous Burgers reference on `feat/stage2-burgers-reference`, based on `17bcddec19930e79f6a57336f43b3c2d0ec793dc` (Stage 1 / PR #11). The version string stays `0.1.0`. These runs used Python 3.12.3 and NumPy 2.4.4. They are a numerical reference and a pilot set of trajectories. They are not an operator model.

`pinnforge eval` and `pinnforge demo --equation burgers` still score residual metrics. `pinnforge.reference.burgers.reference_solution` still raises `NotImplementedError`. The solver is `pinnforge.reference.numerical`.

## Method

The problem is

    u_t + u u_x = ν u_xx,    x ∈ [-1, 1],    t ∈ [0, 1],

with periodic boundary conditions in `x`. The length is `L = 2`. Grid nodes are `x_j = -1 + j L / N` for `j = 0, ..., N-1`. The point `x = 1` is identified with `x = -1` and is not stored.

Wave numbers follow the ordinary derivative on that interval:

    k_m = (2 π / L) m = π m,

in NumPy FFT order. `k_1 = π`. The Nyquist entry `m = N/2` is set to 0, and that coefficient is cleared after every step. A derivative of order `p` multiplies the unnormalized NumPy FFT by `(i k)^p`.

The product `u u_x` uses the 3/2 dealiasing rule. Spectra are padded to `M = 3N/2`, the Nyquist mode is dropped, the product is formed on the fine grid, and the result is truncated back to `N` modes. `N` is even.

Time stepping is ETDRK4 (Cox and Matthews, 2002), with the Kassam and Trefethen (2005) coefficients. In Fourier space the linear part is `L = -ν k²`, treated by the exponential integrator, and `N(U) = -FFT(u u_x)` is explicit. For `|dt L| < 1` the coefficients use a Taylor series about 0. Elsewhere they use an `expm1` form of the same formulas. Fields are `float64`. Spectra are `complex128`.

Diffusion does not set the step. A working explicit-advection guide is

    dt <= 0.5 / (||u||_∞ k_max),    k_max = π (N/2 - 1).

For `||u||_∞ <= 1` and `N = 256` that guide is `dt <= 0.0012531885282826404`. The pilot uses `dt = 0.001`. This is a stability guide. The accuracy statement is the convergence section below.

### Cross-checks

Two checks are independent of trusting one resolution of the Fourier scheme.

1. **Cole–Hopf.** If `φ = 1 + a exp(-ν k² t) cos(k x)` with `k = π` and `|a| < 1`, then `φ_t = ν φ_xx` and `u = -2ν (ln φ)_x` solves Burgers. The solver is started from that field at `t = 0` and compared with the formula.
2. **Finite differences.** Periodic second-order centered differences and classical RK4, on the same Cole–Hopf problem (`a = 0.4`, `ν = 0.1`, `t = 0.05`, `dt = 2.5e-4`):

| Scheme | N | relative L2 at t = 0.05 |
| --- | ---: | ---: |
| finite difference | 32 | 6.872752541370733e-4 |
| finite difference | 64 | 1.7282028784595537e-4 |
| Fourier ETDRK4 | 64 | 5.187351144680026e-15 |

Doubling `N` reduced the finite-difference error by a factor of about 3.98. The Fourier error at `N = 64` is much smaller. The unit test `test_finite_difference_converges_toward_cole_hopf_and_trails_the_spectral_solver` checks the same ordering on this case.

### Mean and energy

For the unforced periodic problem,

    mean = (1/N) Σ_j u_j,

which equals `(1/L) ∫ u dx` for modes the grid resolves, and

    E = (dx / 2) Σ_j u_j²,

the rectangle rule for `∫ u² / 2 dx`. The continuous mean is conserved. The continuous energy is non-increasing because `dE/dt = -ν ∫ (u_x)² dx`. The discrete checks below use the saved samples of the Fourier solution. They are not a proof that every intermediate stage is monotone.

## Initial conditions and viscosity

Each instance is

    u_raw(x) = Σ_{m=1}^{4} [a_m cos(m π x) + b_m sin(m π x)],

with `a_m` and `b_m` drawn in that order as `Uniform(-1, 1) / m`. There is no mean mode. The generator is PCG64 from `SeedSequence([master_seed, instance_id, 0x42555247])`. The field is divided by its maximum absolute value on a 4096-point grid, so the scale does not depend on the solver `N`. A draw with canonical peak below `1e-8` is redrawn. Viscosity is drawn after a successful field:

    ν ~ Uniform(0.02, 0.10).

The pilot master seed and split seed are both `20260926`.

## Convergence

Command:

```bash
python -m pinnforge.reference.numerical convergence --output docs/stage2
```

Raw rows are in [docs/stage2/convergence.json](docs/stage2/convergence.json). Plots: [spatial](docs/stage2/spatial_convergence.svg), [temporal](docs/stage2/temporal_convergence.svg), [invariants](docs/stage2/invariants.svg).

Relative L2 is `||u - v||_2 / ||v||_2` on the shared samples. Spatial comparisons of two grids restrict the fine field by truncating its FFT onto the coarse modes.

### Spatial, steep Cole–Hopf

`a = 0.99`, `ν = 0.05`, `dt = 1e-4`, `t_final = 0.2`. This profile is steep enough that coarse grids still have a Fourier tail. Final-time relative L2 against the exact formula:

| N | relative L2 |
| ---: | ---: |
| 32 | 1.1449392016150604e-3 |
| 64 | 8.468417152794791e-6 |
| 128 | 8.689968585130353e-10 |
| 256 | 7.429183385406563e-14 |
| 512 | 7.42471036038599e-14 |

The error drops by orders of magnitude until `N = 256`, then sits near `7.4e-14`.

### Spatial, smooth Cole–Hopf and the unforced family

Smooth Cole–Hopf (`a = 0.5`, `ν = 0.05`, `dt = 2.5e-4`, `t_final = 1`) at `N = 64, 128, 256, 512, 1024` has final-time relative L2 between `1.5024385378372304e-13` and `1.5040738928626775e-13`. That scan is flat: `N = 64` is already at the floor of this time step.

The unforced family member `master_seed = 0`, `instance_id = 0`, with `ν` fixed at `0.05`, compared at `dt = 5e-4` with an `N = 1024` run:

| N | final-time relative L2 vs N = 1024 |
| ---: | ---: |
| 128 | 2.560528481915664e-16 |
| 256 | 4.3949209479826873e-16 |
| 512 | 2.547197654886563e-16 |
| 1024 | 1.8650086773509688e-16 |

Modes `1..4` are represented exactly on all of these grids, and at `ν = 0.05` the `N = 128` solution already matches `N = 1024` to roundoff.

### Temporal

Unforced problem, `N = 512`, `ν = 0.05`, final-time relative L2 against `dt = 2.5e-4`:

| dt | relative L2 | ratio to the next halved step |
| ---: | ---: | ---: |
| 4e-3 | 2.529443055215068e-10 | 15.81 |
| 2e-3 | 1.6003079203750534e-11 | 14.68 |
| 1e-3 | 1.0901221315314347e-12 | 7.32 |
| 5e-4 | 1.488309153908887e-13 | |
| 2.5e-4 | 0 (reference) | |

The first two halvings are close to the factor 16 expected from a fourth-order step. The next halving is smaller because the error is near `1e-13`.

Smooth Cole–Hopf at `N = 256` falls from `7.695895585955198e-13` at `dt = 4e-3` to `5.459610935034916e-14` at `dt = 2e-3` (factor 14.1), then stays between about `2e-14` and `1.5e-13`.

### Mean and energy on the unforced problem

Pilot settings `N = 256`, `dt = 1e-3`, `ν = 0.05`, samples every `0.04`:

| quantity | value |
| --- | ---: |
| max abs mean drift | 2.7755575615628914e-17 |
| energy at t = 0 | 0.2833973527069327 |
| energy at t = 1 | 0.08346982845346085 |
| max energy increase | 0 |
| max abs at t = 0 | 0.9997738186141545 |
| max abs at t = 1 | 0.42509726731482256 |

The `N = 1024`, `dt = 5e-4` run has max abs mean drift `5.551115123125783e-17` and max energy increase `0`. Energy falls on every saved sample in both runs. Twelve pilot files, instance ids `0, 64, ..., 704`, had max abs mean drift at most `8.326672684688674e-17` and max energy increase `0`. That is a spot check of the generated files, not a scan of all 768.

### Direct pilot settings against the fine run

Same unforced initial condition, `ν = 0.05`, pilot grid `N = 256`, `dt = 1e-3`, `save_dt = 0.01` (101 frames), against `N = 1024`, `dt = 2.5e-4` restricted onto 256 modes:

| quantity | relative L2 |
| --- | ---: |
| final time | 1.0901121054009838e-12 |
| all 101 saved frames | 1.0589374100947252e-11 |

The final-time figure matches the `N = 512` temporal entry at `dt = 1e-3`. On this initial condition the pilot error is the time step, not the spatial grid.

### Viscosity probe

Same initial condition, `dt = 5e-4`, `t_final = 1`, final-time relative L2 of `N = 256` against `N = 512`:

| ν | relative L2 | in the pilot interval |
| ---: | ---: | --- |
| 0.10 | 2.846076987760781e-16 | yes |
| 0.05 | 4.0174850595754505e-16 | yes |
| 0.02 | 2.420415969618625e-16 | yes |
| 0.01 | 6.318965093616459e-12 | no |
| 0.005 | 1.909825737917755e-6 | no |

`ν = 0.005` is above the tolerance below, so it is not a pilot label. `ν = 0.01` is under that tolerance and is still outside the interval the pilot draws.

## Preregistered label tolerance

Written after this pilot and before any operator model is trained.

A label at `N = 256`, `dt = 1e-3`, `save_dt = 0.01`, float64, 3/2 dealiasing, ETDRK4, for the unforced convergence initial condition with `ν ∈ [0.02, 0.10]`, is acceptable when all of the following hold against an `N = 1024`, `dt = 2.5e-4` integration of the same initial data:

- final-time relative L2 ≤ `1e-8`
- space-time relative L2 on the 101 saved frames ≤ `1e-8`
- max abs mean drift ≤ `1e-12`
- max energy increase on the saved samples ≤ `1e-10`

The measured pilot-versus-reference errors are `1.09e-12` and `1.06e-11`. Mean drift and energy increase on that run pass. This tolerance is the gate for these labels. It is not a measured model error, and it is not a certificate for every random draw beyond the probe and the 12-file spot check.

## Pilot dataset

Estimate, from the field sizes, before the full write:

```text
instances=768 times=101 n=256 bytes_per_trajectory=206848 field_mebibytes=151.500
```

A batch of 8 trajectories for 200 steps extrapolated to about 31 seconds for 768 full trajectories. One batch holds `8 * 101 * 256 * 8 = 1654784` bytes of saved fields. Process resident memory was not measured with a profiler.

Actual generation in this workspace: **34.97 seconds**. The directory `artifacts/burgers_pilot/` is **153,813,279 bytes** (146.7 MiB), including npz wrappers and the manifest. Raw fields were estimated at 158,859,264 bytes. Compression did not shrink the float64 arrays by much. The arrays are listed in `.gitignore`. The manifest is committed at [docs/stage2/pilot_manifest.json](docs/stage2/pilot_manifest.json).

```bash
python -m pinnforge.reference.numerical estimate --train 512 --val 128 --test 128 --benchmark
python -m pinnforge.reference.numerical pilot --output artifacts/burgers_pilot --train 512 --val 128 --test 128
```

The second command refuses to overwrite an existing directory.

| item | value |
| --- | --- |
| counts | 512 train, 128 validation, 128 test |
| `N`, `dt`, `save_dt`, `t_final` | 256, 0.001, 0.01, 1 |
| master seed, split seed | 20260926, 20260926 |
| solver config SHA-256 | `2a0c1c3f078ca81f15bf5c33874bc1403fa6a1a30bda1f484bbc6df5d0d48928` |
| pilot config SHA-256 | `1926d9a625136afca2af0d4dbc318cd77616ee0268ef80341e55c45daf156088` |
| training `u` mean | 1.2021494137744596e-18 |
| training `u` std (ddof 0) | 0.3469596293138632 |
| training value count | 13238272 |
| drawn `ν` min, max | 0.02002767492769535, 0.09993362700441269 |

The split is a Fisher–Yates permutation of instance ids `0 .. 767` from PCG64 and `SeedSequence([split_seed, 0x53504C54])`, cut into train, validation, then test. It is computed before integration. No windows are extracted. Stored arrays are raw solver output. The mean and standard deviation above are fit on the training files only and are not applied to the arrays. The manifest lists are sorted for display; regenerating `assign_splits` from the split seed recovers the permutation. Each instance records an npz SHA-256 and a `field_sha256` of the contiguous float64 field. Code SHA-256 values for `solver.py`, `checks.py`, `initial.py`, and `dataset.py` are inside the manifest and matched the tree at generation.

## Gates

| gate | result |
| --- | --- |
| Cole–Hopf or an independent scheme | pass (both) |
| spatial refinement visible on a steep exact solution | pass (`a = 0.99`) |
| 256 / 512 / 1024 recorded | pass |
| halved time steps, order near 4 until the floor | pass |
| mean drift and energy on the unforced problem | pass |
| pilot 512 / 128 / 128 with manifest, seeds, hashes | pass |
| `ν = 0.005` included | fail the `1e-8` probe, excluded |
| label error on the convergence IC under `1e-8` | pass (`1.09e-12` final, `1.06e-11` space-time) |
| coordinate-PINN Burgers hook left unimplemented | pass |
| FNO / inverse viscosity | not in this stage |

## Commands and tests

```bash
python -m ruff check .
python -m pytest -m "not ml" -q --tb=line
```

Lint:

```text
All checks passed!
```

Unit job in this workspace, with the `api` extra installed and without torch. The 92 deselected tests are the `ml` marker:

```text
219 passed, 92 deselected in 2.41s
```

Full suite after installing the CPU torch wheel. The six skips are the no-torch cases in `tests/test_ml_import.py`:

```text
305 passed, 6 skipped in 4.26s
```

## Remaining risks

- The `1e-8` gate was measured on one initial condition, a five-value viscosity probe, and 12 of the 768 files. It is not a twin `N = 1024` integration of every label.
- Energy non-increase is reported on saved output samples. ETDRK4 is not proved monotone at every stage.
- npz hashes and the last bits of the fields can move with a different NumPy. This manifest records NumPy 2.4.4.
- Smooth solutions at `N ≥ 64` do not show a spatial slope. The slope in the spatial figure is the `a = 0.99` case.
- `ν = 0.005` at `N = 256` is outside the gate. Lower viscosities need a new resolution study before they can be labels.
- These trajectories are not wired into `pinnforge eval`.

## Next bounded experiment

Stage 3 can train a Fourier neural operator on windows cut from these trajectories only after this instance split, using the training mean and standard deviation in the manifest, and score relative L2 on the test instances. That training is not part of this stage.
