# v052 diagnosis: NS WSINDy 64x64 noise-0 failure

This note diagnoses the v0.5.1 Navier-Stokes positive-control miss
(`64x64`, noise `0`, seed exact-support rate `0`). It does not edit
[docs/v05/](../v05/) or [docs/v051/](../v051/). Numeric dumps are
[diagnosis_evidence.json](diagnosis_evidence.json). That file is a
single noise-free Kolmogorov trajectory (`nu = 0.04`, instance-style
seed `10028`) assembled with the frozen v05 WSINDy code and the v051
protocol, then with alternate center strides. It is not a test score.

## Checks that are not the bug

The generator, the library true support, the signs, the test-function
derivatives, the time derivative, and column normalization are consistent
with the PDE. The least-squares residual of the true support on
noise-free full-field data is at discretization error **before** STLSQ.

### PDE and library

The solver states

```
ω_t + u · ∇ω = ν Δω + f,   u = ∇^⊥ ψ,   −Δψ = ω
```

with Kolmogorov `f = -k cos(k y)` and `k = 4`, and `u = (-ψ_y, ψ_x)`
from the streamfunction. Decaying uses `f = 0`. The library writes
advection as the mixed products `u_wx = u ω_x` and `v_wy = v ω_y`,
diffusion as `w_xx` and `w_yy`, and puts Kolmogorov forcing in the
library as `force_y` (the same `kolmogorov_forcing` field) with
coefficient `1`. It does not treat `f` as a known term moved to the
right-hand side. True Kolmogorov support
`{u_wx, v_wy, w_xx, w_yy, force_y}` with coefficients
`-1, -1, ν, ν, 1` therefore matches the generator.

On the noise-free `64x64` field, the weak system with those five
columns and the true coefficients has relative residual
`5.700066971865476e-05`. Unconstrained least squares on the same
columns recovers

| term | true | least squares |
| --- | --- | --- |
| u_wx | -1 | -0.9999594188878751 |
| v_wy | -1 | -0.999682198683844 |
| w_xx | 0.04 | 0.04000763751096851 |
| w_yy | 0.04 | 0.039994563046648605 |
| force_y | 1 | 1.0000000873073192 |

Relative residual after that fit: `4.958000701728834e-05`. Condition
number of the true-support matrix: `45.17492999732106`. Rank `5`.

The strong-form residual of the same central time difference,
`||ω_t + u·∇ω − νΔω − f|| / ||ω_t||`, averages
`0.000117652442582397` over the three burst mid-frames. The weak-form
true-support residual sits next to that time-discretization floor.

### Test function and time derivative

The test function is `(1 − ξ²)_+^4` with spatial radius `0.25 L`.
Even-order integration by parts uses a positive sign; third-order uses
a negative sign. At one center, the IBP column for `w_xx` differs from
the strong-form integral `∫ ω_xx φ` by relative
`0.0002240600416578455`; `w_yy` differs by
`0.0005935049225677326`. Mixed products use spectral field derivatives,
which matches `u` from the streamfunction.

A degree-4 bump on a three-frame burst has a trapezoidal `η_t` of
exactly `0` (value and first derivative vanish at the endpoints; the
first derivative is `0` at the midpoint). The implementation therefore
uses the central difference of the field at the middle frame. That is
required, not a sign error.

Column normalization is applied inside STLSQ. It is not what makes the
true support unidentifiable: the five true columns are well conditioned
without the distractors.

## The bug

`center_stride_fraction = 0.25` places bump centers every
`round(0.25 n)` points on the grid the method sees. For every lattice
`n ∈ {8, 16, 32, 64}` that stride equals `n/4`, which is exactly one
period of `cos(4 y)` (and of `cos(4 x)`). On `64x64` the centers sit at
indices `0, 16, 32, 48`. `cos(4 y)` is `1` at every one of them.

Because the periodic bump is translation-invariant, the weak-form
columns `force_y`, `const`, and `force_x` are then constant across
rows. Their pairwise correlations are `-1` or `1`. The three-column
block has rank `1`. The full normalized library has condition number
`1.230123348508887e+16`.

Min-norm STLSQ therefore splits the forcing weight across
`force_y + const + force_x` at every frozen threshold. The recovered
support on this trajectory, and in every v051 `64x64` noise-0 seed, is
`{u_wx, v_wy, w_xx, w_yy, force_y, const, force_x}`. Validation picks
the largest candidate (`0.35`) because no threshold is exact. That is
the v051 control miss. It is a design-matrix degeneracy, not a wrong
PDE and not a sparsity finding.

The same degeneracy appears at `8x8`, `16x16`, and `32x32` with
fraction `0.25`: one unique forcing phase, correlations `±1`, no exact
STLSQ support. That is why every NS lattice in v051 has seed exact
rate `0`.

Offsetting the first center does not help. Equal spacing of one period
repeats the same phase.

## Fixes that recover exact support on this trajectory

Keep `force_y` in the library (same true support as v051) and choose a
stride that is not an integer number of Kolmogorov periods:
`4 · stride / n` is not an integer.

| fraction | 64-stride | unique phases | corr(force_y, const) | any exact STLSQ |
| --- | --- | --- | --- | --- |
| 0.25 | 16 | 1 | -1 | no |
| 0.2 | 13 | 5 | 0.07023066047141817 | yes (`0.02`–`0.25`) |
| 1/6 | 11 | 6 | 0.02001708193192954 | yes (`0.02`–`0.25`) |

`1/6` also breaks the period lock at `8x8` (`stride = 1`, two phases).
`0.2` does not: `round(0.2 · 8) = 2` is still one period.

An alternative that also works, including with fraction `0.25`, is to
subtract the known Kolmogorov field from the weak right-hand side and
drop `force_y` from the library. On this trajectory that recovers
`{u_wx, v_wy, w_xx, w_yy}` at thresholds `0.02` through `0.18`. It
changes the support definition, so it is not the registered v052 fix.
The forcing term stays in the library. The registered change is the
incommensurate center stride.

## What v052 will freeze

A new protocol, committed before any v052 test score, keeps the v051
density × noise grid, the v05 library, the same STLSQ candidates, the
same `0.8` bar at noise `0.01`, and the same reported quantity, for
`ns2d` only. Burgers is not rescored. `center_stride_fraction` becomes
`0.16666666666666666` (`1/6`).
