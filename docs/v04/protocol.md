# v0.4 protocol (frozen before test scores)

This note is the human reading of [protocol.json](protocol.json). That JSON is the contract. It was committed before any test split was scored. It holds no test number.

## Question

Can a 2D FNO first guess plus a bounded solver-in-the-loop refinement recover a scalar viscosity on periodic 2D Burgers and 2D incompressible Navier–Stokes (vorticity form) from sparse noisy probes, with a frozen low-`ν` `hard_ood` slice, calibrated nominal 0.9 intervals, and wall clock well below a cold solver search?

## Problems

Both problems live on a periodic square of length `2π`, grid `64 × 64`, `t ∈ [0, 0.2]`, saved every `0.025`. Viscosity is drawn independently from `Uniform(0.012, 0.072)`. `hard_ood` is the training-split lower quartile of `ν`. That cut is written into the pilot manifest after generation and is not refit.

- **2D viscous Burgers (vector).** `u_t + (u · ∇)u = ν Δu`. Bandlimited velocity initial data, modes through 3, max-abs normalized.
- **2D incompressible Navier–Stokes, vorticity form.** `ω_t + u · ∇ω = ν Δω + f`, `u = ∇^⊥ ψ`, `−Δψ = ω`. Even instance ids use Kolmogorov forcing `f = −4 cos(4y)`. Odd ids are decaying (`f = 0`).

The forward model is a dealiased pseudo-spectral ETDRK4 solver implemented in numpy. A refinement/convergence check is part of the solver contract. Exponax and JAX are not used.

## Sensors

An `8 × 8` lattice (64 points, stride 8) and three bursts of three frames at saved indices `0, 3, 6`. The mask is identical for every instance. Noise is additive Gaussian at `0`, `1%`, `2%`, and `5%` of that instance's clean-field standard deviation.

## Methods

- 2D FNO forward, data loss and hybrid physics residual (weight `1e-2`, frozen).
- FNO-conditioned inverse on a 21-point geometric `ν` grid.
- Cold solver-in-the-loop fit from a sensor-lifted initial condition.
- v0.3-style warm start: FNO first guess, then at most 2 or 3 Gauss–Newton solver evaluations (cap chosen on validation), split-conformal nominal 0.9 intervals, abstain flag.
- Closed-form residual least squares after a trigonometric lift, where the residual is defined.
- Weak-form identifier on the sensor lattice.
- Oracle-IC solver fit (ceiling).

Five seeds. Mean ± sample standard deviation. The primary method is the hybrid-FNO warm start on Burgers.

## Pass bars

Written here before any test score. No retuning after test.

| bar | limit |
| --- | --- |
| in-distribution (`complement`) mean relative `ν` error | `<= 0.05` |
| `hard_ood` / complement gap ratio | `<= 3` |
| warm-start `hard_ood` error vs cold solver | `<= 1.5×` at `<= 0.25×` wall clock |
| full-test coverage of the nominal 0.9 interval | in `[0.85, 0.95]` |

All four must pass on Burgers hybrid warm start. The same bars are scored on Navier–Stokes and published as pass or fail without retuning.

## What is NOT claimed

- a result on The Well `shear_flow` or Petrobras 3W
- 3D or 3D+time
- that the warm start is more accurate than the cold search
- coverage conditional on viscosity, noise, or the training range
- that the abstain flag detects every large error
- a retuned `hard_ood` rule after seeing test scores
- an architecture or loss weight that is optimal outside this frozen set
