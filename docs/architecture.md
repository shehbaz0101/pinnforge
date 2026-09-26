# Architecture

PINNForge trains a small network on a classic residual and scores it against an analytical or fixed reference. The package version is 0.1.0. The pre-release freeze is Day 10. This tree has the Day 1 equation schemas and harmonic closed form, and the Day 2 samplers. There is no network or trainer.

Importing `pinnforge` loads pydantic specs and the numpy samplers. It does not import torch. Torch is declared in the optional `ml` extra for later days and is not installed in CI.

## Today

| Piece | Now |
| --- | --- |
| Harmonic oscillator | Spec with ω or k and m, a time interval, state IC `(u, du_dt)`, optional BC descriptors. Numpy solution `u(t) = A cos(ω(t-t0)) + B sin(ω(t-t0))`. |
| Burgers 1D | Spec with viscosity ν > 0, x and t intervals, a named IC profile, and BC descriptors. Reference raises `NotImplementedError`. |
| Poisson toy | Spec with dimension 1 or 2, a named source, and BC descriptors. Reference raises `NotImplementedError`. |
| Registry | In-process map from `equation_id` to the spec class. |
| Sampler | Uniform, Latin-hypercube, or stratified draws on a `CollocationDomain`, plus IC and BC faces. A `CollocationBatch` holds float64 coordinates and `interior` / `ic` / `bc` labels. |
| CLI | `pinnforge version`, `pinnforge equations`, and `pinnforge sample`. |

Shared types (`Interval`, `CollocationDomain`, `StateInitialCondition`, `ProfileInitialCondition`, `BoundaryCondition`) are what the samplers read. Specs reject unknown fields.

## Sampling

Day 2 draws points from a `CollocationDomain` and does not build a residual. Interior points use one seeded `numpy.random.Generator` on the half-open box `[lower, upper)`. Initial-condition points sit on the initial-time face (`t = t0` for the oscillator and Burgers). Boundary points sit on the faces in `boundary_conditions`: Dirichlet and Neumann fix the named side, and a periodic condition samples both ends. The generator continues from interior to IC to BC, so a seed reproduces the whole batch. Poisson rejects `n_ic > 0`. A harmonic spec with no boundary descriptors rejects `n_bc > 0`. Counts are non-negative integers; a config that asks for no points at all is rejected. `pinnforge sample` prints counts and bounds and can write a JSON record under a relative path in the working directory. Offline fixtures live in `tests/fixtures/sampling/`.

## Later days

1. **Day 2 — Samplers.** Done. Collocation, IC, and BC draws for the three specs. See [daily/day02.md](daily/day02.md).
2. **Day 3 — PINN MLP.** A small fully connected net on CPU, imported only with the `ml` extra.
3. **Day 4 — Train.** Residual loss for the three Day 1 equations. Adam. No GPU requirement.
4. **Day 5 — Eval.** Score a trained net against the harmonic closed form and against fixed Burgers and Poisson references.
5. **Day 6 — Registry.** Persist a run: spec, sampler settings, checkpoint path, and metrics. Extends the Day 1 in-process registry.
6. **Day 7 — Serve.** Local HTTP API. A health check and an offline train or eval call. Loopback by default, no authentication.
7. **Day 8 — Harden.** Path sandbox, request limits, and offline-by-default behavior.
8. **Day 9 — Demo.** One command that trains a tiny harmonic-oscillator problem and prints an error.
9. **Day 10 — Freeze.** v0.1.0 pre-release freeze: status note and changelog.

Days 3–10 are the plan. This tree does not implement them.
