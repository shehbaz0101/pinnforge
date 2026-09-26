# Architecture

PINNForge trains a small network on a classic residual and scores it against an analytical or fixed reference. The package version is 0.1.0 from Day 1. The pre-release freeze is Day 10. This tree is the Day 1 scaffold: equation schemas, one closed form, and a CLI stub. There is no sampler, network, or trainer.

Importing `pinnforge` loads pydantic specs and does not import torch. Numpy is used by the harmonic closed form. Torch is declared in the optional `ml` extra for later days and is not installed in CI.

## Today

| Piece | Day 1 |
| --- | --- |
| Harmonic oscillator | Spec with ω or k and m, a time interval, state IC `(u, du_dt)`, optional BC descriptors. Numpy solution `u(t) = A cos(ω(t-t0)) + B sin(ω(t-t0))`. |
| Burgers 1D | Spec with viscosity ν > 0, x and t intervals, a named IC profile, and BC descriptors. Reference raises `NotImplementedError`. |
| Poisson toy | Spec with dimension 1 or 2, a named source, and BC descriptors. Reference raises `NotImplementedError`. |
| Registry | In-process map from `equation_id` to the spec class. |
| CLI | `pinnforge version` and `pinnforge equations`. |

Shared types (`Interval`, `CollocationDomain`, `StateInitialCondition`, `ProfileInitialCondition`, `BoundaryCondition`) are the extension point for later days. Specs reject unknown fields.

## Later days

1. **Day 2 — Samplers.** Draw collocation points inside `CollocationDomain`, plus IC and BC samples from the descriptors.
2. **Day 3 — PINN MLP.** A small fully connected net on CPU, imported only with the `ml` extra.
3. **Day 4 — Train.** Residual loss for the three Day 1 equations. Adam. No GPU requirement.
4. **Day 5 — Eval.** Score a trained net against the harmonic closed form and against fixed Burgers and Poisson references.
5. **Day 6 — Registry.** Persist a run: spec, sampler settings, checkpoint path, and metrics. Extends the Day 1 in-process registry.
6. **Day 7 — Serve.** Local HTTP API. A health check and an offline train or eval call. Loopback by default, no authentication.
7. **Day 8 — Harden.** Path sandbox, request limits, and offline-by-default behavior.
8. **Day 9 — Demo.** One command that trains a tiny harmonic-oscillator problem and prints an error.
9. **Day 10 — Freeze.** v0.1.0 pre-release freeze: status note and changelog.

Days 2–10 are the plan. This tree does not implement them.
