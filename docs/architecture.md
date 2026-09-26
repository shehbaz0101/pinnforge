# Architecture

PINNForge trains a small network on a classic residual and scores it against an analytical or fixed reference. The package version is 0.1.0. The pre-release freeze is Day 10. This tree has the Day 1 equation schemas and harmonic closed form, the Day 2 samplers, the Day 3 MLP and residual operators, and the Day 4 Adam trainer.

Importing `pinnforge` loads pydantic specs and the numpy samplers. It does not import torch. Torch is the optional `ml` extra. `pinnforge.models`, `pinnforge.residuals`, `pinnforge.losses`, and `pinnforge.training` raise `InstallHint` when it is missing. Default CI runs `pytest -m "not ml"` without torch. A separate job installs a CPU wheel and runs the full suite, including the residual and training tests.

## Today

| Piece | Now |
| --- | --- |
| Harmonic oscillator | Spec with ω or k and m, a time interval, state IC `(u, du_dt)`, optional BC descriptors. Numpy solution `u(t) = A cos(ω(t-t0)) + B sin(ω(t-t0))`. |
| Burgers 1D | Spec with viscosity ν > 0, x and t intervals, a named IC profile, and BC descriptors. Reference raises `NotImplementedError`. |
| Poisson toy | Spec with dimension 1 or 2, a named source, and BC descriptors. Reference raises `NotImplementedError`. |
| Registry | In-process map from `equation_id` to the spec class. |
| Sampler | Uniform, Latin-hypercube, or stratified draws on a `CollocationDomain`, plus IC and BC faces. A `CollocationBatch` holds float64 coordinates and `interior` / `ic` / `bc` labels. |
| MLP | Fully connected field `u`. Input width comes from the collocation domain. Default activation is `tanh`. |
| Residual | `residual(model, coords, spec)` returns ü + ω²u, u_t + u u_x − ν u_xx, or −Δu − f. Derivatives use `torch.autograd.grad` with `create_graph=True`. |
| Penalties | Soft mean-squared initial-condition and Dirichlet terms from a `CollocationBatch` and the spec. Neumann and periodic faces are not penalized. |
| Trainer | Adam on one seeded batch. Loss is `w_pde *` residual MSE plus weighted soft IC and Dirichlet penalties. Writes `metrics.jsonl` and a CPU checkpoint (`state_dict`, config, epoch). |
| CLI | `pinnforge version`, `pinnforge equations`, `pinnforge sample`, `pinnforge residual`, and `pinnforge train`. The last two need torch. |

Shared types (`Interval`, `CollocationDomain`, `StateInitialCondition`, `ProfileInitialCondition`, `BoundaryCondition`) are what the samplers read. Specs reject unknown fields.

## Sampling

Day 2 draws points from a `CollocationDomain` and does not build a residual. Interior points use one seeded `numpy.random.Generator` on the half-open box `[lower, upper)`. Initial-condition points sit on the initial-time face (`t = t0` for the oscillator and Burgers). Boundary points sit on the faces in `boundary_conditions`: Dirichlet and Neumann fix the named side, and a periodic condition samples both ends. The generator continues from interior to IC to BC, so a seed reproduces the whole batch. Poisson rejects `n_ic > 0`. A harmonic spec with no boundary descriptors rejects `n_bc > 0`. Counts are non-negative integers; a config that asks for no points at all is rejected. `pinnforge sample` prints counts and bounds and can write a JSON record under a relative path in the working directory. Offline fixtures live in `tests/fixtures/sampling/`.

## Residuals

`mlp_from_spec` builds a network whose input width is the number of collocation axes. `residual` evaluates that network (or any callable with the same signature) and differentiates the scalar field with `create_graph=True`. The oscillator residual is ü + ω²u. Burgers uses columns `(x, t)` and the residual u_t + u u_x − ν u_xx. Poisson follows the Day 1 statement −Δu = f, so the residual is −Δu − f for the named source. `soft_penalty` reads `ic` and Dirichlet `bc` rows from a `CollocationBatch` and compares them with the prescribed spec values. `pinnforge residual --equation harmonic --seed 0` samples 16 interior points, builds an untrained width-(8, 8) network, and prints the residual MSE.

## Training

Day 4 minimizes those terms with Adam on CPU. `TrainConfig` names the built-in equation, sample counts, MLP widths, epoch count, learning rate, loss weights, seed, and relative output paths. The loop draws one batch with that seed and reuses it every epoch, so `metrics.jsonl` compares the same collocation points from epoch 0 (initial weights) through the last Adam step. The scalar loss is `w_pde * MSE(residual) + w_ic * initial + w_bc * dirichlet`. Each checkpoint is a `torch.save` dict with the model `state_dict`, the config, and the epoch. `load_checkpoint` rebuilds the MLP for evaluation. Optimizer state is not stored. `pinnforge train --equation harmonic --epochs 50 --seed 0` is the small CPU demo.

## Later days

1. **Day 2 — Samplers.** Done. Collocation, IC, and BC draws for the three specs. See [daily/day02.md](daily/day02.md).
2. **Day 3 — PINN MLP.** Done. CPU MLP, autograd residuals, and soft IC/BC penalties. See [daily/day03.md](daily/day03.md).
3. **Day 4 — Train.** Done. Adam on a fixed seeded batch, metrics, and CPU checkpoints. See [daily/day04.md](daily/day04.md).
4. **Day 5 — Eval.** Score a trained net against the harmonic closed form and against fixed Burgers and Poisson references.
5. **Day 6 — Registry.** Persist a run: spec, sampler settings, checkpoint path, and metrics. Extends the Day 1 in-process registry.
6. **Day 7 — Serve.** Local HTTP API. A health check and an offline train or eval call. Loopback by default, no authentication.
7. **Day 8 — Harden.** Path sandbox, request limits, and offline-by-default behavior.
8. **Day 9 — Demo.** One command that trains a tiny harmonic-oscillator problem and prints an error.
9. **Day 10 — Freeze.** v0.1.0 pre-release freeze: status note and changelog.

Days 5–10 are the plan. This tree does not implement them.
