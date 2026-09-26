# Changelog

## 0.1.0 — 2026-09-26

First public freeze of the physics-informed sandbox.

- Equation schemas for the harmonic oscillator, Burgers 1D, and a Poisson toy, plus a harmonic closed form.
- Seeded collocation, initial-condition, and boundary samplers.
- CPU MLP, autograd residuals, and an Adam trainer with checkpoints and metrics.
- Evaluation against the harmonic closed form or a manufactured Poisson field. Burgers reports residual metrics only.
- Experiment files and `pinnforge run --config`.
- Localhost FastAPI service (`pinnforge serve`) with a path sandbox, a rate limit, and an offline socket guard.
- Offline demo (`pinnforge demo`) on the checked-in samples.

[PROJECT_STATUS.md](PROJECT_STATUS.md) is the longer status note.
