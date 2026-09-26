# Project status

PINNForge is a sandbox for physics-informed neural networks on a few classic
ODE and PDE residuals. It trains a small network on the residual, scores a
checkpoint against an analytical or manufactured field, and serves that path
on localhost. An offline demo trains a checked-in sample on CPU.

**Status:** v0.1.0 remains the tagged freeze. Stage 1 correctness is unreleased and keeps the `0.1.0` version string. See [STAGE1_REPORT.md](STAGE1_REPORT.md).

## Days 1–10

| Day | Delivered |
| --- | --- |
| 1 | Package scaffold, equation schemas, harmonic closed form |
| 2 | Collocation, initial-condition, and boundary samplers |
| 3 | MLP and residual autograd |
| 4 | Adam training loop, checkpoints, and metrics |
| 5 | Evaluation against reference fields and residual stats |
| 6 | Experiment config and `pinnforge run --config` |
| 7 | FastAPI service and `pinnforge serve` |
| 8 | Path sandbox, per-client rate limit, offline socket guard |
| 9 | `pinnforge demo` on checked-in samples |
| 10 | Freeze at v0.1.0, this status note, changelog |

## Install

Python 3.11 or newer. Install a CPU build of torch first so the `ml` extra
does not pull a CUDA wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[ml,api]"
```

`pinnforge demo`, `pinnforge train`, `pinnforge eval`, `pinnforge run`,
`pinnforge residual`, and the train, eval, and run HTTP routes need the
`ml` extra. `pinnforge serve` needs the `api` extra. Schemas, sampling,
`pinnforge sample`, and `GET /health` and `GET /equations` do not need
torch. Tests use `pip install -e ".[dev,api]"`. Lint uses
`pip install -e ".[lint]"`.

## Quickstart

```bash
pinnforge demo
pinnforge serve
```

`pinnforge demo` trains `samples/configs/harmonic.yaml` on CPU (2 epochs,
a width-(8, 8) network) and prints the loss and the error against the
closed form. `--equation poisson` uses `samples/configs/poisson.json`.
`--equation burgers` uses `samples/configs/burgers.yaml` and prints residual
metrics, because Burgers has no reference field. It does not open a network
connection. `pinnforge serve` binds to `127.0.0.1:8000`.

## Tests and CI

CI runs on pull requests and on pushes to `main`. The unit-test job installs
`pip install -e ".[dev,api]"` and runs `pytest -m "not ml"` on Python 3.11
and 3.12. A separate job installs the CPU torch wheel, then the same extras,
and runs the full suite on Python 3.12. A lint job runs `ruff check .`
(rules E4, E7, E9, F, I). HTTP tests use FastAPI `TestClient` and do not
bind a port. Nothing in the suite downloads weights.

## Stage 1

Correctness and packaging on top of the v0.1.0 freeze. The version string is still `0.1.0` because the CLI and import tests pin that string. This branch does not move the `v0.1.0` tag.

- Periodic penalties match the field and the derivative along the periodic axis at both endpoints, at each boundary row's other coordinates. Neumann penalties match the outward normal derivative. A condition the penalty cannot represent raises. `w_bc = 0` still drops boundary enforcement, on purpose.
- A Poisson reference is returned only when it matches the source, the domain, and every prescribed boundary condition. Source `one` on `[0, 1]` with zero Dirichlet ends is `x(1 - x) / 2`. Anything else with no match is unavailable. Burgers is still `NotImplementedError`.
- Train, validation, and test draws use separate `SeedSequence` streams. `pinnforge sample` still uses `numpy.random.default_rng(seed)`. The validation draw is hashed into `manifest.json` and is not in the loss.
- Evaluation reports held-out initial and boundary errors, plus relative and max field error beside the residual. A zero field on the default oscillator has a near-zero residual and relative L2 of 1.
- Demo configs ship in `pinnforge.data` and are read with `importlib.resources`. The repository `samples/` tree remains the checkout fallback.
- `relu` is rejected for the strong-form residual. `tanh` and `silu` stay. Checkpoints record an interval, a run manifest, Adam state, and the torch RNG, and resume when that state is present. Loads use `weights_only=True`.

## Known limits

- CPU-first. `TrainConfig.device` is `cpu` only. `pinnforge demo` allows at
  most 5 epochs. The checked-in samples use 2 epochs and a width-(8, 8)
  network. A longer fit stays on `pinnforge train`.
- Localhost bind. `pinnforge serve` defaults to `127.0.0.1:8000`. `0.0.0.0`
  and `::` are refused unless `--allow-remote` is set. There is no
  authentication.
- Offline-by-design. Train, eval, run, demo, and the API refuse non-loopback
  TCP connects. Loopback stays open. The package does not download weights.
- Free and public only. No API keys, no paid services, no secrets, and no
  private data.
- Burgers has no reference field. Evaluation reports residual metrics and
  boundary penalties, not a field error.
- `w_bc = 0` does not enforce boundary conditions. An eval config with
  `n_bc = 0` or `n_ic = 0` does not score that condition.
- Resume requires a checkpoint that stored the optimizer, the torch RNG,
  and the metrics history, and that matches the equation, counts, widths,
  activation, learning rate, and seed. There is no spectral Burgers
  reference and no neural operator.

## Release tag

Stage 1 does not move `v0.1.0`. The steps below are the freeze procedure from that tag.

`pyproject.toml` and `pinnforge.__version__` are `0.1.0`. The annotated tag
`v0.1.0` is created on the tip of `feat/day10-freeze`. A squash merge of
this pull request produces a new commit on `main` and leaves that tag on
the branch tip. After the squash SHA is on `main`, retarget the tag (the
existing tag has to move):

```bash
git fetch origin main
git push origin :refs/tags/v0.1.0
git tag -a v0.1.0 <squash-sha> -m "PINNForge v0.1.0"
git push origin v0.1.0
```

Do not point `v0.1.0` at any commit other than that squash SHA on `main`.
