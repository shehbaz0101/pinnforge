# PINNForge

PINNForge is a sandbox for physics-informed neural networks on a few classic ODE and PDE residuals. Day 1 is the installable package, the equation schemas, and a closed form for the harmonic oscillator. Day 4 trains a small network on the residual with Adam. Day 5 scores a checkpoint against that closed form, or against a manufactured Poisson field, and always reports the interior residual. Day 6 reads one experiment file and runs that train-then-eval path. Day 7 serves the equation catalog and that same path on localhost. Day 8 keeps those file paths inside a sandbox root, rate-limits the train, eval, and run routes, and refuses outbound TCP from that path. Day 9 runs that train-then-eval path from a checked-in sample in one command.

v0.1.0 freezes that surface. [PROJECT_STATUS.md](PROJECT_STATUS.md) lists what shipped. [CHANGELOG.md](CHANGELOG.md) is the release note.

## Quickstart

Python 3.11 or newer. From a clone:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[ml,api]"
pinnforge demo
```

That command trains `samples/configs/harmonic.yaml` on CPU (2 epochs, a width-(8, 8) network) and prints the loss and the error against the closed form. It does not open a network connection. `--equation poisson` uses `samples/configs/poisson.json`. `--equation burgers` uses `samples/configs/burgers.yaml` and prints residual metrics, because Burgers has no reference field. `--epochs 1` shortens the run. `--data-root` sets the sandbox root from Day 8.

## Install

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Core dependencies are pydantic, numpy, and PyYAML. Tests need the `dev` extra (`pytest`, and `httpx2` for the API client). Lint needs the `lint` extra (`ruff`). The `ml` extra installs torch for the MLP, residual operators, the training loop, and evaluation. The `api` extra installs FastAPI and uvicorn for `pinnforge serve`. Schemas, sampling, experiment files, and `pinnforge sample` do not need torch or FastAPI. Default CI skips the torch tests; a separate job installs a CPU wheel and runs them. Both jobs install the `api` extra.

## What works today

`pinnforge version` prints `pinnforge 0.1.0`. `pinnforge equations` lists `burgers_1d`, `harmonic_oscillator`, and `poisson_toy`. `pinnforge sample` draws a seeded collocation batch and prints counts and bounds. `--output` writes a JSON record to a relative path in the working directory. With the `ml` extra, `pinnforge residual --equation harmonic --seed 0` builds a tiny untrained MLP and prints its residual MSE. `pinnforge train --equation harmonic --epochs 50 --seed 0` runs Adam on a fixed seeded batch, writes `metrics.jsonl`, and saves a CPU checkpoint under `checkpoints/`. `pinnforge eval --checkpoint checkpoints/checkpoint.pt --equation harmonic` prints L2 and residual metrics for that checkpoint. `--write-json` adds a record with a residual histogram. `pinnforge run --config samples/configs/harmonic.yaml` trains and then evaluates from one relative YAML or JSON file. `pinnforge demo` runs that path on a checked-in sample and prints the same scores. `pinnforge serve` listens on `127.0.0.1:8000` and exposes that catalog, train, eval, and run path over HTTP. The same commands work as `python -m pinnforge`.

The pydantic specs describe:

- **Harmonic oscillator.** `u'' + ω² u = 0`, with `omega` or both `k` and `m`, a time interval, initial state `(u, du_dt)`, and optional boundary descriptors. Evaluation compares the network with the closed form.
- **Burgers 1D.** `u_t + u u_x = ν u_xx`, with viscosity, x and t bounds, a named initial profile, and boundary descriptors. The evaluation reference raises `NotImplementedError`, so `pinnforge eval` and `pinnforge demo --equation burgers` report residual metrics only. Periodic trajectory labels are a separate Fourier solver: `python -m pinnforge.reference.numerical`. See [STAGE2_REPORT.md](STAGE2_REPORT.md). A 1D Fourier neural operator on windows of those trajectories is `pinnforge fno` (`python -m pinnforge.operator`). The default loss is the Stage 3 normalized data MSE ([STAGE3_REPORT.md](STAGE3_REPORT.md)). `--loss residual` and `--loss hybrid` add a discrete Burgers residual on the same windows ([STAGE4_REPORT.md](STAGE4_REPORT.md)). `pinnforge fno inverse` recovers each instance's viscosity from preregistered sparse sensors by minimizing that residual ([STAGE5_REPORT.md](STAGE5_REPORT.md)). It does not supply the missing Burgers reference field. A harder pilot, with viscosities down to `0.005` and a wider initial band, is generated beside that dataset and is not loaded by `pinnforge fno`: `python -m pinnforge.reference.numerical hard-pilot --output artifacts/burgers_hard_pilot`. Its protocol, including the `hard_ood` training quartile, is [docs/v02/pilot_protocol.json](docs/v02/pilot_protocol.json). See [STAGE_A_REPORT.md](STAGE_A_REPORT.md).
- **Poisson toy.** `-Δu = f` in 1D or 2D, with a named source and boundary descriptors. Each named source has a manufactured field. `sin_pi_x` is `u = sin(π x) / π²`, which matches the default Dirichlet ends on `[0, 1]`.

```bash
pinnforge version
pinnforge equations
pinnforge sample --equation harmonic --n-interior 64 --seed 0
pinnforge residual --equation harmonic --seed 0
pinnforge train --equation harmonic --epochs 50 --seed 0
pinnforge eval --checkpoint checkpoints/checkpoint.pt --equation harmonic
pinnforge run --config samples/configs/harmonic.yaml
pinnforge demo
pinnforge serve
```

## Experiment config

`pinnforge run --config` reads a relative `.yaml`, `.yml`, or `.json` file in the working directory. Absolute paths, `~`, and paths that resolve outside that directory are rejected. The flag-based `train` and `eval` commands still take their own flags.

`equation` is `harmonic`, `burgers`, `poisson`, or the registry id (`harmonic_oscillator`, `burgers_1d`, `poisson_toy`). `equation_params` overrides fields on that built-in spec. The Day 1 schema validates the merge. A mapping in `equation` is a full inline spec instead, and then `equation_params` is not allowed. `train` is a `TrainConfig` block (epochs, widths, weights, relative checkpoint and log paths). `eval` is an `EvalConfig` block (interior count, seed, method, histogram bins). `eval_json` is an optional relative `.json` path for the eval record.

```yaml
equation: harmonic
equation_params:
  omega: 2.0
train:
  epochs: 2
  n_interior: 8
  hidden_widths: [8, 8]
  checkpoint_dir: runs/harmonic/checkpoints
  log_path: runs/harmonic/metrics.jsonl
eval:
  n_interior: 8
  seed: 1
  bins: 4
eval_json: runs/harmonic/eval.json
```

`samples/configs/harmonic.yaml`, `samples/configs/poisson.json`, and `samples/configs/burgers.yaml` are tiny CPU examples. `pinnforge demo` reads those paths. Output under `runs/` is gitignored. See [docs/daily/day06.md](docs/daily/day06.md) and [docs/daily/day09.md](docs/daily/day09.md).

```python
from pinnforge.equations import HarmonicOscillatorSpec, Interval, StateInitialCondition
from pinnforge.reference import displacement

spec = HarmonicOscillatorSpec(
    omega=2.0,
    time=Interval(lower=0.0, upper=1.0),
    initial_condition=StateInitialCondition(components={"u": 1.0, "du_dt": 0.0}),
)
print(float(displacement(0.0, spec)))
```

That displacement is `A cos(ω(t - t0)) + B sin(ω(t - t0))`, with `A` the initial `u`, `B` the initial `du_dt / ω`, and `t0` the lower end of `time`.

## Local API

`pinnforge serve` needs the `api` extra. Train, eval, and run on that server also need the `ml` extra. `/health` and `/equations` do not.

```bash
pip install -e ".[api,ml]"
pinnforge serve
```

The process binds `127.0.0.1` and port `8000`. `--host` can name another address. `0.0.0.0` and `::` are refused unless `--allow-remote` is also set, so a listen on every interface has to be explicit. If you start uvicorn yourself (`uvicorn pinnforge.api:app`), that bind check is not applied; pass `--host 127.0.0.1`. There is no authentication.

Paths in a request are relative to the sandbox root. That root is the server's working directory, or `PINNFORGE_DATA_ROOT` / `pinnforge serve --data-root` when one of those is set. A config path, checkpoint, metrics file, or eval JSON path that resolves outside the root is rejected. Absolute paths and `~` are rejected. `..` and symlinks are resolved before the check.

`POST /train`, `POST /eval`, and `POST /run` share one in-process sliding window per client address. The default is 60 requests per 60 seconds. `PINNFORGE_RATE_LIMIT` and `PINNFORGE_RATE_WINDOW_SECONDS` change it, and so do `--rate-limit` and `--rate-window`. Over the limit the response is HTTP 429 with `Retry-After`. `GET /health` and `GET /equations` are not counted. While the app is up, and while `train`, `eval`, and `run` execute, a non-loopback TCP connect is refused. Loopback stays open. See [docs/daily/day08.md](docs/daily/day08.md).

| Method | Path | Body |
| --- | --- | --- |
| GET | `/health` | Package version. No torch. |
| GET | `/equations` | Registry catalog: id, aliases, summary, parameter names. |
| GET | `/equations/{id_or_alias}` | One catalog entry plus the default spec. Unknown names are 404. |
| POST | `/train` | A `TrainConfig` JSON object. Returns the final loss and relative checkpoint path. |
| POST | `/eval` | `checkpoint`, `equation`, and optional eval settings. `write_json` is an optional relative `.json` path. |
| POST | `/run` | `{"config": "samples/configs/harmonic.yaml"}` or the experiment document itself, not both. Trains, then evaluates. |

A path or document the schema rejects is HTTP 422. A valid train, eval, or run without torch is HTTP 503. A client over the rate limit is HTTP 429. See [docs/daily/day07.md](docs/daily/day07.md).

## v0.1.0

Day 10 freezes the package at 0.1.0. [PROJECT_STATUS.md](PROJECT_STATUS.md) summarizes Days 1–10, the install, and the limits. [CHANGELOG.md](CHANGELOG.md) is the release note. [docs/architecture.md](docs/architecture.md) describes the components in this tree. Day notes: [docs/daily/day01.md](docs/daily/day01.md), [docs/daily/day02.md](docs/daily/day02.md), [docs/daily/day03.md](docs/daily/day03.md), [docs/daily/day04.md](docs/daily/day04.md), [docs/daily/day05.md](docs/daily/day05.md), [docs/daily/day06.md](docs/daily/day06.md), [docs/daily/day07.md](docs/daily/day07.md), [docs/daily/day08.md](docs/daily/day08.md), [docs/daily/day09.md](docs/daily/day09.md), [docs/daily/day10.md](docs/daily/day10.md).

Authentication stays out of this release.

## Tests

```bash
pip install -e ".[dev,api,lint]"
pytest -m "not ml"
ruff check .
```

Residual, training, eval, and the HTTP train/eval/run tests need torch. `pytest -m "not ml"` is what default CI runs. It includes the catalog and path-sandbox API tests.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev,api]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
