# PINNForge

PINNForge is a sandbox for physics-informed neural networks on a few classic ODE and PDE residuals. Day 1 is the installable package, the equation schemas, and a closed form for the harmonic oscillator. Day 4 trains a small network on the residual with Adam. Day 5 scores a checkpoint against that closed form, or against a manufactured Poisson field, and always reports the interior residual. Day 6 reads one experiment file and runs that train-then-eval path.

## Install

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Core dependencies are pydantic, numpy, and PyYAML. Tests need the `dev` extra (`pytest`). Lint needs the `lint` extra (`ruff`). The `ml` extra installs torch for the MLP, residual operators, the training loop, and evaluation. Schemas, sampling, experiment files, and `pinnforge sample` do not need it. Default CI skips the torch tests; a separate job installs a CPU wheel and runs them.

## What works today

`pinnforge version` prints `pinnforge 0.1.0`. `pinnforge equations` lists `burgers_1d`, `harmonic_oscillator`, and `poisson_toy`. `pinnforge sample` draws a seeded collocation batch and prints counts and bounds. `--output` writes a JSON record to a relative path in the working directory. With the `ml` extra, `pinnforge residual --equation harmonic --seed 0` builds a tiny untrained MLP and prints its residual MSE. `pinnforge train --equation harmonic --epochs 50 --seed 0` runs Adam on a fixed seeded batch, writes `metrics.jsonl`, and saves a CPU checkpoint under `checkpoints/`. `pinnforge eval --checkpoint checkpoints/checkpoint.pt --equation harmonic` prints L2 and residual metrics for that checkpoint. `--write-json` adds a record with a residual histogram. `pinnforge run --config samples/configs/harmonic.yaml` trains and then evaluates from one relative YAML or JSON file. The same commands work as `python -m pinnforge`.

The pydantic specs describe:

- **Harmonic oscillator.** `u'' + ω² u = 0`, with `omega` or both `k` and `m`, a time interval, initial state `(u, du_dt)`, and optional boundary descriptors. Evaluation compares the network with the closed form.
- **Burgers 1D.** `u_t + u u_x = ν u_xx`, with viscosity, x and t bounds, a named initial profile, and boundary descriptors. The reference function raises `NotImplementedError`. Evaluation reports residual metrics only.
- **Poisson toy.** `-Δu = f` in 1D or 2D, with a named source and boundary descriptors. Each named source has a manufactured field. `sin_pi_x` is `u = sin(π x) / π²`, which matches the default Dirichlet ends on `[0, 1]`.

```bash
pinnforge version
pinnforge equations
pinnforge sample --equation harmonic --n-interior 64 --seed 0
pinnforge residual --equation harmonic --seed 0
pinnforge train --equation harmonic --epochs 50 --seed 0
pinnforge eval --checkpoint checkpoints/checkpoint.pt --equation harmonic
pinnforge run --config samples/configs/harmonic.yaml
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

`samples/configs/harmonic.yaml` and `samples/configs/poisson.json` are tiny CPU examples. Output under `runs/` is gitignored. See [docs/daily/day06.md](docs/daily/day06.md).

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

## Later

A local API, hardening, a demo, and the v0.1.0 freeze. Experiment configs are Day 6. See [docs/architecture.md](docs/architecture.md). Day notes: [docs/daily/day01.md](docs/daily/day01.md), [docs/daily/day02.md](docs/daily/day02.md), [docs/daily/day03.md](docs/daily/day03.md), [docs/daily/day04.md](docs/daily/day04.md), [docs/daily/day05.md](docs/daily/day05.md), [docs/daily/day06.md](docs/daily/day06.md).

## Tests

```bash
pip install -e ".[dev,lint]"
pytest -m "not ml"
ruff check .
```

Residual, training, and eval tests need torch. `pytest -m "not ml"` is what default CI runs.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
