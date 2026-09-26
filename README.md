# PINNForge

PINNForge is a sandbox for physics-informed neural networks on a few classic ODE and PDE residuals. Day 1 is the installable package, the equation schemas, and a closed form for the harmonic oscillator. Training comes later.

## Install

Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Core dependencies are pydantic and numpy. Tests need the `dev` extra (`pytest`). Lint needs the `lint` extra (`ruff`). The `ml` extra installs torch for the MLP and residual operators. Schemas, sampling, and `pinnforge sample` do not need it. Default CI skips the torch tests; a separate job installs a CPU wheel and runs them.

## What works today

`pinnforge version` prints `pinnforge 0.1.0`. `pinnforge equations` lists `burgers_1d`, `harmonic_oscillator`, and `poisson_toy`. `pinnforge sample` draws a seeded collocation batch and prints counts and bounds. `--output` writes a JSON record to a relative path in the working directory. With the `ml` extra, `pinnforge residual --equation harmonic --seed 0` builds a tiny untrained MLP and prints its residual MSE. The same commands work as `python -m pinnforge`.

The pydantic specs describe:

- **Harmonic oscillator.** `u'' + ω² u = 0`, with `omega` or both `k` and `m`, a time interval, initial state `(u, du_dt)`, and optional boundary descriptors.
- **Burgers 1D.** `u_t + u u_x = ν u_xx`, with viscosity, x and t bounds, a named initial profile, and boundary descriptors. The reference function raises `NotImplementedError`.
- **Poisson toy.** `-Δu = f` in 1D or 2D, with a named source and boundary descriptors. The reference function raises `NotImplementedError`.

```bash
pinnforge version
pinnforge equations
pinnforge sample --equation harmonic --n-interior 64 --seed 0
pinnforge residual --equation harmonic --seed 0
```

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

Training, evaluation, a persisted run registry, a local API, hardening, a demo, and the v0.1.0 freeze. The MLP and residual operators are Day 3; the training loop is Day 4. See [docs/architecture.md](docs/architecture.md). Day notes: [docs/daily/day01.md](docs/daily/day01.md), [docs/daily/day02.md](docs/daily/day02.md), [docs/daily/day03.md](docs/daily/day03.md).

## Tests

```bash
pip install -e ".[dev,lint]"
pytest -m "not ml"
ruff check .
```

Residual tests need torch. `pytest -m "not ml"` is what default CI runs.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
