# Stage 1 report

Correctness and packaging on `feat/stage1-correctness`, based on `a9500a865d398cec5ec1a32957d901fbd499b2e7` (tag `v0.1.0`). The version string stays `0.1.0`. These checks were run in this workspace on Python 3.12.3 with torch `2.14.0+cpu`. They are not a claim that the sandbox is finished.

## What changed

1. **Boundary penalties.** Dirichlet, outward Neumann, and periodic terms are all in `loss_bc`. Periodic compares the field and the derivative along that axis at both endpoints, using each boundary row's own free coordinates. A condition the penalty cannot represent raises. `w_bc = 0` still drops the boundary term on purpose.
2. **Poisson references.** A field is returned only when it matches the source, the domain, and every prescribed boundary condition. Source `one` on `[0, 1]` with `u(0) = u(1) = 0` is `x(1 - x) / 2`. No match raises `ReferenceUnavailable`. Burgers stays `NotImplementedError`.
3. **RNG streams.** Train, validation, and test use separate `SeedSequence` values (`seed`, a stream tag, salt `0x50494E4E`). `sample_equation` without an explicit generator still uses `default_rng(seed)`. The validation draw is hashed into `manifest.json` and is not in the loss.
4. **Eval coverage.** Held-out initial and boundary errors, relative L2, and max absolute field error sit beside the residual. A zero field on the default oscillator has a near-zero residual and relative L2 of 1.
5. **Demo packaging.** The three sample configs are copied into `pinnforge.data.configs` and read with `importlib.resources`. A checkout can still walk up to `samples/`.
6. **Solver configuration.** `relu` is rejected. `checkpoint_interval` defaults to 1. Each run writes `manifest.json`. Checkpoints store Adam state, the torch RNG, and history, and `load_checkpoint` uses `weights_only=True`. Resume continues when that state is present and the equation, counts, widths, activation, learning rate, seed, and spec match.

## Breaking changes

- `loss_bc` includes Neumann and periodic penalties. A default periodic Burgers run with boundary samples no longer records `loss_bc` of 0.
- Train and eval points for a given integer seed are not the old `default_rng(seed)` prefix. `pinnforge sample` is unchanged.
- Poisson source `one` with zero ends on `[0, 1]` is no longer `-x²/2`. Specs with no matching field raise `ReferenceUnavailable` instead of returning a field that breaks the boundary data. A single Dirichlet condition `u(0) = 0` can still return `-x²/2`, because that particular solution meets it.
- `activation="relu"` is rejected.
- Eval records add `max_abs_error`, `ic_error`, `bc_error`, `bc_errors`, and `rng`. The format tag stays `pinnforge.eval.v1`.
- Checkpoints add `torch_rng` and, from the trainer, `optimizer` and `history`. The format tag stays `pinnforge.checkpoint.v1`.

## Commands and results

Lint:

```text
python3 -m ruff check .
All checks passed!
```

Full suite, torch installed (the six skips are the no-torch cases in `tests/test_ml_import.py`):

```text
python3 -m pytest -q --tb=line
288 passed, 6 skipped in 4.41s
```

Unit job, same interpreter, torch still installed so the no-torch cases stay skipped:

```text
python3 -m pytest -m "not ml" -q --tb=no
196 passed, 6 skipped, 92 deselected in 1.86s
```

Wheel, fresh environment, empty working directory. `python3 -m venv` failed because `python3.12-venv` is not installed, so `virtualenv` was used:

```text
python3 -m pip install --user build virtualenv
python3 -m build --wheel --outdir /tmp/pinnforge-wheels
python3 -m virtualenv /tmp/pinnforge-wheel-venv
/tmp/pinnforge-wheel-venv/bin/pip install /tmp/pinnforge-wheels/pinnforge-0.1.0-py3-none-any.whl
cd /tmp/pinnforge-empty-cwd
/tmp/pinnforge-wheel-venv/bin/python -c "from pinnforge.demo import bundled_samples_dir, read_bundled_sample; ..."
```

Result: `bundled_samples_dir()` is `None`. `read_bundled_sample` returned `harmonic.yaml` (403 bytes), `burgers.yaml` (499 bytes), and `poisson.json` (415 bytes).

Editable install, already present as `pip install -e ".[dev,api,lint]"`, run from the same empty directory with the user `python3`:

```text
pinnforge.__file__ = /workspace/src/pinnforge/__init__.py
bundled_samples_dir() = /workspace/samples
importlib.resources file = /workspace/src/pinnforge/data/configs/harmonic.yaml
```

Source tree from that empty directory (`PYTHONPATH=/workspace/src`) resolves to the same checkout, finds `/workspace/samples`, and reads `poisson.json` (415 bytes).

## Still limited

- Burgers has no spectral reference. Evaluation cannot report a field error for it.
- `w_bc = 0` does not enforce prescribed boundary conditions. Eval `n_ic = 0` or `n_bc = 0` does not score that condition.
- Resume refuses a checkpoint that lacks optimizer state, torch RNG state, or history, and refuses a change of equation, counts, widths, activation, learning rate, or seed.
- `docs/daily/` still describes the freeze. The current behavior is this file, `PROJECT_STATUS.md`, and `docs/architecture.md`.
- No FNO, PINO, or inverse-viscosity work.
