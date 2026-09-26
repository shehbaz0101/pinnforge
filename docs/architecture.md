# Architecture

PINNForge trains a small network on a classic residual and scores it against an analytical or fixed reference. The package version is 0.1.0. The pre-release freeze is Day 10. This tree has the Day 1 equation schemas and harmonic closed form, the Day 2 samplers, the Day 3 MLP and residual operators, the Day 4 Adam trainer, the Day 5 evaluation metrics, the Day 6 experiment config, the Day 7 localhost HTTP API, the Day 8 path sandbox, rate limit, and offline guard, and the Day 9 offline demo.

Importing `pinnforge` loads pydantic specs and the numpy samplers. It does not import torch or FastAPI. Torch is the optional `ml` extra. FastAPI and uvicorn are the optional `api` extra. `pinnforge.models`, `pinnforge.residuals`, `pinnforge.losses`, `pinnforge.training`, and `pinnforge.evaluation` raise `InstallHint` when torch is missing. `pinnforge.specs` and `pinnforge.experiments` validate train, eval, and experiment files without torch. `pinnforge.api` imports FastAPI only when the app is built. Default CI runs `pytest -m "not ml"` without torch and with the `api` extra installed. A separate job installs a CPU wheel and runs the full suite, including the residual, training, eval, `pinnforge run`, and HTTP train/eval/run tests.

## Today

| Piece | Now |
| --- | --- |
| Harmonic oscillator | Spec with ω or k and m, a time interval, state IC `(u, du_dt)`, optional BC descriptors. Numpy solution `u(t) = A cos(ω(t-t0)) + B sin(ω(t-t0))`. |
| Burgers 1D | Spec with viscosity ν > 0, x and t intervals, a named IC profile, and BC descriptors. Reference raises `NotImplementedError`. Eval reports residual metrics only. |
| Poisson toy | Spec with dimension 1 or 2, a named source, and BC descriptors. Manufactured fields: `u = 0`, `u = -x² / 2`, `u = sin(π x) / π²`, and `u = sin(π x) sin(π y) / (2 π²)`. |
| Registry | In-process map from `equation_id` to the spec class, plus a catalog of the three built-ins (`list_equations`, aliases, overridable parameters). |
| Sampler | Uniform, Latin-hypercube, or stratified draws on a `CollocationDomain`, plus IC and BC faces. A `CollocationBatch` holds float64 coordinates and `interior` / `ic` / `bc` labels. |
| MLP | Fully connected field `u`. Input width comes from the collocation domain. Default activation is `tanh`. |
| Residual | `residual(model, coords, spec)` returns ü + ω²u, u_t + u u_x − ν u_xx, or −Δu − f. Derivatives use `torch.autograd.grad` with `create_graph=True`. |
| Penalties | Soft mean-squared initial-condition and Dirichlet terms from a `CollocationBatch` and the spec. Neumann and periodic faces are not penalized. |
| Trainer | Adam on one seeded batch. Loss is `w_pde *` residual MSE plus weighted soft IC and Dirichlet penalties. Writes `metrics.jsonl` and a CPU checkpoint (`state_dict`, config, spec, epoch). |
| Eval | Load that checkpoint or pass an in-memory model. Interior mean and max `|residual|`, a numpy histogram, and L2 / relative L2 where a reference field exists. |
| CLI | `pinnforge version`, `pinnforge equations`, `pinnforge sample`, `pinnforge residual`, `pinnforge train`, `pinnforge eval`, `pinnforge run --config`, `pinnforge demo`, and `pinnforge serve`. Residual, train, eval, run, and demo need torch. Serve needs the `api` extra and binds to `127.0.0.1`. |
| Config | `ExperimentConfig` loads a relative YAML or JSON file: equation id or inline spec, optional parameter overrides, nested train and eval settings, MLP widths, and relative output paths. |
| API | `GET /health`, `GET /equations`, `GET /equations/{id_or_alias}`, `POST /train`, `POST /eval`, and `POST /run`. Paths stay inside the sandbox root. Those three POSTs share a per-client rate limit (HTTP 429). The process refuses non-loopback TCP connects. `0.0.0.0` is refused unless `pinnforge serve --allow-remote` is set. |

Shared types (`Interval`, `CollocationDomain`, `StateInitialCondition`, `ProfileInitialCondition`, `BoundaryCondition`) are what the samplers read. Specs reject unknown fields.

## Sampling

Day 2 draws points from a `CollocationDomain` and does not build a residual. Interior points use one seeded `numpy.random.Generator` on the half-open box `[lower, upper)`. Initial-condition points sit on the initial-time face (`t = t0` for the oscillator and Burgers). Boundary points sit on the faces in `boundary_conditions`: Dirichlet and Neumann fix the named side, and a periodic condition samples both ends. The generator continues from interior to IC to BC, so a seed reproduces the whole batch. Poisson rejects `n_ic > 0`. A harmonic spec with no boundary descriptors rejects `n_bc > 0`. Counts are non-negative integers; a config that asks for no points at all is rejected. `pinnforge sample` prints counts and bounds and can write a JSON record under a relative path in the working directory. Offline fixtures live in `tests/fixtures/sampling/`.

## Residuals

`mlp_from_spec` builds a network whose input width is the number of collocation axes. `residual` evaluates that network (or any callable with the same signature) and differentiates the scalar field with `create_graph=True`. The oscillator residual is ü + ω²u. Burgers uses columns `(x, t)` and the residual u_t + u u_x − ν u_xx. Poisson follows the Day 1 statement −Δu = f, so the residual is −Δu − f for the named source. `soft_penalty` reads `ic` and Dirichlet `bc` rows from a `CollocationBatch` and compares them with the prescribed spec values. `pinnforge residual --equation harmonic --seed 0` samples 16 interior points, builds an untrained width-(8, 8) network, and prints the residual MSE.

## Training

Day 4 minimizes those terms with Adam on CPU. `TrainConfig` names the built-in equation, sample counts, MLP widths, epoch count, learning rate, loss weights, seed, and relative output paths. The loop draws one batch with that seed and reuses it every epoch, so `metrics.jsonl` compares the same collocation points from epoch 0 (initial weights) through the last Adam step. The scalar loss is `w_pde * MSE(residual) + w_ic * initial + w_bc * dirichlet`. Each checkpoint is a `torch.save` dict with the model `state_dict`, the config, the epoch, and the equation spec that was trained. `load_checkpoint` rebuilds the MLP from that spec. Files written before the spec field existed still load the built-in problem. Optimizer state is not stored. `pinnforge train --equation harmonic --epochs 50 --seed 0` is the small CPU demo.

## Evaluation

Day 5 scores that checkpoint, or any in-memory callable with the same signature, on a new interior batch. `evaluate_model` reports the mean and max of `|residual|` and a `numpy.histogram` of those absolute residuals. Harmonic L2 uses the Day 1 closed form. Poisson L2 uses the manufactured field for the named source. Burgers still raises `NotImplementedError` from its reference hook, and the eval record leaves `l2` and `relative_l2` null. Relative L2 is `l2 / rms(reference)` and is also null when the reference is identically zero. `pinnforge eval --checkpoint checkpoints/checkpoint.pt --equation harmonic` prints the summary. `--write-json` writes `pinnforge.eval.v1`. Checkpoint and JSON paths stay inside the working directory. See [daily/day05.md](daily/day05.md).

## Experiment config

Day 6 drives train and eval from one file. `ExperimentConfig` accepts `equation: harmonic` (or `burgers` or `poisson`, or the registry id) plus optional `equation_params`, or an inline spec mapping. Overrides are merged onto the built-in and checked by the Day 1 schema. `train` and `eval` are the existing config models. `hidden_widths`, `checkpoint_dir`, and `log_path` live on `train`. `eval_json` is the optional record path. `pinnforge run --config path` is the command. The path must be a relative `.yaml`, `.yml`, or `.json` file inside the working directory. See [daily/day06.md](daily/day06.md).

## Local API

Day 7 exposes the catalog and the train-then-eval path over HTTP. `create_app` in `pinnforge.api` builds the FastAPI app. `pinnforge serve` runs it with uvicorn on `127.0.0.1:8000`. `0.0.0.0` and `::` are refused unless `--allow-remote` is set. `GET /health` and `GET /equations` do not import torch. `GET /equations/{id_or_alias}` returns the catalog entry and the default spec. `POST /run` takes a relative config path or the experiment document. `POST /train` and `POST /eval` take the same settings as the flag-based commands. Config, checkpoint, log, and JSON paths are resolved with the sandbox before torch is imported. A missing `ml` extra on those routes is HTTP 503. There is no authentication. See [daily/day07.md](daily/day07.md).

Day 8 tightens that sandbox and adds the other two gates. The root is the working directory, or `PINNFORGE_DATA_ROOT`, or `--data-root` on `sample`, `train`, `eval`, `run`, `demo`, and `serve`. Absolute paths and `~` are rejected. `..` and symlinks are resolved, and a path that leaves the root is HTTP 422 or a CLI error. `POST /train`, `POST /eval`, and `POST /run` share a sliding window of 60 requests per 60 seconds per client address (`PINNFORGE_RATE_LIMIT`, `PINNFORGE_RATE_WINDOW_SECONDS`, or the matching `serve` flags). Over the limit the response is HTTP 429 with `Retry-After`. Catalog reads are not counted. `create_app` and those CLI commands wrap socket connect so a non-loopback target raises `OfflineError`. Loopback stays open. The guard cannot be turned off. See [daily/day08.md](daily/day08.md).

## Demo

Day 9 is one command over the Day 6 train-then-eval path. `pinnforge demo` loads `samples/configs/harmonic.yaml` unless `--equation` names `poisson` (`samples/configs/poisson.json`) or `burgers` (`samples/configs/burgers.yaml`). The registry id and the short alias are the same sample. The files are 2-epoch, width-(8, 8) CPU runs. `--epochs` may set 1 through 5 and does not edit the file. `--data-root` is the sandbox root. If the relative sample is not already inside that root, the checked-in file is copied there and then loaded. The printed summary is also `summary.txt` beside the eval JSON. Harmonic and Poisson report L2 against the closed form or the manufactured field. Burgers reports residual metrics only. The command installs the offline guard before it imports torch. See [daily/day09.md](daily/day09.md).

## Later days

1. **Day 2 — Samplers.** Done. Collocation, IC, and BC draws for the three specs. See [daily/day02.md](daily/day02.md).
2. **Day 3 — PINN MLP.** Done. CPU MLP, autograd residuals, and soft IC/BC penalties. See [daily/day03.md](daily/day03.md).
3. **Day 4 — Train.** Done. Adam on a fixed seeded batch, metrics, and CPU checkpoints. See [daily/day04.md](daily/day04.md).
4. **Day 5 — Eval.** Done. Harmonic and Poisson field error, residual stats, and a checkpoint CLI. Burgers is residual-only. See [daily/day05.md](daily/day05.md).
5. **Day 6 — Config.** Done. An experiment file selects a built-in equation, optional parameter overrides, and train/eval settings. `pinnforge run --config` trains and then evaluates. The checkpoint stores the spec. See [daily/day06.md](daily/day06.md).
6. **Day 7 — Serve.** Done. Localhost FastAPI app and `pinnforge serve`. Health, equation catalog, train, eval, and run. Loopback by default. `0.0.0.0` needs `--allow-remote`. No authentication. See [daily/day07.md](daily/day07.md).
7. **Day 8 — Harden.** Done. Path sandbox root, per-client rate limit on the train/eval/run routes, and an offline socket guard. See [daily/day08.md](daily/day08.md).
8. **Day 9 — Demo.** Done. `pinnforge demo` trains a checked-in sample and prints the loss and the reference error. See [daily/day09.md](daily/day09.md).
9. **Day 10 — Freeze.** v0.1.0 pre-release freeze: status note and changelog.

Day 10 is still the plan. This tree does not implement it.
