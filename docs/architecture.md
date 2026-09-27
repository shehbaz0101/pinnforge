# Architecture

PINNForge trains a small network on a classic residual and scores it against an analytical or fixed reference. The package version is 0.1.0. Day 10 freezes that version. [PROJECT_STATUS.md](../PROJECT_STATUS.md) is the status note. This tree has the Day 1 equation schemas and harmonic closed form, the Day 2 samplers, the Day 3 MLP and residual operators, the Day 4 Adam trainer, the Day 5 evaluation metrics, the Day 6 experiment config, the Day 7 localhost HTTP API, the Day 8 path sandbox, rate limit, and offline guard, and the Day 9 offline demo.

Importing `pinnforge` loads pydantic specs and the numpy samplers. It does not import torch or FastAPI. Torch is the optional `ml` extra. FastAPI and uvicorn are the optional `api` extra. `pinnforge.models`, `pinnforge.residuals`, `pinnforge.losses`, `pinnforge.training`, and `pinnforge.evaluation` raise `InstallHint` when torch is missing. `pinnforge.specs` and `pinnforge.experiments` validate train, eval, and experiment files without torch. `pinnforge.api` imports FastAPI only when the app is built. Default CI runs `pytest -m "not ml"` without torch and with the `api` extra installed. A separate job installs a CPU wheel and runs the full suite, including the residual, training, eval, `pinnforge run`, and HTTP train/eval/run tests.

## Today

| Piece | Now |
| --- | --- |
| Harmonic oscillator | Spec with ω or k and m, a time interval, state IC `(u, du_dt)`, optional BC descriptors. Numpy solution `u(t) = A cos(ω(t-t0)) + B sin(ω(t-t0))`. |
| Burgers 1D | Spec with viscosity ν > 0, x and t intervals, a named IC profile, and BC descriptors. Reference raises `NotImplementedError`. Eval reports residual metrics only. |
| Poisson toy | Spec with dimension 1 or 2, a named source, and BC descriptors. A reference field is returned only when it matches the source, the domain, and every prescribed boundary condition. Source `one` on `[0, 1]` with zero Dirichlet ends is `x(1 - x) / 2`. Otherwise evaluation marks the reference unavailable. |
| Registry | In-process map from `equation_id` to the spec class, plus a catalog of the three built-ins (`list_equations`, aliases, overridable parameters). |
| Sampler | Uniform, Latin-hypercube, or stratified draws on a `CollocationDomain`, plus IC and BC faces. A `CollocationBatch` holds float64 coordinates and `interior` / `ic` / `bc` labels. |
| MLP | Fully connected field `u`. Input width comes from the collocation domain. Activations are `tanh` (default) and `silu`. `relu` is rejected because a strong-form second derivative is zero almost everywhere. |
| Residual | `residual(model, coords, spec)` returns ü + ω²u, u_t + u u_x − ν u_xx, or −Δu − f. Derivatives use `torch.autograd.grad` with `create_graph=True`. |
| Penalties | Soft mean-squared initial, Dirichlet, outward-Neumann, and periodic terms. Periodic matches the field and its derivative along that axis at both endpoints, at each boundary row's other coordinates. An unsupported condition raises. `w_bc = 0` drops the boundary terms on purpose. |
| Trainer | Adam on one batch from the train RNG stream. The validation stream is hashed into `manifest.json` and is not in the loss. Loss is `w_pde *` residual MSE plus weighted initial and boundary penalties. Checkpoints store weights, the spec, Adam state, the torch RNG, and history, and load with `weights_only=True`. |
| Eval | Test-stream collocation, including held-out initial and boundary rows when the equation has them. Residual mean and max, L2, relative L2, max absolute field error when a reference exists, and held-out IC and BC errors. |
| CLI | `pinnforge version`, `pinnforge equations`, `pinnforge sample`, `pinnforge residual`, `pinnforge train`, `pinnforge eval`, `pinnforge run --config`, `pinnforge demo`, `pinnforge serve`, and `pinnforge fno`. Residual, train, eval, run, demo, and `fno train`/`fno eval` need torch. Serve needs the `api` extra and binds to `127.0.0.1`. `fno` is the Burgers window operator, with a data, residual, or hybrid loss. It is not the coordinate-PINN trainer. |
| Config | `ExperimentConfig` loads a relative YAML or JSON file: equation id or inline spec, optional parameter overrides, nested train and eval settings, MLP widths, and relative output paths. |
| API | `GET /health`, `GET /equations`, `GET /equations/{id_or_alias}`, `POST /train`, `POST /eval`, and `POST /run`. Paths stay inside the sandbox root. Those three POSTs share a per-client rate limit (HTTP 429). The process refuses non-loopback TCP connects. `0.0.0.0` is refused unless `pinnforge serve --allow-remote` is set. |

Shared types (`Interval`, `CollocationDomain`, `StateInitialCondition`, `ProfileInitialCondition`, `BoundaryCondition`) are what the samplers read. Specs reject unknown fields.

## Sampling

Day 2 draws points from a `CollocationDomain` and does not build a residual. Interior points use one seeded `numpy.random.Generator` on the half-open box `[lower, upper)`. Initial-condition points sit on the initial-time face (`t = t0` for the oscillator and Burgers). Boundary points sit on the faces in `boundary_conditions`: Dirichlet and Neumann fix the named side, and a periodic condition samples both ends. The generator continues from interior to IC to BC, so a seed reproduces the whole batch. Poisson rejects `n_ic > 0`. A harmonic spec with no boundary descriptors rejects `n_bc > 0`. Counts are non-negative integers; a config that asks for no points at all is rejected. `pinnforge sample` prints counts and bounds and can write a JSON record under a relative path in the working directory. Offline fixtures live in `tests/fixtures/sampling/`.

## Residuals

`mlp_from_spec` builds a network whose input width is the number of collocation axes. `residual` evaluates that network (or any callable with the same signature) and differentiates the scalar field with `create_graph=True`. The oscillator residual is ü + ω²u. Burgers uses columns `(x, t)` and the residual u_t + u u_x − ν u_xx. Poisson follows the Day 1 statement −Δu = f, so the residual is −Δu − f for the named source. `soft_penalty` reads `ic` rows and Dirichlet, Neumann, and periodic `bc` rows. Dirichlet matches the prescribed value. Neumann matches the outward normal derivative (side `max` is `+∂/∂variable`, side `min` is `-∂/∂variable`). Periodic matches the field and the derivative along that axis at both endpoints, using each boundary row's own free coordinates, so Burgers compares the two ends at the same time. A condition this penalty cannot represent raises `ValueError`. `relu` is not an allowed activation. `pinnforge residual --equation harmonic --seed 0` samples 16 interior points, builds an untrained width-(8, 8) network, and prints the residual MSE.

## Training

Day 4 minimizes those terms with Adam on CPU. `TrainConfig` names the built-in equation, sample counts, MLP widths, epoch count, learning rate, loss weights, seed, and relative output paths. The loop draws one batch from the train RNG stream (`SeedSequence` of the seed, the name `train`, and a fixed salt) and reuses it every epoch, so `metrics.jsonl` compares the same collocation points from epoch 0 (initial weights) through the last Adam step. `pinnforge sample` still uses `numpy.random.default_rng(seed)`. The training batch is not that stream. A validation batch is drawn from the validation stream, hashed, and written to `manifest.json`. It is not part of the loss. The scalar loss is `w_pde * MSE(residual) + w_ic * initial + w_bc * boundary`, where `boundary` is the Dirichlet, outward-Neumann, and periodic penalties. `w_bc = 0` skips boundary enforcement. `w_bc > 0` with a prescribed face that has no samples raises `ValueError`. Each checkpoint is a `torch.save` dict with the model `state_dict`, the config, the epoch, the equation spec, the torch RNG state, the Adam state, and the metrics history. The format tag stays `pinnforge.checkpoint.v1`. `load_checkpoint` rebuilds the MLP and loads with `weights_only=True`. Files written before the spec field existed still load the built-in problem. `checkpoint_interval` defaults to 1; epoch 0 and the last epoch are always written. `resume_from` continues a run whose equation, counts, widths, activation, learning rate, seed, and spec match, and whose checkpoint stored the optimizer, the torch RNG, and the history. `pinnforge train --equation harmonic --epochs 50 --seed 0` is the small CPU demo.

## Evaluation

Day 5 scores that checkpoint, or any in-memory callable with the same signature, on a batch from the test RNG stream. The integer seed is `EvalConfig.seed`. That stream is not `default_rng(seed)` and is not the train stream, including when the seed and the interior count match. `evaluate_model` reports the mean and max of `|residual|`, a `numpy.histogram` of those absolute residuals, and, when a reference exists, L2, relative L2, and max absolute field error. Held-out initial-condition error and boundary error are reported when those rows are drawn. Omitted `n_ic` and `n_bc` draw a small held-out set (none for a Poisson initial condition; one row per boundary face). Explicit `n_ic = 0` or `n_bc = 0` leaves that score null. Harmonic L2 uses the Day 1 closed form. Poisson L2 uses a reference only when the source, domain, and every prescribed boundary condition match; source `one` on `[0, 1]` with zero ends is `x(1 - x) / 2`. A mismatch raises `ReferenceUnavailable`, and the eval record leaves the field errors null. Burgers still raises `NotImplementedError` from its reference hook. Relative L2 is `l2 / rms(reference)` and is also null when the reference is identically zero. A zero field on the default oscillator has a near-zero residual and relative L2 of 1, because `u(0) = 1` is missed. `pinnforge eval --checkpoint checkpoints/checkpoint.pt --equation harmonic` prints the summary, including those condition scores and the test-stream identity. `--write-json` writes `pinnforge.eval.v1`. Checkpoint and JSON paths stay inside the working directory. See [daily/day05.md](daily/day05.md).

## Experiment config

Day 6 drives train and eval from one file. `ExperimentConfig` accepts `equation: harmonic` (or `burgers` or `poisson`, or the registry id) plus optional `equation_params`, or an inline spec mapping. Overrides are merged onto the built-in and checked by the Day 1 schema. `train` and `eval` are the existing config models. `hidden_widths`, `checkpoint_dir`, and `log_path` live on `train`. `eval_json` is the optional record path. `pinnforge run --config path` is the command. The path must be a relative `.yaml`, `.yml`, or `.json` file inside the working directory. See [daily/day06.md](daily/day06.md).

## Local API

Day 7 exposes the catalog and the train-then-eval path over HTTP. `create_app` in `pinnforge.api` builds the FastAPI app. `pinnforge serve` runs it with uvicorn on `127.0.0.1:8000`. `0.0.0.0` and `::` are refused unless `--allow-remote` is set. `GET /health` and `GET /equations` do not import torch. `GET /equations/{id_or_alias}` returns the catalog entry and the default spec. `POST /run` takes a relative config path or the experiment document. `POST /train` and `POST /eval` take the same settings as the flag-based commands. Config, checkpoint, log, and JSON paths are resolved with the sandbox before torch is imported. A missing `ml` extra on those routes is HTTP 503. There is no authentication. See [daily/day07.md](daily/day07.md).

Day 8 tightens that sandbox and adds the other two gates. The root is the working directory, or `PINNFORGE_DATA_ROOT`, or `--data-root` on `sample`, `train`, `eval`, `run`, `demo`, and `serve`. Absolute paths and `~` are rejected. `..` and symlinks are resolved, and a path that leaves the root is HTTP 422 or a CLI error. `POST /train`, `POST /eval`, and `POST /run` share a sliding window of 60 requests per 60 seconds per client address (`PINNFORGE_RATE_LIMIT`, `PINNFORGE_RATE_WINDOW_SECONDS`, or the matching `serve` flags). Over the limit the response is HTTP 429 with `Retry-After`. Catalog reads are not counted. `create_app` and those CLI commands wrap socket connect so a non-loopback target raises `OfflineError`. Loopback stays open. The guard cannot be turned off. See [daily/day08.md](daily/day08.md).

## Demo

Day 9 is one command over the Day 6 train-then-eval path. `pinnforge demo` loads `samples/configs/harmonic.yaml` unless `--equation` names `poisson` (`samples/configs/poisson.json`) or `burgers` (`samples/configs/burgers.yaml`). The registry id and the short alias are the same sample. The files are 2-epoch, width-(8, 8) CPU runs. `--epochs` may set 1 through 5 and does not edit the file. `--data-root` is the sandbox root. If the relative sample is not already inside that root, the command writes it there from package data (`importlib.resources` on `pinnforge.data.configs`) and then loads it. A checkout can still fall back to the repository `samples/` tree. The printed summary is also `summary.txt` beside the eval JSON. Harmonic and Poisson report L2 against the closed form or a matched Poisson field. Burgers reports residual metrics only. The command installs the offline guard before it imports torch. See [daily/day09.md](daily/day09.md).

## Stage 1 correctness

The tagged v0.1.0 tree skipped Neumann and periodic penalties, returned `u = -x² / 2` for every Poisson source `one`, drew train and eval points from one `default_rng` stream, and found demo configs only by walking parents of the source file. Those gaps are covered by regression tests. [STAGE1_REPORT.md](../STAGE1_REPORT.md) records the commands. The package version string is still `0.1.0`. The Burgers evaluation hook still raises `NotImplementedError`. Periodic trajectory labels are a separate Fourier package, described in [STAGE2_REPORT.md](../STAGE2_REPORT.md). `docs/daily/` keeps the day-by-day notes from the freeze and is not a description of this branch.

## Stage 2 Burgers reference

`pinnforge.reference.numerical` integrates periodic viscous Burgers on `x ∈ [-1, 1]`, `t ∈ [0, 1]` with a dealiased Fourier method and ETDRK4. It does not train a network and it is not called by `evaluate_model`. `pinnforge demo --equation burgers` still prints residual metrics. The convergence numbers, the pilot manifest, and the generation commands are in [STAGE2_REPORT.md](../STAGE2_REPORT.md).

## Stage 3 data-only FNO

`pinnforge.operator` cuts temporal windows from those trajectories after the instance split in `docs/stage2/pilot_manifest.json`. A 1D Fourier neural operator maps the input window plus a viscosity channel to the following window. The default loss is mean squared error in the manifest's normalized `u`. Relative L2 for model selection and for the test score is computed after denormalizing. `pinnforge fno` and `python -m pinnforge.operator` are the commands. They do not call `reference_solution`, and `evaluate_model` does not call them. Measured test error, the window length, and the claims that are out of scope are in [STAGE3_REPORT.md](../STAGE3_REPORT.md).

## Stage 4 residual losses

The same operator accepts `--loss residual` and `--loss hybrid`. The residual is the pilot spectral derivative in space and a central difference in time, with the instance viscosity known. Checkpoint selection stays on validation mean relative L2. The stencil, the preregistered weights, and the test table are in [STAGE4_REPORT.md](../STAGE4_REPORT.md).

## Later days

1. **Day 2 — Samplers.** Done. Collocation, IC, and BC draws for the three specs. See [daily/day02.md](daily/day02.md).
2. **Day 3 — PINN MLP.** Done. CPU MLP, autograd residuals, and soft IC/BC penalties. See [daily/day03.md](daily/day03.md).
3. **Day 4 — Train.** Done. Adam on a fixed seeded batch, metrics, and CPU checkpoints. See [daily/day04.md](daily/day04.md).
4. **Day 5 — Eval.** Done. Harmonic and Poisson field error, residual stats, and a checkpoint CLI. Burgers is residual-only. See [daily/day05.md](daily/day05.md).
5. **Day 6 — Config.** Done. An experiment file selects a built-in equation, optional parameter overrides, and train/eval settings. `pinnforge run --config` trains and then evaluates. The checkpoint stores the spec. See [daily/day06.md](daily/day06.md).
6. **Day 7 — Serve.** Done. Localhost FastAPI app and `pinnforge serve`. Health, equation catalog, train, eval, and run. Loopback by default. `0.0.0.0` needs `--allow-remote`. No authentication. See [daily/day07.md](daily/day07.md).
7. **Day 8 — Harden.** Done. Path sandbox root, per-client rate limit on the train/eval/run routes, and an offline socket guard. See [daily/day08.md](daily/day08.md).
8. **Day 9 — Demo.** Done. `pinnforge demo` trains a checked-in sample and prints the loss and the reference error. See [daily/day09.md](daily/day09.md).
9. **Day 10 — Freeze.** Done. v0.1.0 status note and changelog. See [daily/day10.md](daily/day10.md).

v0.1.0 stops here. Authentication stays out of this release.
