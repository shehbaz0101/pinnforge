# Day 8

Hardening for the localhost API and the CLI paths that read or write a config, a checkpoint, metrics, or JSON. The bind address is still `127.0.0.1`. There is no authentication.

## Shipped

- `pinnforge.specs.paths` is the path sandbox. The root is the working directory, or `PINNFORGE_DATA_ROOT` when that is set. `sample`, `train`, `eval`, `run`, and `serve` take `--data-root`, which sets the variable for that command and then restores it. User paths must be relative to the root. Absolute paths and `~` are rejected. `..` and symlinks are resolved before the check. A path that leaves the root is a `PathSandboxError` (a `ValueError`). The CLI prints it as an argparse error (exit 2). The API returns HTTP 422. A missing file inside the root is still "not found".
- `sample --output`, `train` checkpoint and log paths, `eval` checkpoint and `--write-json`, and `run --config` (including checkpoint, log, and `eval_json` inside the file) are checked before torch is imported. The same rejection happens with or without the `ml` extra.
- `pinnforge.ratelimit`: one in-process sliding window for `POST /train`, `POST /eval`, and `POST /run`, keyed by the client address. The default is 60 requests per 60 seconds. `PINNFORGE_RATE_LIMIT` and `PINNFORGE_RATE_WINDOW_SECONDS` change it. `pinnforge serve` also takes `--rate-limit` and `--rate-window`. Over the limit the response is HTTP 429 with `Retry-After` (seconds until the oldest hit leaves the window). `GET /health` and `GET /equations` are not counted. `TestClient` shares one bucket.
- `pinnforge.offline`: `create_app`, and the `serve`, `train`, `eval`, and `run` commands, wrap `socket` connect. A non-loopback TCP target raises `OfflineError`. Loopback stays open so the server can accept local clients. The guard sets `PINNFORGE_OFFLINE=1` and does not offer a way to disable itself. Optional hub and telemetry variables are set if they were unset (`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_DATASETS_OFFLINE`, `WANDB_MODE`, `DO_NOT_TRACK`). None of those libraries are dependencies.
- `pinnforge serve` still defaults to `127.0.0.1`. `0.0.0.0` and `::` still need `--allow-remote`.

## Failed

- Nothing in the Day 8 scope failed its check. On Python 3.12.3, `pytest -m "not ml"` passed (186 tests) before torch was installed, and the full suite with a CPU torch wheel passed (260 tests, 5 skipped; those skips are the missing-torch cases). `ruff check .` passed with the existing rule selection (E4, E7, E9, F, I). Python 3.11 was not executed in this environment. CI still runs the suite on 3.11 and 3.12.

## Tomorrow

A demo command. Authentication stays out of scope.
