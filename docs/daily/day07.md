# Day 7

A localhost HTTP API over the equation catalog and the Day 6 train-then-eval path. `pinnforge serve` binds to `127.0.0.1:8000`. `GET /health` and `GET /equations` do not import torch. `POST /train`, `POST /eval`, and `POST /run` do, and they need the `ml` extra.

`create_app` in `pinnforge.api` builds the FastAPI app. `pinnforge.api:app` is that app for uvicorn. FastAPI and uvicorn are the optional `api` extra, separate from torch. Importing `pinnforge` does not import either one. `pinnforge serve` prints an install hint when the extra is missing, the same way `train` does for torch.

The bind address defaults to `127.0.0.1`. `--host` can name another address. `0.0.0.0` and `::` are refused unless `--allow-remote` is also set, so a listen on every interface has to be explicit. Starting uvicorn by hand bypasses that check; pass `--host 127.0.0.1` there. There is no authentication and no rate limit. Those stay on the Day 8 list.

`GET /equations` lists the registry catalog: id, aliases, the residual summary, and the parameter names a config may override. `GET /equations/{id_or_alias}` adds the default spec for that built-in. `harmonic` and `harmonic_oscillator` are the same entry. An unknown name is HTTP 404.

`POST /run` accepts either `{"config": "samples/configs/harmonic.yaml"}` or the same experiment document `pinnforge run --config` reads. The config path, checkpoint directory, metrics path, and `eval_json` must stay inside the working directory after `..` and symlinks are resolved. That check runs before torch is imported. A path that escapes is HTTP 422 with or without the `ml` extra. A missing torch install on a valid body is HTTP 503. The response carries the final training loss, the relative checkpoint path, and the eval record (`pinnforge.eval.v1`).

`POST /train` takes a `TrainConfig` body. `POST /eval` takes a relative `.pt` checkpoint, the equation id or alias, and optional `EvalConfig` fields. `write_json` is a relative `.json` path for the eval record. Both routes use the same path sandbox as the CLI. An equation name that does not match the checkpoint is HTTP 422.

Tests call the app with FastAPI's `TestClient` and do not open a port. `pinnforge serve` is checked by replacing `uvicorn.run`, including the refusal of `0.0.0.0`. The torch routes are marked `ml`. `pytest -m "not ml"` covers health, the catalog, path rejection, and the serve bind check.
