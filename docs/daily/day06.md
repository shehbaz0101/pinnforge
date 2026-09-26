# Day 6

Experiment configs. One relative YAML or JSON file names the equation, the training settings, and the evaluation settings. `pinnforge run --config samples/configs/harmonic.yaml` trains that problem and then scores the checkpoint. The flag-based `train` and `eval` commands are unchanged.

`ExperimentConfig` is a pydantic model. `equation` is a registry id or a short alias (`harmonic`, `burgers`, `poisson`), or a full inline Day 1 spec. `equation_params` merges onto the built-in spec when `equation` is an id. Nested mappings merge field by field, and lists such as boundary conditions are replaced. The merged document is validated by the Day 1 schema, so a negative `omega` or a Poisson source in the wrong dimension is rejected. An inline spec cannot also carry `equation_params`. `equation_id` is accepted as another name for the string form, and not together with `equation`.

`train` holds `TrainConfig` fields. The equation id comes from `equation`, so the train block does not have to repeat it. Omitted `n_ic` and `n_bc` still use the Day 2 per-equation defaults. `hidden_widths` is the MLP width list. `checkpoint_dir` and `log_path` stay relative paths. `eval` holds `EvalConfig` fields: interior count, seed, method, and histogram bins. `eval_json`, when set, is a relative `.json` path for the eval record.

The config path itself is sandboxed the same way as checkpoints and sample output. It must be a relative `.yaml`, `.yml`, or `.json` path, and after `..` and symlinks are resolved it must stay inside the working directory. Absolute paths are rejected. Loading the file does not import torch. `pinnforge.specs` holds `TrainConfig` and `EvalConfig` so that check can run without the `ml` extra. `pinnforge run` imports torch only after the file loads, and without torch it prints the same install hint as `train`.

The registry now documents the three built-ins. `list_equations` returns an `EquationInfo` for each id: short aliases, the residual statement, and the parameter names a config may override. `get_equation` accepts the id or the alias and returns the spec class. `build_equation` is the selection used by the config. The catalog and the registered classes are the same set.

Training can take that spec instead of always rebuilding the built-in. `train_loop(..., spec=...)` draws the batch and the residual from the spec, and the checkpoint stores it next to the config and the `state_dict`. The format tag is still `pinnforge.checkpoint.v1`. Older checkpoints omit `spec` and still load the built-in problem. `evaluate_checkpoint` scores the stored spec, so a run with `omega: 2` is not later graded as `omega: 1`. `run` evaluates the in-memory network on that same spec before it writes `eval_json`.

`samples/configs/harmonic.yaml` is a 2-epoch oscillator with `omega: 2`. `samples/configs/poisson.json` is a 2-epoch 1D Poisson problem with source `sin_pi_x`. Both use a width-(8, 8) network and write under `runs/`. That directory is gitignored, along with `checkpoints/` and `metrics.jsonl`.

Tests that only load and reject configs are not marked `ml`. A few-epoch `run` and a checkpoint reload of the overridden spec are marked `ml`. Default CI still runs `pytest -m "not ml"` without torch.
