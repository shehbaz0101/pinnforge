# Day 9

A one-command offline demo. `pinnforge demo` trains a checked-in sample on CPU, scores it against the analytical or manufactured reference, and prints the loss and the field error. Burgers still has no reference field, so that sample prints residual metrics. There is no authentication.

## Shipped

- `pinnforge demo`. The default sample is `samples/configs/harmonic.yaml` (2 epochs, width `(8, 8)`, `omega: 2`). `--equation poisson` uses `samples/configs/poisson.json`. `--equation burgers` uses `samples/configs/burgers.yaml`. Ids and the short aliases both work. `--epochs` replaces the sample's epoch count in memory, from 1 to 5, and does not rewrite the file. `--data-root` is the Day 8 sandbox root.
- When that relative sample is already inside the root, the demo reads it. When it is not, the checked-in file is copied to the same relative path under the root. Checkpoint, metrics, and eval JSON paths stay the ones in the sample (`runs/<equation>/`). The same text that is printed is written to `summary.txt` next to the eval record.
- The sample path is checked before torch is imported. A path that leaves the root, including a symlink, is an argparse error (exit 2) with or without the `ml` extra. After that check the command installs the offline socket guard. A missing `ml` extra exits with the same `pip install -e ".[ml]"` hint as `pinnforge train`.
- `samples/configs/burgers.yaml` is the third tiny CPU example, next to the harmonic YAML and the Poisson JSON.

## Failed

- Nothing in the Day 9 scope failed its check. On Python 3.12.3, `pytest -m "not ml"` passed before torch was installed (196 tests), and the full suite with a CPU torch wheel passed (271 tests, 6 skipped; those skips are the missing-torch cases, including `pinnforge demo`). `ruff check .` passed with the existing rule selection (E4, E7, E9, F, I). Python 3.11 was not executed in this environment. CI still runs the suite on 3.11 and 3.12.

## Tomorrow

The v0.1.0 freeze: a status note and a changelog. Authentication stays out of scope.
