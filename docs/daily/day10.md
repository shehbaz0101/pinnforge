# Day 10

Freeze Project C at v0.1.0. No new product surface. Authentication stays out of this release.

## Shipped

- `PROJECT_STATUS.md` at the repo root. It lists Days 1–10, the install (`pip install torch` from the CPU index, then `pip install -e ".[ml,api]"`), the quickstart (`pinnforge demo`, `pinnforge serve`), CI (`pytest -m "not ml"`, the CPU-torch job, and `ruff check .`), and the known limits: CPU-only training, localhost bind, offline-by-design, and free/public only.
- `CHANGELOG.md` records the v0.1.0 highlights.
- Package version stays `0.1.0` in `pyproject.toml` and `pinnforge.__version__`. That string was already the package version. Day 10 freezes it. No new dependency.
- The annotated tag `v0.1.0` is created on the tip of `feat/day10-freeze` after this freeze commit. A squash merge discards that tip. After the squash SHA is on `main`, retarget the tag:

  ```bash
  git fetch origin main
  git push origin :refs/tags/v0.1.0
  git tag -a v0.1.0 <squash-sha> -m "PINNForge v0.1.0"
  git push origin v0.1.0
  ```

- README opening and the v0.1.0 section point at `PROJECT_STATUS.md` and `CHANGELOG.md`.

## Failed

- Nothing in the Day 10 scope is a new runtime path. The freeze is documentation plus the existing `0.1.0` version string.
- On Python 3.12.3, with the CPU torch wheel installed, `pytest -m "not ml"` passed (190 passed, 6 skipped; those skips are the missing-torch cases) and the full suite passed (271 passed, 6 skipped). `ruff check .` passed with the existing rule selection (E4, E7, E9, F, I). `pinnforge version` printed `pinnforge 0.1.0`. `pinnforge demo --epochs 1` wrote the harmonic summary and exited 0. Python 3.11 was not executed in this environment. CI still runs the suite on 3.11 and 3.12.

## Tomorrow

Project C is complete at v0.1.0. There is no Day 11 in this repository's Project C plan. Authentication stays out of this release.
