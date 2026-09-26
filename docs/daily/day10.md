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

- Nothing in the Day 10 scope is a new runtime path. The freeze is documentation plus the existing `0.1.0` version string. Local pytest and ruff results are recorded once this note's tree has been checked.

## Tomorrow

Project C is complete at v0.1.0. There is no Day 11 in this repository's Project C plan. Authentication stays out of this release.
