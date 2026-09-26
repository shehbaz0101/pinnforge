# Day 2

Seeded collocation for the three Day 1 specs. `SampleConfig` sets `n_interior`, `n_ic`, `n_bc`, a seed, and a method: `uniform`, `latin_hypercube`, or `stratified`. `sample_equation` returns a `CollocationBatch` of float64 coordinates in domain-axis order, labeled `interior`, `ic`, and `bc`.

One `numpy.random.default_rng` draws the batch in order: interior, then the initial-time slice, then boundary faces. Interior points lie on `[lower, upper)`. Initial-condition rows sit on the initial slice (`t = t0` for the oscillator and Burgers). Dirichlet and Neumann rows sit on the named side of that axis; a periodic condition samples the minimum face and then the maximum face. The prescribed value is not a coordinate. Poisson has no initial condition, so `n_ic` must be 0. A harmonic spec with an empty boundary list rejects `n_bc > 0`. Counts are non-negative integers, and a config with every count at zero is rejected.

`pinnforge sample --equation harmonic --n-interior 64 --seed 0` prints counts and bounds for the built-in initial-value problem. `--output` writes a JSON record to a relative `.json` path inside the working directory. Fixed-seed batches live in `tests/fixtures/sampling/` (JSON plus a `.npy` of the stacked coordinates) so tests stay offline. Torch is not imported.
