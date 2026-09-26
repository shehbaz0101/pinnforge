# Day 5

Evaluation for a trained checkpoint or an in-memory model. `EvalConfig` names an interior count, a sampler seed and method, and a residual-histogram bin count. The default is 64 interior points, seed 0, uniform sampling, and 10 bins. Those points are a fresh Day 2 collocation batch with `n_ic` and `n_bc` set to 0. They are not the training batch unless the seed and the interior count happen to match.

`evaluate_model` runs the Day 3 residual on that interior set and records the mean and the max of `|residual|`. Where a reference field exists, it also records an RMS L2 error and a relative L2 error on the same points. Relative L2 is `l2 / rms(reference)`. It is omitted when the reference is identically zero, because that denominator is zero. Burgers still has no field reference, so those two numbers are omitted and the residual stats are the whole field score. `numpy.histogram` bins `|residual|`. The JSON record uses the format tag `pinnforge.eval.v1`. There is no plot dependency.

The harmonic reference is the Day 1 closed form. Poisson now has a manufactured field for each named source. `zero` is `u = 0`. `one` is the particular solution `u = -x² / 2`, which satisfies `-Δu = 1` and does not match homogeneous Dirichlet data. `sin_pi_x` is `u = sin(π x) / π²`, and `sin_pi_x_sin_pi_y` is `u = sin(π x) sin(π y) / (2 π²)`. Those sinusoidal fields are zero on the unit-interval faces, which matches the default Dirichlet specs.

`evaluate_checkpoint` loads a Day 4 CPU checkpoint, rebuilds the built-in spec for the stored equation id, and calls `evaluate_model`. Training stores that id, not a custom spec, so the scored problem is the built-in one. An optional equation name must match the checkpoint. The checkpoint path and `--write-json` path stay inside the working directory. A missing checkpoint is a `ValueError`.

`pinnforge eval --checkpoint checkpoints/checkpoint.pt --equation harmonic` prints the summary. `--write-json eval.json` writes the record, including the histogram. The command needs the `ml` extra and prints the same install hint as `train` when torch is missing. `version`, `equations`, and `sample` still import with no torch.

Tests that load a model or a residual are marked `ml`. The numpy Poisson reference checks are not. Default CI still runs `pytest -m "not ml"`. The `cpu-torch` job runs the full suite, including an exact harmonic field whose residual is near zero, a random network with finite scores, a checkpoint eval after a tiny train, and path checks.
