# Day 3

A fully connected PINN and the residual operators for the three Day 1 equations. `MLP` takes an input width, hidden widths, and an activation (`tanh` by default, because the residual needs a smooth second derivative). `mlp_from_spec` sets the input width from the collocation domain: `t` for the oscillator, `(x, t)` for Burgers, and `x` or `(x, y)` for Poisson. The output is the scalar field `u`.

`residual(model, coords, spec)` returns a tensor of shape `(n, 1)`. Derivatives go through `torch.autograd.grad` with `create_graph=True`. The oscillator residual is ü + ω²u, with ω from the spec (`omega`, or `sqrt(k / m)`). Burgers is u_t + u u_x − ν u_xx on columns `(x, t)`. Poisson follows the Day 1 statement −Δu = f, so the residual is −Δu − f for the named source (`zero`, `one`, `sin_pi_x`, `sin_pi_x_sin_pi_y`). `residual_from_field` is the same operator on a tensor that already depends on the coordinates, which is how a closed form is checked without a network.

`soft_penalty` builds two mean-squared terms from a `CollocationBatch` and the prescribed spec values. The initial term matches harmonic `(u, du_dt)` or a Burgers profile on the `ic` rows. The Dirichlet term matches `value` on Dirichlet `bc` rows (`du_dt` boundaries use the time derivative). Neumann and periodic faces are skipped. Poisson has no initial condition, so that term is zero. Day 4 wires these terms and the residual into the training loop. This day does not train.

Importing `pinnforge.models`, `pinnforge.residuals`, or `pinnforge.losses` without torch raises `InstallHint`. The package root, the schemas, and the samplers still import with no torch. `pinnforge residual --equation harmonic --seed 0` seeds a width-(8, 8) network, samples 16 interior points, and prints the residual MSE. It needs the `ml` extra.

Default CI runs `pytest -m "not ml"` without torch. A separate job installs the CPU wheel and runs the full suite, including the residual tests.
