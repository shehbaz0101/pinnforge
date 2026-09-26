# Day 1

Scaffolded the `pinnforge` 0.1.0 package: src layout, a `pinnforge` console script, the MIT license already on main, ruff, and GitHub Actions for pytest on Python 3.11 and 3.12. Equation schemas cover the harmonic oscillator (ω or k/m, time bounds, IC/BC), viscous Burgers in 1D (viscosity, spatial and temporal bounds, IC/BC), and a 1D or 2D Poisson toy (source name, boundary type). The oscillator has a numpy closed form. Burgers and Poisson references raise `NotImplementedError`. No sampler, network, or trainer yet.
