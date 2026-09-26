"""Spatial and temporal convergence for periodic viscous Burgers.

The study is a fixed protocol, not an automatic accuracy certificate.
It records:

- Cole–Hopf errors at N = 64, 128, 256, 512, 1024 and at halved steps
- an unforced low-frequency initial condition compared with a finer run
- discrete mean drift and energy change on that unforced problem
- a viscosity probe below the pilot interval

Numbers are written as JSON. Plots are drawn from those numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.reference.numerical.checks import (
    energy,
    relative_l2,
    spatial_mean,
    spectral_error_against_cole_hopf,
)
from pinnforge.reference.numerical.initial import draw_initial_condition
from pinnforge.reference.numerical.plots import write_chart
from pinnforge.reference.numerical.solver import recommended_dt, solve

FORMAT = "pinnforge.burgers_convergence.v1"
COLE_NU = 0.05
COLE_AMPLITUDE = 0.5
COLE_DT = 2.5e-4
# a = 0.99 makes φ nearly touch zero, so the initial field is steep and the
# Fourier tail is visible at modest N. a = 0.5 is already at roundoff by N = 64.
STEEP_AMPLITUDE = 0.99
STEEP_DT = 1e-4
STEEP_T_FINAL = 0.2
UNFORCED_MASTER_SEED = 0
UNFORCED_INSTANCE = 0
UNFORCED_NU = 0.05
REFERENCE_N = 1024
REFERENCE_DT = 5e-4
PILOT_N = 256
PILOT_DT = 1e-3
T_FINAL = 1.0
# 0.04 is an integer multiple of every dt used below, including 2.5e-4.
SAVE_DT = 0.04


def run_convergence(output: Path) -> dict[str, Any]:
    """Run the fixed study and write JSON plus SVG charts under ``output``."""

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    print("Cole-Hopf spatial convergence", flush=True)
    cole_spatial = [
        _cole_row(n=n, dt=COLE_DT)
        for n in (64, 128, 256, 512, 1024)
    ]
    print("Cole-Hopf steep spatial convergence", flush=True)
    cole_steep = [
        _cole_row(n=n, dt=STEEP_DT, amplitude=STEEP_AMPLITUDE, t_final=STEEP_T_FINAL, save_dt=STEEP_T_FINAL)
        for n in (32, 64, 128, 256, 512)
    ]
    print("Cole-Hopf temporal convergence", flush=True)
    cole_temporal = [
        _cole_row(n=PILOT_N, dt=dt)
        for dt in (4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4)
    ]
    print("Unforced reference and comparisons", flush=True)
    unforced = _unforced()
    print("Pilot settings versus the fine reference", flush=True)
    pilot_versus_reference = _pilot_versus_reference()
    print("Viscosity probe", flush=True)
    probe = _viscosity_probe()
    document = {
        "format": FORMAT,
        "definitions": {
            "relative_l2": "||u-v||_2 / ||v||_2 on the shared flattened samples",
            "spatial_mean": "(1/N) sum_j u_j = (1/L) integral u dx for resolved modes",
            "energy": "(dx/2) sum_j u_j^2, rectangle rule for integral u^2/2 dx",
            "domain": "[-1, 1] periodic, t in [0, 1]",
            "cole_hopf": "a=0.5, wave=1, nu=0.05, t_final=1, dt=2.5e-4 for the smooth spatial scan",
            "cole_hopf_steep": "a=0.99, wave=1, nu=0.05, t_final=0.2, dt=1e-4",
            "unforced_ic": "master_seed=0, instance_id=0, modes 1..4, max-norm 1, nu fixed at 0.05",
        },
        "cole_hopf": {
            "spatial": cole_spatial,
            "temporal": cole_temporal,
            "steep_spatial": cole_steep,
        },
        "unforced": unforced,
        "pilot_versus_reference": pilot_versus_reference,
        "viscosity_probe": probe,
        "pilot_step_guide": {
            "n": PILOT_N,
            "dt": PILOT_DT,
            "recommended_dt_cfl_0.5": recommended_dt(PILOT_N, 1.0, cfl=0.5),
        },
    }
    (destination / "convergence.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_plots(destination, document)
    print(f"wrote {destination / 'convergence.json'}", flush=True)
    return document


def _cole_row(
    *,
    n: int,
    dt: float,
    amplitude: float = COLE_AMPLITUDE,
    t_final: float = T_FINAL,
    save_dt: float = SAVE_DT,
) -> dict[str, float | int]:
    print(f"  cole-hopf n={n} dt={dt} a={amplitude} t_final={t_final}", flush=True)
    _, final, spacetime = spectral_error_against_cole_hopf(
        n=n,
        nu=COLE_NU,
        dt=dt,
        t_final=t_final,
        amplitude=amplitude,
        save_dt=save_dt,
    )
    return {
        "n": n,
        "dt": dt,
        "amplitude": amplitude,
        "t_final": t_final,
        "rel_l2_final": final,
        "rel_l2_spacetime": spacetime,
    }


def _unforced() -> dict[str, Any]:
    reference = _unforced_trajectory(REFERENCE_N, REFERENCE_DT)
    spatial = []
    for n in (128, 256, 512, 1024):
        print(f"  unforced spatial n={n}", flush=True)
        trajectory = reference if n == REFERENCE_N else _unforced_trajectory(n, REFERENCE_DT)
        compared = _restrict_stack(reference.u, n)
        spatial.append(
            {
                "n": n,
                "dt": REFERENCE_DT,
                "rel_l2_final": relative_l2(trajectory.u[-1], compared[-1]),
                "rel_l2_spacetime": relative_l2(trajectory.u, compared),
                "reference_n": REFERENCE_N,
            }
        )
    temporal = []
    finest = _unforced_trajectory(512, 2.5e-4)
    for dt in (4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4):
        print(f"  unforced temporal dt={dt}", flush=True)
        trajectory = finest if dt == 2.5e-4 else _unforced_trajectory(512, dt)
        temporal.append(
            {
                "n": 512,
                "dt": dt,
                "rel_l2_final": relative_l2(trajectory.u[-1], finest.u[-1]),
                "reference_dt": 2.5e-4,
            }
        )
    pilot = _unforced_trajectory(PILOT_N, PILOT_DT)
    fine_invariants = _invariant_record(reference)
    pilot_invariants = _invariant_record(pilot)
    return {
        "nu": UNFORCED_NU,
        "master_seed": UNFORCED_MASTER_SEED,
        "instance_id": UNFORCED_INSTANCE,
        "spatial_against_reference": spatial,
        "temporal_against_finest": temporal,
        "invariants_reference": fine_invariants,
        "invariants_pilot_settings": pilot_invariants,
    }


def _unforced_trajectory(n: int, dt: float):
    initial = draw_initial_condition(UNFORCED_MASTER_SEED, UNFORCED_INSTANCE, n)
    # The family also draws a viscosity. The study fixes ν so the comparison
    # is not mixed with that draw. Coefficients and scale stay as drawn.
    return solve(initial.u0, UNFORCED_NU, dt=dt, t_final=T_FINAL, save_dt=SAVE_DT)


def _invariant_record(trajectory) -> dict[str, Any]:
    means = spatial_mean(trajectory.u)
    energies = energy(trajectory.u)
    drift = np.max(np.abs(means - means[0]))
    increase = float(np.max(energies - energies[0]))
    return {
        "n": int(trajectory.u.shape[1]),
        "dt": trajectory.dt,
        "max_abs_mean_drift": float(drift),
        "mean_initial": float(means[0]),
        "energy_initial": float(energies[0]),
        "energy_final": float(energies[-1]),
        "max_energy_increase": increase,
        "max_abs_initial": float(np.max(np.abs(trajectory.u[0]))),
        "max_abs_final": float(np.max(np.abs(trajectory.u[-1]))),
        "max_abs_over_time": float(np.max(np.abs(trajectory.u))),
        "energy_nonincreasing_at_1e-10": bool(increase <= 1e-10),
        "t": [float(value) for value in trajectory.t],
        "mean": [float(value) for value in means],
        "energy": [float(value) for value in energies],
    }


def _pilot_versus_reference() -> dict[str, float | int]:
    """N=256, dt=1e-3 against N=1024, dt=2.5e-4 on the pilot save grid."""

    coarse_ic = draw_initial_condition(UNFORCED_MASTER_SEED, UNFORCED_INSTANCE, PILOT_N)
    fine_ic = draw_initial_condition(UNFORCED_MASTER_SEED, UNFORCED_INSTANCE, REFERENCE_N)
    coarse = solve(coarse_ic.u0, UNFORCED_NU, dt=PILOT_DT, t_final=T_FINAL, save_dt=0.01)
    fine = solve(fine_ic.u0, UNFORCED_NU, dt=2.5e-4, t_final=T_FINAL, save_dt=0.01)
    restricted = _restrict_stack(fine.u, PILOT_N)
    return {
        "n": PILOT_N,
        "dt": PILOT_DT,
        "reference_n": REFERENCE_N,
        "reference_dt": 2.5e-4,
        "save_dt": 0.01,
        "nu": UNFORCED_NU,
        "rel_l2_final": relative_l2(coarse.u[-1], restricted[-1]),
        "rel_l2_spacetime": relative_l2(coarse.u, restricted),
    }


def _viscosity_probe() -> list[dict[str, float | bool]]:
    rows = []
    for nu in (0.10, 0.05, 0.02, 0.01, 0.005):
        print(f"  probe nu={nu}", flush=True)
        coarse = _forced_nu(256, nu)
        fine = _forced_nu(512, nu)
        error = relative_l2(coarse.u[-1], _restrict_stack(fine.u, 256)[-1])
        rows.append(
            {
                "nu": nu,
                "n_coarse": 256,
                "n_fine": 512,
                "dt": REFERENCE_DT,
                "rel_l2_final": error,
                "inside_pilot_interval": bool(0.02 <= nu <= 0.10),
            }
        )
    return rows


def _forced_nu(n: int, nu: float):
    initial = draw_initial_condition(UNFORCED_MASTER_SEED, UNFORCED_INSTANCE, n)
    return solve(initial.u0, nu, dt=REFERENCE_DT, t_final=T_FINAL, save_dt=SAVE_DT)


def _restrict_stack(u: np.ndarray, n_out: int) -> np.ndarray:
    from pinnforge.reference.numerical.checks import restrict_fourier

    return restrict_fourier(u, n_out)


def _write_plots(destination: Path, document: dict[str, Any]) -> None:
    spatial = document["cole_hopf"]["spatial"]
    steep = document["cole_hopf"]["steep_spatial"]
    temporal = document["cole_hopf"]["temporal"]
    unforced_spatial = document["unforced"]["spatial_against_reference"]
    write_chart(
        destination / "spatial_convergence.svg",
        [
            {
                "name": "Cole-Hopf a=0.99",
                "x": [row["n"] for row in steep],
                "y": [_positive(row["rel_l2_final"]) for row in steep],
            },
            {
                "name": "Cole-Hopf a=0.5",
                "x": [row["n"] for row in spatial],
                "y": [_positive(row["rel_l2_final"]) for row in spatial],
            },
            {
                "name": "Unforced vs N=1024",
                "x": [row["n"] for row in unforced_spatial if row["n"] != REFERENCE_N],
                "y": [
                    _positive(row["rel_l2_final"])
                    for row in unforced_spatial
                    if row["n"] != REFERENCE_N
                ],
            },
        ],
        xlabel="N",
        ylabel="relative L2 at t_final",
        title="Spatial convergence, periodic Burgers",
        xlog=True,
        ylog=True,
    )
    write_chart(
        destination / "temporal_convergence.svg",
        [
            {
                "name": "Cole-Hopf, N=256",
                "x": [row["dt"] for row in temporal],
                "y": [_positive(row["rel_l2_final"]) for row in temporal],
            },
            {
                "name": "Unforced, N=512 vs dt=2.5e-4",
                "x": [
                    row["dt"]
                    for row in document["unforced"]["temporal_against_finest"]
                    if row["dt"] != 2.5e-4
                ],
                "y": [
                    _positive(row["rel_l2_final"])
                    for row in document["unforced"]["temporal_against_finest"]
                    if row["dt"] != 2.5e-4
                ],
            },
        ],
        xlabel="dt",
        ylabel="relative L2 at t_final",
        title="Temporal convergence, ETDRK4",
        xlog=True,
        ylog=True,
    )
    invariants = document["unforced"]["invariants_pilot_settings"]
    write_chart(
        destination / "invariants.svg",
        [
            {
                "name": "energy",
                "x": invariants["t"],
                "y": invariants["energy"],
            },
            {
                "name": "mean + 0.5",
                "x": invariants["t"],
                "y": [value + 0.5 for value in invariants["mean"]],
            },
        ],
        xlabel="t",
        ylabel="energy, and mean shifted by 0.5",
        title="Unforced invariants at N=256, dt=1e-3",
        xlog=False,
        ylog=False,
    )


def _positive(value: float) -> float:
    """Keep a positive display value when a comparison is at roundoff or identical."""

    return max(float(value), 1e-16)
