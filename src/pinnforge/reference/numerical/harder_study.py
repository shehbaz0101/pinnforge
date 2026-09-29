"""Convergence study for the harder Burgers pilot.

The protocol is fixed. It records initial-condition sharpness against the
Stage 2 family, spatial and temporal errors on the steepest draw, the
label grid against a finer run, and a viscosity probe down to ``ν = 0.005``.
It does not train an operator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.reference.numerical.checks import energy, relative_l2, restrict_fourier, spatial_mean
from pinnforge.reference.numerical.harder import (
    HARD_DT,
    HARD_MASTER_SEED,
    HARD_N,
    HARD_SAVE_DT,
    HARD_T_FINAL,
    LABEL_ENERGY_INCREASE_MAX,
    LABEL_MEAN_DRIFT_MAX,
    LABEL_REL_L2_MAX,
    M_KEEP,
    NU_MAX,
    NU_MIN,
    REFERENCE_DT,
    REFERENCE_N,
    draw_hard_initial_condition,
    high_mode_energy_fraction,
    label_gate_document,
    spectral_slope_max,
)
from pinnforge.reference.numerical.initial import draw_initial_condition
from pinnforge.reference.numerical.plots import write_chart
from pinnforge.reference.numerical.solver import Trajectory, recommended_dt, solve

FORMAT = "pinnforge.burgers_hard_convergence.v1"
PROBE_POOL = 64
N_STEEPEST = 3
# 5e-4 is outside the N=1024 advection guide and is not a label step.
# It is kept so the temporal table has two halvings.
TEMPORAL_STEPS = (5e-4, 2.5e-4, 1.25e-4)
SPATIAL_NS = (256, 512, 1024, 2048)
COARSE_SETTINGS = ((256, 1e-3), (512, 5e-4))
VISCOSITIES = (0.10, 0.05, 0.02, 0.01, 0.005)


def run_hard_convergence(output: Path) -> dict[str, Any]:
    """Run the fixed harder-pilot study and write JSON plus SVG charts."""

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    print("Initial-condition sharpness", flush=True)
    sharpness = _sharpness()
    steepest = [row["instance_id"] for row in sharpness["steepest"]]
    steep_id = steepest[0]
    label_instances = list(dict.fromkeys([0, *steepest]))
    cache: dict[tuple[object, ...], Trajectory] = {}
    print("Spatial convergence on the steepest initial condition", flush=True)
    spatial = [
        _spatial_row(cache, instance_id=steep_id, n=n) for n in SPATIAL_NS
    ]
    print("Temporal convergence on the steepest initial condition", flush=True)
    temporal = [
        _temporal_row(cache, instance_id=steep_id, dt=dt) for dt in TEMPORAL_STEPS
    ]
    print("Label grid versus the fine reference", flush=True)
    pilot_rows = [
        _label_row(cache, instance_id=instance_id) for instance_id in label_instances
    ]
    print("Coarser grids versus the fine reference", flush=True)
    coarse_rows = [
        _coarse_row(cache, instance_id=steep_id, n=n, dt=dt) for n, dt in COARSE_SETTINGS
    ]
    print("Viscosity probe", flush=True)
    probe = [_viscosity_row(cache, instance_id=steep_id, nu=nu) for nu in VISCOSITIES]
    pilot_trajectory = _solve(cache, steep_id, HARD_N, HARD_DT, NU_MIN, HARD_SAVE_DT)
    reference_trajectory = _solve(cache, steep_id, REFERENCE_N, REFERENCE_DT, NU_MIN, HARD_SAVE_DT)
    document = {
        "format": FORMAT,
        "ic_family": "tanh_bandlimited",
        "definitions": {
            "relative_l2": "||u-v||_2 / ||v||_2 on the shared flattened samples",
            "spatial_comparison": (
                "fine field restricted by truncating its FFT onto the coarse modes"
            ),
            "domain": "[-1, 1] periodic, t in [0, 1]",
            "ic_family": (
                "tanh(3 * p_hat) projected onto modes |m| <= 48, mean removed, "
                "max-abs normalized on 8192 nodes"
            ),
            "sharpness_pool": (
                f"instance ids 0..{PROBE_POOL - 1}, master_seed {HARD_MASTER_SEED}, "
                "slope on N=256"
            ),
            "steepest_rule": f"{N_STEEPEST} largest initial slopes in that pool, then instance 0",
            "label_resolution": f"N={HARD_N}, dt={HARD_DT}, save_dt={HARD_SAVE_DT}",
            "reference_resolution": f"N={REFERENCE_N}, dt={REFERENCE_DT}, save_dt={HARD_SAVE_DT}",
            "forced_viscosity": (
                "convergence viscosities are fixed. They are not the random draw "
                "stored with that instance id"
            ),
        },
        "initial_condition_sharpness": sharpness,
        "spatial_against_reference": spatial,
        "temporal_against_finest": temporal,
        "pilot_versus_reference": pilot_rows,
        "coarse_versus_reference": coarse_rows,
        "viscosity_probe": probe,
        "invariants_pilot_settings": _invariant_record(pilot_trajectory),
        "invariants_reference": _invariant_record(reference_trajectory),
        "label_gate": label_gate_document(),
        "pilot_step_guide": {
            "n": HARD_N,
            "dt": HARD_DT,
            "recommended_dt_cfl_0.5": recommended_dt(HARD_N, 1.0, cfl=0.5),
            "dt_inside_guide": bool(HARD_DT <= recommended_dt(HARD_N, 1.0, cfl=0.5)),
        },
    }
    (destination / "convergence.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_plots(destination, document)
    print(f"wrote {destination / 'convergence.json'}", flush=True)
    return document


def _sharpness() -> dict[str, Any]:
    hard_rows = []
    stage_slopes = []
    stage_fractions = []
    for instance_id in range(PROBE_POOL):
        hard = draw_hard_initial_condition(HARD_MASTER_SEED, instance_id, 256)
        stage = draw_initial_condition(HARD_MASTER_SEED, instance_id, 256)
        hard_rows.append(
            {
                "instance_id": instance_id,
                "slope_max": spectral_slope_max(hard.u0),
                "high_mode_energy_fraction": high_mode_energy_fraction(hard.u0, cutoff=4),
                "retained_tail_fraction": high_mode_energy_fraction(hard.u0, cutoff=M_KEEP),
            }
        )
        stage_slopes.append(spectral_slope_max(stage.u0))
        stage_fractions.append(high_mode_energy_fraction(stage.u0, cutoff=4))
    steepest = sorted(hard_rows, key=lambda row: (-row["slope_max"], row["instance_id"]))[:N_STEEPEST]
    hard_slopes = [row["slope_max"] for row in hard_rows]
    hard_fractions = [row["high_mode_energy_fraction"] for row in hard_rows]
    return {
        "master_seed": HARD_MASTER_SEED,
        "n": 256,
        "pool": PROBE_POOL,
        "hard": _sample_summary(hard_slopes, hard_fractions),
        "stage2_same_ids": _sample_summary(stage_slopes, stage_fractions),
        "steepest": [
            {"instance_id": row["instance_id"], "slope_max": row["slope_max"]} for row in steepest
        ],
    }


def _sample_summary(slopes: list[float], fractions: list[float]) -> dict[str, float]:
    slope = np.asarray(slopes, dtype=np.float64)
    fraction = np.asarray(fractions, dtype=np.float64)
    return {
        "slope_min": float(np.min(slope)),
        "slope_median": float(np.median(slope)),
        "slope_max": float(np.max(slope)),
        "high_mode_energy_fraction_min": float(np.min(fraction)),
        "high_mode_energy_fraction_median": float(np.median(fraction)),
        "high_mode_energy_fraction_max": float(np.max(fraction)),
    }


def _spatial_row(cache: dict[tuple[object, ...], Trajectory], *, instance_id: int, n: int) -> dict[str, Any]:
    print(f"  spatial n={n} dt={REFERENCE_DT}", flush=True)
    coarse = _solve(cache, instance_id, n, REFERENCE_DT, NU_MIN, HARD_SAVE_DT)
    reference = _solve(cache, instance_id, REFERENCE_N, REFERENCE_DT, NU_MIN, HARD_SAVE_DT)
    compared = _against(coarse, reference)
    return {
        "instance_id": instance_id,
        "nu": NU_MIN,
        "n": n,
        "dt": REFERENCE_DT,
        "reference_n": REFERENCE_N,
        "reference_dt": REFERENCE_DT,
        **compared,
    }


def _temporal_row(cache: dict[tuple[object, ...], Trajectory], *, instance_id: int, dt: float) -> dict[str, Any]:
    print(f"  temporal n={HARD_N} dt={dt}", flush=True)
    coarse = _solve(cache, instance_id, HARD_N, dt, NU_MIN, HARD_SAVE_DT)
    reference = _solve(cache, instance_id, HARD_N, REFERENCE_DT, NU_MIN, HARD_SAVE_DT)
    guide = recommended_dt(HARD_N, 1.0, cfl=0.5)
    compared = _against(coarse, reference)
    return {
        "instance_id": instance_id,
        "nu": NU_MIN,
        "n": HARD_N,
        "dt": dt,
        "reference_n": HARD_N,
        "reference_dt": REFERENCE_DT,
        "inside_step_guide": bool(dt <= guide),
        **compared,
    }


def _label_row(cache: dict[tuple[object, ...], Trajectory], *, instance_id: int) -> dict[str, Any]:
    print(f"  label inst={instance_id} nu={NU_MIN}", flush=True)
    coarse = _solve(cache, instance_id, HARD_N, HARD_DT, NU_MIN, HARD_SAVE_DT)
    reference = _solve(cache, instance_id, REFERENCE_N, REFERENCE_DT, NU_MIN, HARD_SAVE_DT)
    compared = _against(coarse, reference)
    row = {
        "instance_id": instance_id,
        "nu": NU_MIN,
        "n": HARD_N,
        "dt": HARD_DT,
        "reference_n": REFERENCE_N,
        "reference_dt": REFERENCE_DT,
        "save_dt": HARD_SAVE_DT,
        **compared,
    }
    row["passes_label_gate"] = _passes(row)
    return row


def _coarse_row(
    cache: dict[tuple[object, ...], Trajectory],
    *,
    instance_id: int,
    n: int,
    dt: float,
) -> dict[str, Any]:
    print(f"  coarse n={n} dt={dt}", flush=True)
    coarse = _solve(cache, instance_id, n, dt, NU_MIN, HARD_SAVE_DT)
    reference = _solve(cache, instance_id, REFERENCE_N, REFERENCE_DT, NU_MIN, HARD_SAVE_DT)
    compared = _against(coarse, reference)
    row = {
        "instance_id": instance_id,
        "nu": NU_MIN,
        "n": n,
        "dt": dt,
        "reference_n": REFERENCE_N,
        "reference_dt": REFERENCE_DT,
        "save_dt": HARD_SAVE_DT,
        **compared,
    }
    row["passes_label_gate"] = _passes(row)
    return row


def _viscosity_row(
    cache: dict[tuple[object, ...], Trajectory],
    *,
    instance_id: int,
    nu: float,
) -> dict[str, Any]:
    print(f"  probe nu={nu}", flush=True)
    coarse = _solve(cache, instance_id, HARD_N, HARD_DT, nu, HARD_SAVE_DT)
    reference = _solve(cache, instance_id, REFERENCE_N, REFERENCE_DT, nu, HARD_SAVE_DT)
    compared = _against(coarse, reference)
    row = {
        "instance_id": instance_id,
        "nu": nu,
        "n": HARD_N,
        "dt": HARD_DT,
        "reference_n": REFERENCE_N,
        "reference_dt": REFERENCE_DT,
        "inside_hard_interval": bool(NU_MIN <= nu <= NU_MAX),
        "inside_stage2_interval": bool(0.02 <= nu <= 0.10),
        **compared,
    }
    row["passes_label_gate"] = _passes(row)
    return row


def _solve(
    cache: dict[tuple[object, ...], Trajectory],
    instance_id: int,
    n: int,
    dt: float,
    nu: float,
    save_dt: float,
) -> Trajectory:
    key = (instance_id, n, float(dt), float(nu), float(save_dt))
    cached = cache.get(key)
    if cached is not None:
        return cached
    initial = draw_hard_initial_condition(HARD_MASTER_SEED, instance_id, n)
    trajectory = solve(initial.u0, nu, dt=dt, t_final=HARD_T_FINAL, save_dt=save_dt)
    cache[key] = trajectory
    return trajectory


def _against(coarse: Trajectory, reference: Trajectory) -> dict[str, float]:
    restricted = restrict_fourier(reference.u, coarse.u.shape[1])
    return {
        "rel_l2_final": relative_l2(coarse.u[-1], restricted[-1]),
        "rel_l2_spacetime": relative_l2(coarse.u, restricted),
    }


def _passes(row: dict[str, Any]) -> bool:
    return bool(
        row["rel_l2_final"] <= LABEL_REL_L2_MAX and row["rel_l2_spacetime"] <= LABEL_REL_L2_MAX
    )


def _invariant_record(trajectory: Trajectory) -> dict[str, Any]:
    means = spatial_mean(trajectory.u)
    energies = energy(trajectory.u)
    drift = float(np.max(np.abs(means - means[0])))
    increase = float(np.max(energies - energies[0]))
    return {
        "n": int(trajectory.u.shape[1]),
        "dt": trajectory.dt,
        "nu": trajectory.nu,
        "max_abs_mean_drift": drift,
        "mean_initial": float(means[0]),
        "energy_initial": float(energies[0]),
        "energy_final": float(energies[-1]),
        "max_energy_increase": increase,
        "max_abs_initial": float(np.max(np.abs(trajectory.u[0]))),
        "max_abs_final": float(np.max(np.abs(trajectory.u[-1]))),
        "max_abs_over_time": float(np.max(np.abs(trajectory.u))),
        "mean_drift_within_gate": bool(drift <= LABEL_MEAN_DRIFT_MAX),
        "energy_nonincreasing_within_gate": bool(increase <= LABEL_ENERGY_INCREASE_MAX),
        "t": [float(value) for value in trajectory.t],
        "mean": [float(value) for value in means],
        "energy": [float(value) for value in energies],
    }


def _write_plots(destination: Path, document: dict[str, Any]) -> None:
    spatial = [
        row for row in document["spatial_against_reference"] if row["n"] != REFERENCE_N
    ]
    temporal = [
        row for row in document["temporal_against_finest"] if row["dt"] != REFERENCE_DT
    ]
    write_chart(
        destination / "spatial_convergence.svg",
        [
            {
                "name": "Final time vs N=2048",
                "x": [row["n"] for row in spatial],
                "y": [_positive(row["rel_l2_final"]) for row in spatial],
            },
            {
                "name": "Saved frames vs N=2048",
                "x": [row["n"] for row in spatial],
                "y": [_positive(row["rel_l2_spacetime"]) for row in spatial],
            },
        ],
        xlabel="N",
        ylabel="relative L2 at nu=0.005",
        title="Spatial convergence, tanh-bandlimited Burgers",
        xlog=True,
        ylog=True,
    )
    write_chart(
        destination / "temporal_convergence.svg",
        [
            {
                "name": "Final time vs dt=1.25e-4",
                "x": [row["dt"] for row in temporal],
                "y": [_positive(row["rel_l2_final"]) for row in temporal],
            },
            {
                "name": "Saved frames vs dt=1.25e-4",
                "x": [row["dt"] for row in temporal],
                "y": [_positive(row["rel_l2_spacetime"]) for row in temporal],
            },
        ],
        xlabel="dt",
        ylabel="relative L2 at N=1024, nu=0.005",
        title="Temporal convergence, ETDRK4",
        xlog=True,
        ylog=True,
    )
    invariants = document["invariants_pilot_settings"]
    write_chart(
        destination / "invariants.svg",
        [
            {"name": "energy", "x": invariants["t"], "y": invariants["energy"]},
            {
                "name": "mean + 0.5",
                "x": invariants["t"],
                "y": [value + 0.5 for value in invariants["mean"]],
            },
        ],
        xlabel="t",
        ylabel="energy, and mean shifted by 0.5",
        title="Invariants at N=1024, dt=2.5e-4, nu=0.005",
        xlog=False,
        ylog=False,
    )


def _positive(value: float) -> float:
    return max(float(value), 1e-16)
