"""Spectral Burgers reference: derivatives, exact solution, conservation, dataset."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pinnforge.equations import BoundaryCondition, Burgers1DSpec, Interval, ProfileInitialCondition
from pinnforge.reference.burgers import reference_solution as burgers_reference
from pinnforge.reference.numerical.checks import (
    cole_hopf,
    cole_hopf_initial,
    energy,
    finite_difference_solve,
    relative_l2,
    restrict_fourier,
    spatial_mean,
    spectral_error_against_cole_hopf,
)
from pinnforge.reference.numerical.dataset import (
    PilotConfig,
    RunningStats,
    _field_sha256,
    assign_splits,
    estimate_storage,
    generate_pilot,
    sha256_json,
)
from pinnforge.reference.numerical.initial import (
    NU_MAX,
    NU_MIN,
    draw_initial_condition,
    trigonometric_field,
)
from pinnforge.reference.numerical.solver import (
    _phi_series_coefficients,
    dealiased_advection_hat,
    etdrk4_coefficients,
    grid,
    recommended_dt,
    solve,
    spectral_derivative,
    wavenumbers,
)


def _burgers_spec() -> Burgers1DSpec:
    return Burgers1DSpec(
        nu=0.01,
        x=Interval(lower=-1.0, upper=1.0),
        t=Interval(lower=0.0, upper=1.0),
        initial_condition=ProfileInitialCondition(profile="negative_sin_pi_x"),
        boundary_conditions=[BoundaryCondition(variable="x", kind="periodic")],
    )


def test_wavenumber_and_derivative_use_two_pi_over_length() -> None:
    k = wavenumbers(16)
    assert k[1] == pytest.approx(math.pi)
    assert k[16 // 2] == 0.0
    x = grid(64)
    field = np.sin(math.pi * x)
    derivative = spectral_derivative(field)
    assert np.allclose(derivative, math.pi * np.cos(math.pi * x), atol=1e-12)


def test_dealiased_product_matches_a_resolved_identity() -> None:
    n = 32
    x = grid(n)
    field = np.cos(math.pi * x)
    product = np.fft.ifft(dealiased_advection_hat(np.fft.fft(field), wavenumbers(n))).real
    expected = -(math.pi / 2.0) * np.sin(2.0 * math.pi * x)
    assert np.allclose(product, expected, atol=1e-10)


def test_dealiasing_drops_the_aliased_double_angle() -> None:
    n = 16
    x = grid(n)
    mode = 6
    field = np.cos(mode * math.pi * x)
    dealiased = dealiased_advection_hat(np.fft.fft(field), wavenumbers(n))
    assert np.max(np.abs(dealiased)) < 1e-8
    naive = np.fft.fft(field * spectral_derivative(field))
    assert np.max(np.abs(naive)) > 1.0


def test_etdrk4_coefficients_match_rk4_at_zero_and_the_bracket_at_z() -> None:
    q, g1, g2, g3 = _phi_series_coefficients()
    assert q[0] == pytest.approx(0.5)
    assert g1[0] == pytest.approx(1.0 / 6.0)
    assert g2[0] == pytest.approx(1.0 / 6.0)
    assert g3[0] == pytest.approx(1.0 / 6.0)
    dt = 0.2
    linear = np.array([-10.0])
    _, _, q_coeff, f1, f2, f3 = etdrk4_coefficients(linear, dt)
    z = dt * linear[0]
    exponential = math.exp(z)
    bracket_1 = (-4.0 - z + exponential * (4.0 - 3.0 * z + z * z)) / z**3
    bracket_2 = (2.0 + z + exponential * (-2.0 + z)) / z**3
    bracket_3 = (-4.0 - 3.0 * z - z * z + exponential * (4.0 - z)) / z**3
    assert f1[0] == pytest.approx(dt * bracket_1, rel=1e-12)
    assert f2[0] == pytest.approx(dt * bracket_2, rel=1e-12)
    assert f3[0] == pytest.approx(dt * bracket_3, rel=1e-12)
    assert q_coeff[0] == pytest.approx(dt * math.expm1(z / 2.0) / z, rel=1e-12)


def test_zero_nonlinearity_is_the_exact_heat_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    def _zero(u_hat: np.ndarray, k: np.ndarray) -> np.ndarray:
        del k
        return np.zeros_like(u_hat)

    monkeypatch.setattr(
        "pinnforge.reference.numerical.solver.dealiased_advection_hat",
        _zero,
    )
    n = 32
    nu = 0.07
    x = grid(n)
    initial = np.sin(math.pi * x)
    trajectory = solve(initial, nu, dt=1e-2, t_final=0.2, save_dt=0.1)
    exact = np.exp(-nu * math.pi**2 * trajectory.t[-1]) * initial
    assert relative_l2(trajectory.u[-1], exact) < 1e-12


def test_steep_cole_hopf_error_drops_when_the_grid_is_refined() -> None:
    coarse = spectral_error_against_cole_hopf(
        n=32,
        nu=0.05,
        dt=1e-4,
        t_final=0.05,
        amplitude=0.99,
        save_dt=0.05,
    )[1]
    fine = spectral_error_against_cole_hopf(
        n=64,
        nu=0.05,
        dt=1e-4,
        t_final=0.05,
        amplitude=0.99,
        save_dt=0.05,
    )[1]
    assert fine < coarse / 20.0


def test_cole_hopf_spectral_error_is_small_and_fourth_order_in_time() -> None:
    _, final, _ = spectral_error_against_cole_hopf(
        n=64,
        nu=0.05,
        dt=1e-3,
        t_final=0.2,
        save_dt=0.05,
    )
    assert final < 1e-8
    coarse = spectral_error_against_cole_hopf(n=128, nu=0.08, dt=4e-3, t_final=0.2, save_dt=0.04)[1]
    fine = spectral_error_against_cole_hopf(n=128, nu=0.08, dt=2e-3, t_final=0.2, save_dt=0.04)[1]
    assert fine < coarse / 8.0


def test_finite_difference_converges_toward_cole_hopf_and_trails_the_spectral_solver() -> None:
    nu = 0.1
    amplitude = 0.4
    t_final = 0.05
    dt = 2.5e-4
    errors = []
    for n in (32, 64):
        initial = cole_hopf_initial(n, nu=nu, amplitude=amplitude)
        trajectory = finite_difference_solve(initial, nu, dt=dt, t_final=t_final, save_dt=t_final)
        exact = cole_hopf(trajectory.x, trajectory.t[-1], nu=nu, amplitude=amplitude)
        errors.append(relative_l2(trajectory.u[-1], exact))
    assert errors[1] < errors[0] / 2.5
    spectral = spectral_error_against_cole_hopf(
        n=64,
        nu=nu,
        dt=dt,
        t_final=t_final,
        amplitude=amplitude,
        save_dt=t_final,
    )[1]
    assert spectral < errors[1] / 10.0


def test_unforced_run_conserves_mean_and_does_not_increase_energy() -> None:
    initial = draw_initial_condition(7, 1, 64)
    trajectory = solve(initial.u0, 0.05, dt=1e-3, t_final=0.1, save_dt=0.02)
    means = spatial_mean(trajectory.u)
    energies = energy(trajectory.u)
    assert np.max(np.abs(means - means[0])) < 1e-12
    assert float(np.max(energies - energies[0])) < 1e-10
    assert energies[-1] < energies[0]


def test_solver_is_deterministic_for_the_same_initial_field() -> None:
    initial = draw_initial_condition(3, 4, 32)
    first = solve(initial.u0, initial.nu, dt=2e-3, t_final=0.02, save_dt=0.02)
    second = solve(initial.u0, initial.nu, dt=2e-3, t_final=0.02, save_dt=0.02)
    assert np.array_equal(first.u, second.u)


def test_initial_condition_family_is_normalized_bounded_and_seeded() -> None:
    first = draw_initial_condition(11, 2, 64)
    again = draw_initial_condition(11, 2, 128)
    other = draw_initial_condition(11, 3, 64)
    assert first.a.shape == (4,)
    assert np.allclose(first.a, again.a)
    assert np.allclose(first.b, again.b)
    assert first.scale == again.scale
    assert first.nu == pytest.approx(again.nu)
    assert not np.allclose(first.u0, other.u0)
    assert NU_MIN <= first.nu <= NU_MAX
    canonical = first.scale * trigonometric_field(first.a, first.b, 4096)
    assert np.max(np.abs(canonical)) == pytest.approx(1.0)
    assert abs(float(np.mean(first.u0))) < 1e-12
    for m, (a_m, b_m) in enumerate(zip(first.a, first.b, strict=True), start=1):
        assert abs(a_m) <= 1.0 / m + 1e-15
        assert abs(b_m) <= 1.0 / m + 1e-15
    guide = recommended_dt(256, 1.0)
    assert guide == pytest.approx(0.5 / (math.pi * 127))


def test_splits_are_a_partition_and_normalization_uses_training_values_only() -> None:
    splits = assign_splits(4, 2, 2, split_seed=5)
    flat = splits["train"] + splits["val"] + splits["test"]
    assert sorted(flat) == list(range(8))
    assert len(set(flat)) == 8
    again = assign_splits(4, 2, 2, split_seed=5)
    assert splits == again
    stats = RunningStats()
    stats.update(np.array([0.0, 2.0]))
    stats.update(np.array([4.0]))
    # Population variance of {0, 2, 4} is 8/3, std is sqrt(8/3).
    assert stats.mean == pytest.approx(2.0)
    assert stats.population_std() == pytest.approx(math.sqrt(8.0 / 3.0))


def test_tiny_pilot_writes_hashes_splits_and_train_only_normalization(tmp_path) -> None:
    output = tmp_path / "pilot"
    manifest = generate_pilot(
        output,
        PilotConfig(
            n_train=2,
            n_val=1,
            n_test=1,
            n=16,
            dt=0.01,
            t_final=0.04,
            save_dt=0.02,
            master_seed=9,
            split_seed=4,
            batch_size=2,
        ),
    )
    assert manifest["format"] == "pinnforge.burgers_pilot.v1"
    assert manifest["normalization"]["applied_to_files"] is False
    assert manifest["normalization"]["fit_on"] == "train"
    assert manifest["pilot"]["windowing"] == "none"
    assert sha256_json(manifest["solver"]) == manifest["solver_config_sha256"]
    splits = manifest["splits"]
    assert sorted(splits["train"] + splits["val"] + splits["test"]) == [0, 1, 2, 3]
    train_values = []
    for record in manifest["instances"]:
        archive = np.load(output / record["path"])
        assert archive["u"].dtype == np.float64
        assert archive["u"].shape[1] == 16
        assert record["field_sha256"] == _field_sha256(archive["u"])
        assert int(archive["instance_id"]) == record["instance_id"]
        assert float(archive["nu"]) == pytest.approx(record["nu"])
        if record["split"] == "train":
            train_values.append(archive["u"])
    stacked = np.concatenate([values.ravel() for values in train_values])
    assert manifest["normalization"]["u_mean"] == pytest.approx(float(np.mean(stacked)))
    assert manifest["normalization"]["u_std"] == pytest.approx(float(np.std(stacked)))
    assert manifest["normalization"]["n_values"] == stacked.size
    storage = estimate_storage(PilotConfig(n_train=2, n_val=1, n_test=1, n=16, save_dt=0.02, t_final=0.04))
    assert storage["n_times"] == 3
    assert storage["field_bytes"] == 4 * 3 * 16 * 8


def test_coordinate_pinn_burgers_hook_still_has_no_field() -> None:
    with pytest.raises(NotImplementedError, match="burgers_1d"):
        burgers_reference(_burgers_spec(), np.array([0.0]), np.array([0.0]))


def test_saved_study_and_pilot_manifest_meet_the_label_tolerance() -> None:
    root = Path(__file__).resolve().parents[1]
    document = json.loads((root / "docs/stage2/convergence.json").read_text(encoding="utf-8"))
    direct = document["pilot_versus_reference"]
    assert direct["n"] == 256
    assert direct["dt"] == 0.001
    assert direct["rel_l2_final"] < 1e-8
    assert direct["rel_l2_spacetime"] < 1e-8
    invariants = document["unforced"]["invariants_pilot_settings"]
    assert invariants["max_abs_mean_drift"] < 1e-12
    assert invariants["max_energy_increase"] <= 1e-10
    manifest = json.loads((root / "docs/stage2/pilot_manifest.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "pinnforge.burgers_pilot.v1"
    assert [len(manifest["splits"][name]) for name in ("train", "val", "test")] == [512, 128, 128]
    assert manifest["normalization"]["fit_on"] == "train"
    assert manifest["normalization"]["applied_to_files"] is False
    assert manifest["pilot"]["windowing"] == "none"
    covered = manifest["splits"]["train"] + manifest["splits"]["val"] + manifest["splits"]["test"]
    assert sorted(covered) == list(range(768))


def test_numerical_cli_help_lists_the_study_commands() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "pinnforge.reference.numerical", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "convergence" in completed.stdout
    assert "pilot" in completed.stdout


def test_restrict_fourier_reproduces_a_bandlimited_field() -> None:
    fine = grid(32)
    coarse = grid(16)
    values = np.sin(math.pi * fine) + 0.3 * np.cos(2.0 * math.pi * fine)
    expected = np.sin(math.pi * coarse) + 0.3 * np.cos(2.0 * math.pi * coarse)
    assert np.allclose(restrict_fourier(values, 16), expected, atol=1e-12)
