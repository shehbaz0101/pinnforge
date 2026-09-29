"""Harder Burgers pilot: family, manifest, and the unchanged Stage 2 defaults."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pinnforge.reference.numerical.checks import relative_l2, restrict_fourier
from pinnforge.reference.numerical.dataset import (
    FORMAT as STAGE2_FORMAT,
)
from pinnforge.reference.numerical.dataset import (
    PilotConfig,
    generate_pilot,
    sha256_json,
)
from pinnforge.reference.numerical.harder import (
    FORMAT,
    HARD_DT,
    HARD_MASTER_SEED,
    HARD_N,
    HARD_OOD_QUANTILE,
    LABEL_REL_L2_MAX,
    M_KEEP,
    NU_MAX,
    NU_MIN,
    PROTOCOL_FORMAT,
    HardPilotConfig,
    build_protocol,
    draw_hard_initial_condition,
    estimate_hard_storage,
    generate_hard_pilot,
    hard_ood_threshold,
    high_mode_energy_fraction,
    source_hashes,
    spectral_slope_max,
)
from pinnforge.reference.numerical.initial import NU_MIN as STAGE2_NU_MIN
from pinnforge.reference.numerical.initial import draw_initial_condition
from pinnforge.reference.numerical.solver import SolverConfig, recommended_dt
from pinnforge.reference.numerical.stress import STRESS_FORMAT

ROOT = Path(__file__).resolve().parents[1]
STAGE2_PILOT_HASH = "1926d9a625136afca2af0d4dbc318cd77616ee0268ef80341e55c45daf156088"
STAGE2_SOLVER_HASH = "2a0c1c3f078ca81f15bf5c33874bc1403fa6a1a30bda1f484bbc6df5d0d48928"
STAGE5_SENSORS32_MEAN_ABS = 6.091698430189056e-05
STAGE5_SENSORS32_MEAN_REL = 1.3303134076090104e-03
MANIFEST_KEYS = {
    "format",
    "ic_family",
    "pilot",
    "solver",
    "solver_config_sha256",
    "pilot_config_sha256",
    "code_sha256",
    "numpy",
    "splits",
    "normalization",
    "nu_stats",
    "invariants",
    "label_gate",
    "protocol",
    "instances",
}
INSTANCE_KEYS = {
    "instance_id",
    "split",
    "path",
    "sha256",
    "field_sha256",
    "nu",
    "amplitude_scale",
    "max_abs_mean_drift",
    "max_energy_increase",
    "n",
    "n_times",
}


def _tiny_config() -> HardPilotConfig:
    return HardPilotConfig(
        n_train=2,
        n_val=1,
        n_test=1,
        n=128,
        dt=0.002,
        t_final=0.02,
        save_dt=0.01,
        master_seed=9,
        split_seed=4,
        batch_size=2,
    )


def test_stage2_defaults_and_config_hash_are_unchanged() -> None:
    config = PilotConfig()
    assert config.n_train == 512
    assert config.n_val == 128
    assert config.n_test == 128
    assert config.n == 256
    assert config.dt == 1e-3
    assert config.t_final == 1.0
    assert config.save_dt == 0.01
    assert config.master_seed == 20260926
    assert config.split_seed == 20260926
    assert config.nu_min == STAGE2_NU_MIN == 0.02
    assert config.nu_max == 0.10
    assert sha256_json(config.to_dict()) == STAGE2_PILOT_HASH
    solver = SolverConfig(n=256, dt=1e-3, t_final=1.0, save_dt=0.01)
    assert sha256_json(solver.to_dict()) == STAGE2_SOLVER_HASH


def test_hard_config_covers_lower_viscosity_and_stays_inside_the_step_guide() -> None:
    config = HardPilotConfig()
    assert config.nu_min == NU_MIN == 0.005
    assert config.nu_max == NU_MAX == 0.10
    assert config.n == HARD_N == 1024
    assert config.dt == HARD_DT == 2.5e-4
    assert config.n_modes == 8
    assert config.m_keep == M_KEEP == 48
    assert config.beta == 3.0
    assert config.master_seed == HARD_MASTER_SEED == 20260929
    assert config.master_seed != PilotConfig().master_seed
    assert config.to_dict()["ic_family"] == "tanh_bandlimited"
    assert config.dt <= recommended_dt(config.n, 1.0)
    assert LABEL_REL_L2_MAX == 1e-9
    with pytest.raises(ValueError, match="cover"):
        estimate_hard_storage(HardPilotConfig(nu_min=0.02))


def test_hard_initial_condition_is_seeded_bandlimited_and_steeper() -> None:
    first = draw_hard_initial_condition(11, 2, 256)
    again = draw_hard_initial_condition(11, 2, 512)
    other = draw_hard_initial_condition(11, 3, 256)
    assert first.a.shape == (8,)
    assert np.allclose(first.a, again.a)
    assert np.allclose(first.b, again.b)
    assert first.scale == again.scale
    assert first.nu == pytest.approx(again.nu)
    assert not np.allclose(first.u0, other.u0)
    assert NU_MIN <= first.nu <= NU_MAX
    assert abs(float(np.mean(first.u0))) < 1e-10
    assert float(np.max(np.abs(first.u0))) <= 1.0 + 1e-8
    for mode, (a_m, b_m) in enumerate(zip(first.a, first.b, strict=True), start=1):
        bound = float(mode) ** -0.5
        assert abs(a_m) <= bound + 1e-15
        assert abs(b_m) <= bound + 1e-15
    assert high_mode_energy_fraction(first.u0, cutoff=4) > 1e-3
    assert high_mode_energy_fraction(first.u0, cutoff=M_KEEP) < 1e-10
    assert relative_l2(first.u0, restrict_fourier(again.u0, 256)) < 1e-12
    hard = draw_hard_initial_condition(HARD_MASTER_SEED, 0, 256)
    stage = draw_initial_condition(HARD_MASTER_SEED, 0, 256)
    assert spectral_slope_max(hard.u0) > 15.0
    assert spectral_slope_max(stage.u0) < 10.0
    with pytest.raises(ValueError, match="m_keep"):
        draw_hard_initial_condition(1, 0, 64)


def test_tiny_hard_pilot_is_deterministic_and_matches_the_manifest_schema(tmp_path: Path) -> None:
    config = _tiny_config()
    first_dir = tmp_path / "a"
    second_dir = tmp_path / "b"
    first = generate_hard_pilot(first_dir, config)
    second = generate_hard_pilot(second_dir, config)
    assert first["format"] == FORMAT
    assert MANIFEST_KEYS <= set(first)
    assert first["ic_family"] == "tanh_bandlimited"
    assert first["normalization"]["applied_to_files"] is False
    assert first["normalization"]["fit_on"] == "train"
    assert first["pilot"]["windowing"] == "none"
    assert first["pilot"]["split_unit"] == "problem_instance"
    assert sha256_json(first["solver"]) == first["solver_config_sha256"]
    assert sha256_json(first["pilot"]) == first["pilot_config_sha256"]
    assert first["code_sha256"] == source_hashes()
    assert first["label_gate"]["relative_l2_final_max"] == LABEL_REL_L2_MAX
    splits = first["splits"]
    assert sorted(splits["train"] + splits["val"] + splits["test"]) == [0, 1, 2, 3]
    assert [record["field_sha256"] for record in first["instances"]] == [
        record["field_sha256"] for record in second["instances"]
    ]
    assert [record["nu"] for record in first["instances"]] == [
        record["nu"] for record in second["instances"]
    ]
    train_values = []
    for record in first["instances"]:
        assert INSTANCE_KEYS <= set(record)
        assert len(record["field_sha256"]) == 64
        assert len(record["sha256"]) == 64
        assert NU_MIN <= record["nu"] <= NU_MAX
        archive = np.load(first_dir / record["path"])
        assert archive["u"].dtype == np.float64
        assert archive["u"].shape == (3, 128)
        assert int(archive["instance_id"]) == record["instance_id"]
        assert float(archive["nu"]) == pytest.approx(record["nu"])
        assert int(archive["m_keep"]) == M_KEEP
        if record["split"] == "train":
            train_values.append(archive["u"])
    stacked = np.concatenate([values.ravel() for values in train_values])
    assert first["normalization"]["u_mean"] == pytest.approx(float(np.mean(stacked)))
    assert first["normalization"]["u_std"] == pytest.approx(float(np.std(stacked)))
    assert first["nu_stats"]["all"]["count"] == 4
    assert first["nu_stats"]["train"]["count"] == 2
    assert first["invariants"]["max_abs_mean_drift"] <= 1e-12
    assert first["invariants"]["max_energy_increase"] <= 1e-10
    train_nu = [record["nu"] for record in first["instances"] if record["split"] == "train"]
    protocol = first["protocol"]
    assert protocol["format"] == PROTOCOL_FORMAT
    assert protocol["frozen_before_operator_training"] is True
    assert protocol["retuned_after_inverse_measurement"] is False
    assert protocol["hard_ood"]["name"] == "hard_ood"
    assert protocol["hard_ood"]["quantile"] == HARD_OOD_QUANTILE == 0.25
    assert protocol["hard_ood"]["method"] == "linear"
    assert protocol["hard_ood"]["fit_on"] == "train"
    assert protocol["hard_ood"]["threshold_nu"] == pytest.approx(hard_ood_threshold(train_nu))
    assert protocol["hard_ood"]["threshold_nu"] == pytest.approx(
        float(np.quantile(np.asarray(train_nu, dtype=np.float64), 0.25, method="linear"))
    )
    assert protocol["label_gate"]["pilot_n"] == HARD_N
    assert protocol["label_gate"]["relative_l2_final_max"] == LABEL_REL_L2_MAX
    assert protocol["pilot_config_sha256"] == first["pilot_config_sha256"]
    assert protocol == build_protocol(config, train_nu, first["solver_config_sha256"])
    assert protocol == second["protocol"]
    storage = estimate_hard_storage(config)
    assert storage["n_times"] == 3
    assert storage["field_bytes"] == 4 * 3 * 128 * 8
    with pytest.raises(FileExistsError):
        generate_hard_pilot(first_dir, config)


def test_stage2_tiny_pilot_format_is_unchanged(tmp_path: Path) -> None:
    manifest = generate_pilot(
        tmp_path / "stage2",
        PilotConfig(
            n_train=1,
            n_val=1,
            n_test=1,
            n=16,
            dt=0.01,
            t_final=0.02,
            save_dt=0.02,
            master_seed=3,
            split_seed=1,
            batch_size=1,
        ),
    )
    assert manifest["format"] == STAGE2_FORMAT
    assert manifest["pilot"]["nu_min"] == 0.02
    assert manifest["pilot"]["n_modes"] == 4
    assert "ic_family" not in manifest


def test_numerical_cli_keeps_stage2_defaults_and_adds_the_hard_pilot() -> None:
    stage2 = subprocess.run(
        [sys.executable, "-m", "pinnforge.reference.numerical", "estimate"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "instances=768" in stage2.stdout
    assert "times=101" in stage2.stdout
    assert "n=256" in stage2.stdout
    hard = subprocess.run(
        [sys.executable, "-m", "pinnforge.reference.numerical", "hard-estimate"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "instances=768" in hard.stdout
    assert "times=101" in hard.stdout
    assert "n=1024" in hard.stdout
    assert "ic_family=tanh_bandlimited" in hard.stdout
    assert "nu_min=0.005" in hard.stdout
    assert "nu_max=0.1" in hard.stdout
    help_text = subprocess.run(
        [sys.executable, "-m", "pinnforge.reference.numerical", "--help"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for name in (
        "estimate",
        "convergence",
        "pilot",
        "hard-estimate",
        "hard-convergence",
        "hard-pilot",
        "hard-stress",
    ):
        assert name in help_text


def test_saved_hard_study_and_manifest_meet_the_label_gate() -> None:
    study = json.loads((ROOT / "docs" / "stage_a" / "convergence.json").read_text(encoding="utf-8"))
    assert study["format"] == "pinnforge.burgers_hard_convergence.v1"
    assert study["pilot_step_guide"]["dt_inside_guide"] is True
    sharpness = study["initial_condition_sharpness"]
    assert sharpness["hard"]["slope_median"] > 4.0 * sharpness["stage2_same_ids"]["slope_median"]
    assert sharpness["hard"]["high_mode_energy_fraction_median"] > 1e-2
    assert study["label_gate"]["relative_l2_final_max"] == LABEL_REL_L2_MAX
    for row in study["pilot_versus_reference"]:
        assert row["n"] == 1024
        assert row["dt"] == 2.5e-4
        assert row["nu"] == 0.005
        assert row["rel_l2_final"] <= LABEL_REL_L2_MAX
        assert row["rel_l2_spacetime"] <= LABEL_REL_L2_MAX
        assert row["passes_label_gate"] is True
    for row in study["viscosity_probe"]:
        assert 0.005 <= row["nu"] <= 0.10
        assert row["passes_label_gate"] is True
    for row in study["coarse_versus_reference"]:
        assert row["passes_label_gate"] is False
    invariants = study["invariants_pilot_settings"]
    assert invariants["max_abs_mean_drift"] <= 1e-12
    assert invariants["max_energy_increase"] <= 1e-10

    manifest = json.loads(
        (ROOT / "docs" / "stage_a" / "pilot_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["format"] == FORMAT
    assert manifest["ic_family"] == "tanh_bandlimited"
    assert [len(manifest["splits"][name]) for name in ("train", "val", "test")] == [512, 128, 128]
    assert manifest["normalization"]["fit_on"] == "train"
    assert manifest["normalization"]["applied_to_files"] is False
    assert manifest["pilot"]["windowing"] == "none"
    assert manifest["pilot_config_sha256"] == sha256_json(HardPilotConfig().to_dict())
    assert manifest["solver_config_sha256"] == sha256_json(manifest["solver"])
    assert manifest["code_sha256"] == source_hashes()
    covered = manifest["splits"]["train"] + manifest["splits"]["val"] + manifest["splits"]["test"]
    assert sorted(covered) == list(range(768))
    assert len(manifest["instances"]) == 768
    stats = manifest["nu_stats"]["all"]
    assert stats["count"] == 768
    assert stats["min"] >= 0.005
    assert stats["min"] < 0.02
    assert stats["max"] <= 0.10
    assert manifest["invariants"]["max_abs_mean_drift"] <= 1e-12
    assert manifest["invariants"]["max_energy_increase"] <= 1e-10
    assert manifest["label_gate"]["pilot_n"] == 1024
    for record in manifest["instances"]:
        assert INSTANCE_KEYS <= set(record)
        assert len(record["field_sha256"]) == 64
        assert len(record["sha256"]) == 64
        assert 0.005 <= record["nu"] <= 0.10
    train_nu = [record["nu"] for record in manifest["instances"] if record["split"] == "train"]
    protocol = manifest["protocol"]
    assert protocol["format"] == PROTOCOL_FORMAT
    assert protocol["frozen_before_operator_training"] is True
    assert protocol["retuned_after_inverse_measurement"] is False
    assert protocol["ic_family"] == "tanh_bandlimited"
    assert protocol["split_sizes"] == {
        "train": 512,
        "val": 128,
        "test": 128,
        "unit": "problem_instance",
    }
    assert protocol["nu_range"]["min"] == 0.005
    assert protocol["nu_range"]["max"] == 0.10
    assert protocol["seeds"]["master_seed"] == HARD_MASTER_SEED
    assert protocol["seeds"]["split_seed"] == HARD_MASTER_SEED
    assert protocol["hard_ood"]["name"] == "hard_ood"
    assert protocol["hard_ood"]["quantile"] == 0.25
    assert protocol["hard_ood"]["method"] == "linear"
    assert protocol["hard_ood"]["fit_on"] == "train"
    threshold = hard_ood_threshold(train_nu)
    assert protocol["hard_ood"]["threshold_nu"] == pytest.approx(threshold)
    assert protocol["hard_ood"]["n_train_at_or_below"] == sum(nu <= threshold for nu in train_nu)
    assert protocol["label_gate"]["relative_l2_final_max"] == LABEL_REL_L2_MAX
    assert protocol == build_protocol(
        HardPilotConfig(),
        train_nu,
        manifest["solver_config_sha256"],
    )
    committed = json.loads(
        (ROOT / "docs" / "v02" / "pilot_protocol.json").read_text(encoding="utf-8")
    )
    assert committed == protocol

    stress = json.loads((ROOT / "docs" / "v02" / "inverse_stress.json").read_text(encoding="utf-8"))
    assert stress["format"] == STRESS_FORMAT
    assert stress["estimator"] == "sensors32_bursts"
    assert stress["operator_trained"] is False
    assert stress["observation_retuned"] is False
    assert stress["hard_ood"]["threshold_nu"] == protocol["hard_ood"]["threshold_nu"]
    primary = stress["sensors32_bursts"]
    assert primary["pattern"] == "sensors32_bursts"
    assert primary["n_instances"] == 128
    assert primary["mean_abs_error"] > 10.0 * STAGE5_SENSORS32_MEAN_ABS
    assert primary["mean_rel_error"] > 10.0 * STAGE5_SENSORS32_MEAN_REL
    assert primary["n_failures"] >= 1
    ood = stress["hard_ood_sensors32_bursts"]
    assert ood["n_failures"] >= 1
    assert stress["hard_ood"]["n_test"] == ood["n_instances"]
    assert stress["complement_sensors32_bursts"]["n_failures"] == 0
    assert stress["dense_reference"]["n_failures"] == 0
    assert stress["hard_ood_dense_reference"]["n_failures"] == 0
    assert len(stress["failures"]) == primary["n_failures"]
    assert all(row["nu"] <= protocol["hard_ood"]["threshold_nu"] for row in stress["failures"])
    assert all(row["rel_error"] > 0.5 and row["nu_hat_above_nu"] for row in stress["failures"])
    assert stress["uxx_alias"]["test_median"] > 0.2
    assert stress["uxx_alias"]["failure_median"] > stress["uxx_alias"]["test_median"]
    stage2 = json.loads(
        (ROOT / "docs" / "stage2" / "pilot_manifest.json").read_text(encoding="utf-8")
    )
    assert stage2["format"] == "pinnforge.burgers_pilot.v1"
    assert stage2["solver_config_sha256"] == STAGE2_SOLVER_HASH
    assert "protocol" not in stage2
    assert "ic_family" not in stage2
