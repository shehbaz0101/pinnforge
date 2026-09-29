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
    LABEL_REL_L2_MAX,
    M_KEEP,
    NU_MAX,
    NU_MIN,
    HardPilotConfig,
    draw_hard_initial_condition,
    estimate_hard_storage,
    generate_hard_pilot,
    high_mode_energy_fraction,
    source_hashes,
    spectral_slope_max,
)
from pinnforge.reference.numerical.initial import NU_MIN as STAGE2_NU_MIN
from pinnforge.reference.numerical.initial import draw_initial_condition
from pinnforge.reference.numerical.solver import SolverConfig, recommended_dt

ROOT = Path(__file__).resolve().parents[1]
STAGE2_PILOT_HASH = "1926d9a625136afca2af0d4dbc318cd77616ee0268ef80341e55c45daf156088"
STAGE2_SOLVER_HASH = "2a0c1c3f078ca81f15bf5c33874bc1403fa6a1a30bda1f484bbc6df5d0d48928"
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
    for name in ("estimate", "convergence", "pilot", "hard-estimate", "hard-convergence", "hard-pilot"):
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

    manifest = json.loads((ROOT / "docs" / "stage_a" / "pilot_manifest.json").read_text(encoding="utf-8"))
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
