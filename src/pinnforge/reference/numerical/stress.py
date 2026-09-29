"""Stage 5 residual least squares on the harder Burgers pilot.

The estimator is :data:`pinnforge.operator.inverse.PREREGISTERED_OBSERVATION`
(``sensors32_bursts``). This module does not change that pattern and does
not train an operator. The harder manifest is
``pinnforge.burgers_hard_pilot.v1``, so the Stage 2 loader is not used.

``hard_ood`` is the slice ``ν <=`` the training-split quartile recorded in
the pilot protocol. The training mean is the baseline for every slice.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.operator.inverse import (
    FAILURE_RELATIVE_ERROR,
    PREREGISTERED_FRAME_DT,
    PREREGISTERED_N_SENSORS,
    PREREGISTERED_OBSERVATION,
    ObservationSpec,
    RecoveryScore,
    dense_reference_observation,
    score_trajectories,
)
from pinnforge.operator.windows import Trajectory
from pinnforge.reference.numerical.checks import relative_l2
from pinnforge.reference.numerical.dataset import _field_sha256
from pinnforge.reference.numerical.harder import (
    FORMAT,
    HARD_OOD_NAME,
    PROTOCOL_FORMAT,
    STAGE2_NU_MIN,
    HardPilotConfig,
    build_protocol,
    high_mode_energy_fraction,
)
from pinnforge.reference.numerical.solver import spectral_derivative

STRESS_FORMAT = "pinnforge.burgers_hard_inverse_stress.v1"
# Stage 5 easy-pilot test table in STAGE5_REPORT.md. Those trajectories are
# not reloaded here. The ratios below use these literals as the reference.
STAGE5_SENSORS32_MEAN_ABS = 6.091698430189056e-05
STAGE5_SENSORS32_MEAN_REL = 1.3303134076090104e-03
STAGE5_SENSORS32_MAX_REL = 1.8379471956251635e-02
STAGE5_SENSORS32_FAILURES = 0
STAGE5_SENSORS32_CORRELATION = 0.9999942242288621
STAGE5_DENSE_MEAN_ABS = 2.778977972493421e-05
STAGE5_DENSE_MEAN_REL = 4.9296765640965002e-04
STAGE5_DENSE_MAX_REL = 1.2616261445804054e-03
_ALIAS_MODE_CUTOFF = 16


def load_hard_manifest(path: Path) -> dict[str, Any]:
    """Read a harder-pilot manifest and check its protocol block."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != FORMAT:
        raise ValueError(f"manifest format must be {FORMAT}")
    protocol = payload.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("format") != PROTOCOL_FORMAT:
        raise ValueError("manifest is missing the harder-pilot protocol")
    expected = build_protocol(
        _config_from_pilot(payload["pilot"]),
        [float(row["nu"]) for row in payload["instances"] if row["split"] == "train"],
        str(payload["solver_config_sha256"]),
    )
    if protocol != expected:
        raise ValueError("manifest protocol does not match the training viscosities")
    return payload


def load_hard_trajectories(
    pilot_dir: Path,
    manifest: dict[str, Any],
    split: str,
) -> list[Trajectory]:
    """Load one split. ``field_sha256`` in the manifest must match the array."""

    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r}")
    root = Path(pilot_dir)
    trajectories: list[Trajectory] = []
    for record in manifest["instances"]:
        if record["split"] != split:
            continue
        with np.load(root / record["path"]) as archive:
            field = np.asarray(archive["u"], dtype=np.float64)
            nu = float(archive["nu"])
            instance_id = int(archive["instance_id"])
        if _field_sha256(field) != record["field_sha256"]:
            raise ValueError(f"field hash mismatch for instance {record['instance_id']}")
        if instance_id != int(record["instance_id"]) or nu != float(record["nu"]):
            raise ValueError(f"instance {record['instance_id']} metadata does not match the file")
        trajectories.append(Trajectory(instance_id=instance_id, split=split, nu=nu, u=field))
    if not trajectories:
        raise ValueError(f"split {split!r} is empty")
    return trajectories


def frame_diagnostics(field: np.ndarray, *, frame_dt: float, n_sensors: int) -> dict[str, Any]:
    """Slope and sensor ``u_xx`` alias at the steepest saved frame.

    ``alias_rel_l2`` is the relative discrete L2 between the spectral second
    derivative of the equispaced sensors and the full-grid spectral second
    derivative restricted to those same nodes. It is computed on the saved
    frame where ``||u_x||_∞`` is largest. ``high_mode_energy_above_16`` is
    the energy fraction of ``u`` itself above mode 16 on that frame, which
    is a different quantity from the second-derivative alias.
    """

    values = np.asarray(field, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("field must have shape (n_times, n_space)")
    if values.shape[1] % n_sensors != 0:
        raise ValueError("n_space must be divisible by n_sensors")
    slopes = np.max(np.abs(spectral_derivative(values, order=1)), axis=1)
    frame = int(np.argmax(slopes))
    full_xx = spectral_derivative(values[frame], order=2)
    stride = values.shape[1] // n_sensors
    restricted = full_xx[::stride]
    coarse = spectral_derivative(values[frame, ::stride], order=2)
    return {
        "frame": frame,
        "t": float(frame) * float(frame_dt),
        "slope_t0": float(slopes[0]),
        "slope_max": float(slopes[frame]),
        "alias_rel_l2": relative_l2(coarse, restricted),
        "high_mode_energy_above_16": high_mode_energy_fraction(
            values[frame],
            cutoff=_ALIAS_MODE_CUTOFF,
        ),
    }


def run_inverse_stress(pilot_dir: Path, manifest_path: Path) -> dict[str, Any]:
    """Score the test split and the preregistered ``hard_ood`` slice.

    The baseline is the harder pilot's training-split mean. Stage 5 numbers
    in the payload are the easy-pilot table, not a second fit.
    """

    manifest = load_hard_manifest(manifest_path)
    protocol = manifest["protocol"]
    threshold = float(protocol["hard_ood"]["threshold_nu"])
    train_nu = np.asarray(
        [float(row["nu"]) for row in manifest["instances"] if row["split"] == "train"],
        dtype=np.float64,
    )
    baseline = float(np.mean(train_nu))
    if baseline != float(manifest["nu_stats"]["train"]["mean"]):
        raise ValueError("training viscosity mean does not match nu_stats")
    test = load_hard_trajectories(pilot_dir, manifest, "test")
    _require_common_grid(test)
    n_times = int(test[0].u.shape[0])
    sparse = score_trajectories(test, PREREGISTERED_OBSERVATION, baseline)
    dense_spec = dense_reference_observation(n_times, frame_dt=PREREGISTERED_FRAME_DT)
    dense = score_trajectories(test, dense_spec, baseline)
    hard = [item for item in test if item.nu <= threshold]
    complement = [item for item in test if item.nu > threshold]
    below_floor = [item for item in test if item.nu < STAGE2_NU_MIN]
    if not hard or not complement or not below_floor:
        raise ValueError("a required test slice is empty")
    diagnostics = {
        item.instance_id: frame_diagnostics(
            item.u,
            frame_dt=PREREGISTERED_FRAME_DT,
            n_sensors=PREREGISTERED_N_SENSORS,
        )
        for item in test
    }
    return {
        "format": STRESS_FORMAT,
        "estimator": PREREGISTERED_OBSERVATION.name,
        "estimator_symbol": "pinnforge.operator.inverse.PREREGISTERED_OBSERVATION",
        "observation_retuned": False,
        "operator_trained": False,
        "failure_rule": f"nu_hat <= 0 or rel_error > {FAILURE_RELATIVE_ERROR}",
        "baseline_nu": baseline,
        "baseline_source": "mean viscosity of the harder-pilot training split",
        "pilot_config_sha256": manifest["pilot_config_sha256"],
        "solver_config_sha256": manifest["solver_config_sha256"],
        "protocol_format": PROTOCOL_FORMAT,
        "hard_ood": {
            "name": HARD_OOD_NAME,
            "threshold_nu": threshold,
            "quantile": protocol["hard_ood"]["quantile"],
            "method": protocol["hard_ood"]["method"],
            "fit_on": protocol["hard_ood"]["fit_on"],
            "n_train_at_or_below": protocol["hard_ood"]["n_train_at_or_below"],
            "n_test": len(hard),
            "n_test_complement": len(complement),
            "n_test_below_stage2_floor": len(below_floor),
            "stage2_nu_min": STAGE2_NU_MIN,
        },
        "stage5_easy_pilot": {
            "source": "STAGE5_REPORT.md test table on the Stage 2 pilot",
            "sensors32_bursts": {
                "mean_abs_error": STAGE5_SENSORS32_MEAN_ABS,
                "mean_rel_error": STAGE5_SENSORS32_MEAN_REL,
                "max_rel_error": STAGE5_SENSORS32_MAX_REL,
                "n_failures": STAGE5_SENSORS32_FAILURES,
                "correlation": STAGE5_SENSORS32_CORRELATION,
            },
            "dense_reference": {
                "mean_abs_error": STAGE5_DENSE_MEAN_ABS,
                "mean_rel_error": STAGE5_DENSE_MEAN_REL,
                "max_rel_error": STAGE5_DENSE_MAX_REL,
                "n_failures": 0,
            },
        },
        "sensors32_bursts": _with_stage5_ratios(sparse, kind="sensors32"),
        "hard_ood_sensors32_bursts": _with_stage5_ratios(
            score_trajectories(hard, PREREGISTERED_OBSERVATION, baseline),
            kind="sensors32",
        ),
        "complement_sensors32_bursts": _with_stage5_ratios(
            score_trajectories(complement, PREREGISTERED_OBSERVATION, baseline),
            kind="sensors32",
        ),
        "below_stage2_floor_sensors32_bursts": _with_stage5_ratios(
            score_trajectories(below_floor, PREREGISTERED_OBSERVATION, baseline),
            kind="sensors32",
        ),
        "dense_reference": _with_stage5_ratios(dense, kind="dense"),
        "hard_ood_dense_reference": _with_stage5_ratios(
            score_trajectories(hard, dense_spec, baseline),
            kind="dense",
        ),
        "uxx_alias": _alias_summary(
            diagnostics,
            hard_ids={item.instance_id for item in hard},
            failure_ids={row.instance_id for row in sparse.instances if row.failure},
        ),
        "failures": _example_rows(sparse, diagnostics, failure_only=True),
        "worse_than_baseline": _example_rows(sparse, diagnostics, failure_only=False),
    }


def write_inverse_stress(payload: dict[str, Any], path: Path) -> Path:
    """Write the stress record. Parent directories are created."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _with_stage5_ratios(score: RecoveryScore, *, kind: str) -> dict[str, Any]:
    payload = score.to_dict(include_instances=False)
    if kind == "sensors32":
        reference_abs = STAGE5_SENSORS32_MEAN_ABS
        reference_rel = STAGE5_SENSORS32_MEAN_REL
    elif kind == "dense":
        reference_abs = STAGE5_DENSE_MEAN_ABS
        reference_rel = STAGE5_DENSE_MEAN_REL
    else:
        raise ValueError(f"unknown Stage 5 reference {kind!r}")
    payload["ratio_mean_abs_error_vs_stage5"] = payload["mean_abs_error"] / reference_abs
    payload["ratio_mean_rel_error_vs_stage5"] = payload["mean_rel_error"] / reference_rel
    return payload


def _alias_summary(
    diagnostics: dict[int, dict[str, Any]],
    *,
    hard_ids: set[int],
    failure_ids: set[int],
) -> dict[str, Any]:
    rows = [
        {"instance_id": instance_id, **diagnostics[instance_id]}
        for instance_id in sorted(diagnostics)
    ]
    alias = np.asarray([row["alias_rel_l2"] for row in rows], dtype=np.float64)
    hard_alias = np.asarray(
        [row["alias_rel_l2"] for row in rows if row["instance_id"] in hard_ids],
        dtype=np.float64,
    )
    failure_alias = np.asarray(
        [row["alias_rel_l2"] for row in rows if row["instance_id"] in failure_ids],
        dtype=np.float64,
    )
    summary = {
        "definition": (
            "At the saved frame of maximum ||u_x||_inf, relative discrete L2 "
            "between the spectral u_xx of the 32 equispaced sensors and the "
            "full-grid spectral u_xx restricted to those sensors. "
            "high_mode_energy_above_16 is the energy fraction of u, not of u_xx."
        ),
        "n_sensors": PREREGISTERED_N_SENSORS,
        "frame_dt": PREREGISTERED_FRAME_DT,
        "per_instance": rows,
        "test_median": float(np.median(alias)),
        "test_max": float(np.max(alias)),
        "hard_ood_median": float(np.median(hard_alias)),
    }
    if failure_alias.size:
        summary["failure_median"] = float(np.median(failure_alias))
        summary["failure_min"] = float(np.min(failure_alias))
        summary["failure_max"] = float(np.max(failure_alias))
    return summary


def _example_rows(
    score: RecoveryScore,
    diagnostics: dict[int, dict[str, Any]],
    *,
    failure_only: bool,
) -> list[dict[str, Any]]:
    if failure_only:
        chosen = [row for row in score.instances if row.failure]
    else:
        chosen = [row for row in score.instances if row.abs_error > row.baseline_abs_error]
    chosen.sort(key=lambda row: (-row.rel_error, row.instance_id))
    examples: list[dict[str, Any]] = []
    for row in chosen:
        detail = diagnostics[row.instance_id]
        payload = row.to_dict()
        payload.update(detail)
        payload["nu_hat_above_nu"] = bool(row.nu_hat > row.nu)
        examples.append(payload)
    return examples


def _require_common_grid(trajectories: list[Trajectory]) -> None:
    shape = trajectories[0].u.shape
    if any(item.u.shape != shape for item in trajectories):
        raise ValueError("test trajectories do not share a grid")
    spec = PREREGISTERED_OBSERVATION
    if not isinstance(spec, ObservationSpec):
        raise TypeError("preregistered observation is missing")
    spec.check_grid(int(shape[0]), int(shape[1]))


def _config_from_pilot(pilot: dict[str, Any]) -> HardPilotConfig:
    return HardPilotConfig(
        n_train=int(pilot["n_train"]),
        n_val=int(pilot["n_val"]),
        n_test=int(pilot["n_test"]),
        n=int(pilot["n"]),
        dt=float(pilot["dt"]),
        t_final=float(pilot["t_final"]),
        save_dt=float(pilot["save_dt"]),
        master_seed=int(pilot["master_seed"]),
        split_seed=int(pilot["split_seed"]),
        n_modes=int(pilot["n_modes"]),
        m_keep=int(pilot["m_keep"]),
        beta=float(pilot["beta"]),
        amplitude_decay=float(pilot["amplitude_decay"]),
        batch_size=int(pilot["batch_size"]),
        nu_min=float(pilot["nu_min"]),
        nu_max=float(pilot["nu_max"]),
    )
