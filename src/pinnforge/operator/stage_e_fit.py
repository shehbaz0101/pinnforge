"""Score the Stage E stress grid.

Torch is imported here. The sparse window is built in
:mod:`pinnforge.operator.stage_d` from the noisy mask. ``λ`` stays 0.
The oracle one-step path is not run. Fraction 0 on ``sensors32_bursts``
is checked against the Stage A record before any other condition is written.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.operator.inverse import training_baseline_nu
from pinnforge.operator.stage_d import (
    assert_published_baselines,
    baseline_tables,
    prepare_sparse_batch,
    residual_moments,
    sensor_field,
)
from pinnforge.operator.stage_d_fit import _require_checkpoint, _require_same_norm
from pinnforge.operator.stage_e import (
    CERTIFICATE_STRIDE,
    CONDITION_FORMAT,
    SCORES_FORMAT,
    SLICE_NAMES,
    aggregate_seed_summaries,
    apply_noise,
    breakdown_table,
    coarse_indexes,
    condition_by_id,
    golden_probe_indexes,
    golden_shrink,
    grid_from_protocol,
    is_failure,
    load_stage_e_protocol,
    local_minimum_count,
    noise_seeds_for,
    observation_spec,
    operator_applicable,
    recovery_from_operator_records,
    reference_examples,
    slice_summaries,
    stage_e_protocol_sha256,
    standard_normal_field,
    unimodal_cache_argmin,
    window_spec,
    write_stage_e_figures,
    write_stage_e_json,
)
from pinnforge.operator.windows import (
    FieldNorm,
    Trajectory,
    field_norm_from_manifest,
    load_pilot_manifest,
)
from pinnforge.reference.numerical.stress import frame_diagnostics

_NORM_ABS = 1e-12


def sensor_mse_on_grid(
    model: Any,
    norm: FieldNorm,
    windows: np.ndarray,
    targets: np.ndarray,
    slots: tuple[tuple[int, int, int, int], ...],
    grid: np.ndarray,
    *,
    n_sensors: int,
    nu_chunk: int = 32,
    instance_chunk: int = 64,
) -> np.ndarray:
    """Sensor mean square for every instance and every grid viscosity.

    The reduction matches the Stage D rollout: float64 physical sensors,
    summed over the masked nodes of the target bursts, divided by the node
    count. Viscosities are batched. The argmin is not taken here.
    """

    import torch

    from pinnforge.operator.fno import pack_inputs

    if not isinstance(norm, FieldNorm):
        raise TypeError("norm must be a FieldNorm")
    initial = np.asarray(windows, dtype=np.float64)
    observed = np.asarray(targets, dtype=np.float64)
    nodes = np.asarray(grid, dtype=np.float64)
    if initial.ndim != 3 or observed.ndim != 4:
        raise ValueError("curve tensors have the wrong rank")
    if initial.shape[0] != observed.shape[0]:
        raise ValueError("curve batch sizes do not match")
    if not np.isfinite(initial).all() or not np.isfinite(observed).all() or not np.isfinite(nodes).all():
        raise ValueError("curve inputs must be finite")
    if int(nu_chunk) < 1 or int(instance_chunk) < 1:
        raise ValueError("chunks must be positive")
    n_batch, frames, n_space = initial.shape
    if nodes.ndim == 1:
        shared = True
        n_nu = int(nodes.size)
    elif nodes.ndim == 2 and nodes.shape[0] == n_batch:
        shared = False
        n_nu = int(nodes.shape[1])
    else:
        raise ValueError("viscosity grid must be 1D or one row per instance")
    if n_space % int(n_sensors) != 0:
        raise ValueError("n_space must be divisible by n_sensors")
    n_steps = max(step for _, step, _, _ in slots) + 1
    stride = int(n_space) // int(n_sensors)
    needed = {step for _, step, _, _ in slots}
    norm_windows = torch.as_tensor(norm.normalize_u(initial), dtype=torch.float32)
    targets_t = torch.as_tensor(observed, dtype=torch.float64)
    u_std = float(norm.u_std)
    u_mean = float(norm.u_mean)
    nu_mean = float(norm.nu_mean)
    nu_std = float(norm.nu_std)
    mse = np.empty((n_batch, n_nu), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for i0 in range(0, n_batch, int(instance_chunk)):
            i1 = min(i0 + int(instance_chunk), n_batch)
            base = norm_windows[i0:i1]
            batch = i1 - i0
            goal = targets_t[i0:i1]
            for g0 in range(0, n_nu, int(nu_chunk)):
                g1 = min(g0 + int(nu_chunk), n_nu)
                width = g1 - g0
                current = (
                    base.unsqueeze(1)
                    .expand(batch, width, frames, n_space)
                    .reshape(batch * width, frames, n_space)
                    .contiguous()
                )
                if shared:
                    nu_phys = torch.as_tensor(nodes[g0:g1], dtype=torch.float64).unsqueeze(0).expand(batch, width)
                else:
                    nu_phys = torch.as_tensor(
                        np.ascontiguousarray(nodes[i0:i1, g0:g1]),
                        dtype=torch.float64,
                    )
                nu_norm = ((nu_phys.reshape(batch * width) - nu_mean) / nu_std).to(dtype=torch.float32)
                total = torch.zeros((batch, width), dtype=torch.float64)
                count = 0
                for step in range(n_steps):
                    predicted = model(pack_inputs(current, nu_norm))
                    if step in needed:
                        physical = predicted.detach().to(dtype=torch.float64) * u_std + u_mean
                        sensors = physical[:, :, ::stride].reshape(batch, width, frames, int(n_sensors))
                        for target_index, slot_step, local, length in slots:
                            if slot_step != step:
                                continue
                            diff = sensors[:, :, local : local + length, :] - goal[:, target_index].unsqueeze(1)
                            total = total + torch.sum(diff * diff, dim=(2, 3))
                            count += int(length) * int(n_sensors)
                    current = predicted
                if count < 1:
                    raise ValueError("sensor objective has no nodes")
                mse[i0:i1, g0:g1] = (total / count).cpu().numpy()
    if not np.isfinite(mse).all():
        raise ValueError("sensor mean square is not finite")
    return mse


def sensor_curve_hat(
    model: Any,
    norm: FieldNorm,
    windows: np.ndarray,
    targets: np.ndarray,
    slots: tuple[tuple[int, int, int, int], ...],
    grid: np.ndarray,
    *,
    n_sensors: int,
    verify_full: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Leftmost 191-grid minimizer of the sensor curve, and its range.

    Golden section reads a unimodal curve. A stride-16 subsample with more
    than one local minimum is scored on the full grid instead. ``verify_full``
    recomputes that full grid and refuses a mismatch. Identifiability is
    ``(max - min) / min`` on the nodes that were evaluated, which is the
    full curve when the fallback or the check runs.
    """

    nodes = np.asarray(grid, dtype=np.float64)
    if nodes.ndim != 1 or int(nodes.size) < 2:
        raise ValueError("viscosity grid must be a vector")
    n_batch = int(np.asarray(windows).shape[0])
    n_points = int(nodes.size)
    cache = np.full((n_batch, n_points), np.nan, dtype=np.float64)
    coarse = coarse_indexes(n_points, CERTIFICATE_STRIDE)
    coarse_mse = sensor_mse_on_grid(
        model,
        norm,
        windows,
        targets,
        slots,
        nodes[coarse],
        n_sensors=n_sensors,
        nu_chunk=max(int(coarse.size), 32),
        instance_chunk=max(n_batch, 1),
    )
    cache[:, coarse] = coarse_mse
    multi = np.array(
        [local_minimum_count(coarse_mse[row]) > 1 for row in range(n_batch)],
        dtype=bool,
    )
    left = np.zeros(n_batch, dtype=int)
    right = np.full(n_batch, n_points - 1, dtype=int)
    pending = ~multi
    for _ in range(n_points + 2):
        if not bool(pending.any()):
            break
        probe_rows: list[tuple[int, int, int]] = []
        scan_rows: list[int] = []
        for row in np.flatnonzero(pending):
            probes = golden_probe_indexes(int(left[row]), int(right[row]))
            if probes is None:
                scan_rows.append(int(row))
                continue
            i1, i2 = probes
            if np.isfinite(cache[row, i1]) and np.isfinite(cache[row, i2]):
                updated = golden_shrink(int(left[row]), int(right[row]), float(cache[row, i1]), float(cache[row, i2]))
                if updated == (int(left[row]), int(right[row])):
                    scan_rows.append(int(row))
                else:
                    left[row], right[row] = updated
                continue
            probe_rows.append((int(row), i1, i2))
        if scan_rows:
            _write_brackets(model, norm, windows, targets, slots, nodes, cache, left, right, scan_rows, n_sensors)
            pending[np.array(scan_rows, dtype=int)] = False
        if probe_rows:
            _write_probes(model, norm, windows, targets, slots, nodes, cache, probe_rows, n_sensors)
            for row, i1, i2 in probe_rows:
                updated = golden_shrink(int(left[row]), int(right[row]), float(cache[row, i1]), float(cache[row, i2]))
                if updated == (int(left[row]), int(right[row])):
                    continue
                left[row], right[row] = updated
    if bool(pending.any()):
        raise ValueError("grid search left an instance unfinished")
    if bool(multi.any()):
        rows = np.flatnonzero(multi)
        cache[rows] = sensor_mse_on_grid(
            model,
            norm,
            windows[rows],
            targets[rows],
            slots,
            nodes,
            n_sensors=n_sensors,
        )
    best = np.array([unimodal_cache_argmin(cache[row]) for row in range(n_batch)], dtype=int)
    if verify_full:
        full = sensor_mse_on_grid(model, norm, windows, targets, slots, nodes, n_sensors=n_sensors)
        if not np.array_equal(np.argmin(full, axis=1), best):
            raise ValueError("golden search left the full-grid argmin")
        cache = full
        best = np.argmin(full, axis=1)
    floor = cache[np.arange(n_batch), best]
    if np.any(floor <= 0.0) or not np.isfinite(floor).all():
        raise ValueError("sensor curve minimum is not positive")
    peak = np.nanmax(cache, axis=1)
    ident = (peak - floor) / floor
    if not np.isfinite(ident).all():
        raise ValueError("sensor-curve identifiability is not finite")
    return nodes[best], ident, int(np.sum(multi))


def _write_probes(
    model: Any,
    norm: FieldNorm,
    windows: np.ndarray,
    targets: np.ndarray,
    slots: tuple[tuple[int, int, int, int], ...],
    grid: np.ndarray,
    cache: np.ndarray,
    probes: list[tuple[int, int, int]],
    n_sensors: int,
) -> None:
    rows = np.array([row for row, _, _ in probes], dtype=int)
    nodes = np.empty((rows.size, 2), dtype=np.float64)
    for offset, (_, i1, i2) in enumerate(probes):
        nodes[offset, 0] = grid[i1]
        nodes[offset, 1] = grid[i2]
    mse = sensor_mse_on_grid(
        model,
        norm,
        windows[rows],
        targets[rows],
        slots,
        nodes,
        n_sensors=n_sensors,
        nu_chunk=2,
        instance_chunk=max(int(rows.size), 1),
    )
    for offset, (row, i1, i2) in enumerate(probes):
        cache[row, i1] = mse[offset, 0]
        cache[row, i2] = mse[offset, 1]


def _write_brackets(
    model: Any,
    norm: FieldNorm,
    windows: np.ndarray,
    targets: np.ndarray,
    slots: tuple[tuple[int, int, int, int], ...],
    grid: np.ndarray,
    cache: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    rows: list[int],
    n_sensors: int,
) -> None:
    by_width: dict[int, list[int]] = {}
    for row in rows:
        width = int(right[row]) - int(left[row]) + 1
        by_width.setdefault(width, []).append(int(row))
    for width, group in by_width.items():
        picked = np.array(group, dtype=int)
        nodes = np.empty((picked.size, width), dtype=np.float64)
        for offset, row in enumerate(group):
            nodes[offset] = grid[int(left[row]) : int(right[row]) + 1]
        mse = sensor_mse_on_grid(
            model,
            norm,
            windows[picked],
            targets[picked],
            slots,
            nodes,
            n_sensors=n_sensors,
            nu_chunk=max(width, 1),
            instance_chunk=max(int(picked.size), 1),
        )
        for offset, row in enumerate(group):
            cache[row, int(left[row]) : int(right[row]) + 1] = mse[offset]


def run_stage_e(
    *,
    protocol_path: Path,
    pilot_dir: Path,
    manifest_path: Path,
    run_root: Path,
    output: Path,
    figures: Path,
    part: str = "all",
    arm: str | None = None,
) -> dict[str, Any]:
    """Score the frozen grid. Existing condition files are kept."""

    if part not in ("all", "closed-form", "operator", "assemble"):
        raise ValueError(f"unknown stage E part {part!r}")
    source = Path(protocol_path)
    protocol = load_stage_e_protocol(source)
    digest = stage_e_protocol_sha256(source)
    manifest = load_pilot_manifest(Path(manifest_path))
    norm = field_norm_from_manifest(manifest)
    from pinnforge.operator.data import load_split_trajectories

    trajectories = load_split_trajectories(
        Path(pilot_dir),
        Path(manifest_path),
        ("test",),
        check_field_hash=True,
    )
    baseline = training_baseline_nu(Path(manifest_path))
    published = baseline_tables(trajectories, baseline, protocol)
    assert_published_baselines(published, Path(protocol["baseline_record"]))
    root = Path(run_root) / "stress"
    if part in ("all", "closed-form"):
        _score_closed_form(protocol, trajectories, baseline, published, root, digest)
    if part in ("all", "operator"):
        _score_operators(protocol, trajectories, norm, baseline, Path(run_root), root, digest, arm=arm)
    if part == "closed-form":
        return {"format": SCORES_FORMAT, "partial": "closed-form", "protocol_sha256": digest}
    if part == "operator":
        return {"format": SCORES_FORMAT, "partial": "operator", "protocol_sha256": digest}
    payload = assemble_stage_e(protocol, root, published, digest)
    write_stage_e_json(payload, Path(output))
    write_stage_e_figures(payload, protocol, Path(figures))
    return payload


def assemble_stage_e(
    protocol: dict[str, Any],
    stress_root: Path,
    published: dict[str, dict[str, Any]],
    digest: str,
) -> dict[str, Any]:
    """Reduce condition files into the committed score document."""

    conditions: dict[str, Any] = {}
    checkpoints = _checkpoint_meta(stress_root, protocol)
    for condition in protocol["conditions"]:
        identifier = str(condition["id"])
        conditions[identifier] = _assemble_condition(protocol, condition, stress_root, published)
    training = {
        name: _public_score(published["training_mean"][name]) for name in SLICE_NAMES
    }
    dense = {name: _public_score(published["dense_reference"][name]) for name in SLICE_NAMES}
    for identifier, block in conditions.items():
        block["training_mean"] = {"applicable": True, "slices": training}
    return {
        "format": SCORES_FORMAT,
        "protocol_sha256": digest,
        "protocol": "docs/v02/stage_e_stress_protocol.json",
        "lambda": 0.0,
        "lambda_retuned": False,
        "nu_search": {
            "optimizer": "uniform_grid_argmin",
            "n_grid": int(protocol["nu_search"]["n_grid"]),
            "tie_break": "smallest nu on the grid",
            "evaluator": "golden_section",
            "certificate_stride": CERTIFICATE_STRIDE,
            "full_grid_when": "the stride subsample has more than one local minimum",
            "reference_seed0_checked_against_full_grid": True,
        },
        "threshold_nu": float(protocol["hard_ood"]["threshold_nu"]),
        "failure_rule": protocol["failure_rule"],
        "primary_noise_seed": int(protocol["noise"]["primary_noise_seed"]),
        "reference_matches_stage_a": True,
        "checkpoints": checkpoints,
        "training_mean": training,
        "dense_ls_clean": dense,
        "conditions": conditions,
        "breakdowns": breakdown_table(protocol, conditions),
    }


def _score_closed_form(
    protocol: dict[str, Any],
    trajectories: list[Trajectory],
    baseline: float,
    published: dict[str, dict[str, Any]],
    root: Path,
    digest: str,
) -> None:
    from pinnforge.operator.inverse import observed_residual_terms

    master = int(protocol["noise"]["master_seed"])
    n_times = int(trajectories[0].u.shape[0])
    for condition in protocol["conditions"]:
        spec = observation_spec(protocol, int(condition["n_sensors"]), int(condition["n_bursts"]))
        alias = {
            item.instance_id: float(
                frame_diagnostics(
                    item.u,
                    frame_dt=float(spec.frame_dt),
                    n_sensors=int(spec.n_sensors or 0),
                )["alias_rel_l2"]
            )
            for item in trajectories
        }
        for noise_seed in noise_seeds_for(protocol, condition, "closed_form_ls"):
            destination = root / "closed_form" / str(condition["id"]) / f"seed_{noise_seed}.json"
            if destination.is_file():
                _require_existing(destination, digest, "closed_form_ls")
                continue
            epsilon = {
                item.instance_id: standard_normal_field(master, noise_seed, item.instance_id, item.u.shape)
                for item in trajectories
            }
            records = []
            for item in trajectories:
                field = apply_noise(item.u, epsilon[item.instance_id], float(condition["noise_fraction"]))
                hat = _closed_form_hat(field, spec)
                if hat is None:
                    raise ValueError(f"instance {item.instance_id} has no u_xx energy on {condition['id']}")
                absolute = abs(hat - item.nu)
                relative = absolute / item.nu
                advection, diffusion = observed_residual_terms(field, spec)
                residual_abs, residual_square = residual_moments(advection, diffusion, hat)
                truth_abs, truth_square = residual_moments(advection, diffusion, item.nu)
                records.append(
                    {
                        "instance_id": item.instance_id,
                        "nu": item.nu,
                        "nu_hat": hat,
                        "abs_error": absolute,
                        "rel_error": relative,
                        "failure": is_failure(hat, item.nu),
                        "nonpositive": hat <= 0.0,
                        "alias_rel_l2": alias[item.instance_id],
                        "mean_abs_residual": residual_abs,
                        "mean_abs_residual_at_truth": truth_abs,
                        "mean_square_residual": residual_square,
                        "mean_square_residual_at_truth": truth_square,
                    }
                )
            if (
                noise_seed == int(protocol["noise"]["primary_noise_seed"])
                and float(condition["noise_fraction"]) == 0.0
                and spec.name == "sensors32_bursts"
            ):
                _assert_reference(records, published, baseline, n_times)
            payload = _condition_payload(
                protocol,
                condition,
                digest,
                baseline,
                method="closed_form_ls",
                noise_seed=noise_seed,
                records=records,
                applicable=True,
            )
            write_stage_e_json(payload, destination)
            print(
                f"closed-form {condition['id']} seed {noise_seed} "
                f"failures={sum(1 for row in records if row['failure'])}",
                flush=True,
            )


def _score_operators(
    protocol: dict[str, Any],
    trajectories: list[Trajectory],
    norm: FieldNorm,
    baseline: float,
    run_root: Path,
    stress_root: Path,
    digest: str,
    *,
    arm: str | None = None,
) -> None:
    import torch

    from pinnforge.operator.checkpoint import load_fno_checkpoint

    torch.set_num_threads(max(1, int(__import__("os").environ.get("PINNFORGE_THREADS", "4"))))
    window = window_spec(protocol)
    grid = grid_from_protocol(protocol)
    master = int(protocol["noise"]["master_seed"])
    primary = int(protocol["noise"]["primary_noise_seed"])
    epsilon = {
        item.instance_id: standard_normal_field(master, primary, item.instance_id, item.u.shape)
        for item in trajectories
    }
    arms = ("data_only", "hybrid_1e-2") if arm is None else (arm,)
    if any(name not in protocol["operator"]["arms"] for name in arms):
        raise ValueError(f"unknown operator arm {arm}")
    specs = {
        str(condition["id"]): observation_spec(protocol, int(condition["n_sensors"]), int(condition["n_bursts"]))
        for condition in protocol["conditions"]
    }
    for arm_name in arms:
        for seed in protocol["operator"]["arms"][arm_name]["seeds"]:
            loaded = None
            for condition in protocol["conditions"]:
                destination = stress_root / "operator" / arm_name / f"seed_{int(seed)}" / f"{condition['id']}.json"
                if destination.is_file():
                    _require_existing(destination, digest, arm_name)
                    continue
                spec = specs[str(condition["id"])]
                applicable = operator_applicable(spec, window)
                if not applicable:
                    payload = _condition_payload(
                        protocol,
                        condition,
                        digest,
                        baseline,
                        method=arm_name,
                        noise_seed=primary,
                        records=[],
                        applicable=False,
                        seed=int(seed),
                        reason="no_target_burst",
                    )
                    write_stage_e_json(payload, destination)
                    continue
                if loaded is None:
                    checkpoint = Path(run_root) / arm_name / f"seed_{int(seed)}" / "checkpoint.pt"
                    loaded = load_fno_checkpoint(checkpoint)
                    _require_checkpoint(loaded, _stage_d_view(protocol), arm_name)
                    _require_same_norm(loaded.norm, norm)
                    if abs(float(loaded.norm.u_std) - float(norm.u_std)) > _NORM_ABS:
                        raise ValueError("checkpoint field norm drifted")
                started = time.perf_counter()
                noisy = _noisy(trajectories, epsilon, float(condition["noise_fraction"]))
                verify = str(condition["id"]) == "noise0_sensors32_bursts4" and int(seed) == 0
                records, n_full = _operator_records(
                    loaded.model,
                    norm,
                    noisy,
                    spec,
                    window,
                    grid,
                    verify_full=verify,
                )
                payload = _condition_payload(
                    protocol,
                    condition,
                    digest,
                    baseline,
                    method=arm_name,
                    noise_seed=primary,
                    records=records,
                    applicable=True,
                    seed=int(seed),
                    selected_epoch=int(loaded.epoch),
                    val_relative_l2=float(loaded.val_relative_l2),
                )
                write_stage_e_json(payload, destination)
                failures = sum(1 for row in records if row["failure"])
                elapsed = time.perf_counter() - started
                print(
                    f"operator {arm_name} seed {seed} {condition['id']} failures={failures} "
                    f"full_grid={n_full} seconds={elapsed:.1f}",
                    flush=True,
                )


def _operator_records(
    model: Any,
    norm: FieldNorm,
    trajectories: list[Trajectory],
    spec: Any,
    window: Any,
    grid: np.ndarray,
    *,
    verify_full: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    from pinnforge.operator.inverse import observed_residual_terms

    prepared = prepare_sparse_batch(trajectories, spec, window)
    hats, idents, n_full = sensor_curve_hat(
        model,
        norm,
        prepared["windows"],
        prepared["targets"],
        prepared["slots"],
        grid,
        n_sensors=int(spec.n_sensors),
        verify_full=verify_full,
    )
    n_times = int(trajectories[0].u.shape[0])
    records = []
    for index, item in enumerate(trajectories):
        hat = float(hats[index])
        ident = float(idents[index])
        if not np.isfinite(ident):
            raise ValueError(f"instance {item.instance_id} sensor curve is flat at 0")
        advection, diffusion = observed_residual_terms(
            sensor_field(prepared["masks"][index], spec, n_times),
            spec,
        )
        residual_abs, residual_square = residual_moments(advection, diffusion, hat)
        truth_abs, truth_square = residual_moments(advection, diffusion, item.nu)
        absolute = abs(hat - item.nu)
        records.append(
            {
                "instance_id": item.instance_id,
                "nu": item.nu,
                "nu_hat": hat,
                "abs_error": absolute,
                "rel_error": absolute / item.nu,
                "failure": is_failure(hat, item.nu),
                "nonpositive": hat <= 0.0,
                "identifiability": ident,
                "mean_abs_residual": residual_abs,
                "mean_abs_residual_at_truth": truth_abs,
                "mean_square_residual": residual_square,
                "mean_square_residual_at_truth": truth_square,
            }
        )
    return records, n_full


def _assemble_condition(
    protocol: dict[str, Any],
    condition: dict[str, Any],
    root: Path,
    published: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    identifier = str(condition["id"])
    primary = int(protocol["noise"]["primary_noise_seed"])
    closed = _read_condition(root / "closed_form" / identifier / f"seed_{primary}.json")
    block: dict[str, Any] = {
        "id": identifier,
        "noise_fraction": float(condition["noise_fraction"]),
        "n_sensors": int(condition["n_sensors"]),
        "n_bursts": int(condition["n_bursts"]),
        "axes": list(condition["axes"]),
        "noise_seed": primary,
        "closed_form_ls": _method_block(closed, protocol, "closed_form_ls"),
    }
    extra = {}
    for noise_seed in noise_seeds_for(protocol, condition, "closed_form_ls"):
        if noise_seed == primary:
            continue
        payload = _read_condition(root / "closed_form" / identifier / f"seed_{noise_seed}.json")
        extra[str(noise_seed)] = _method_block(payload, protocol, "closed_form_ls")
    if extra:
        block["closed_form_extra_seeds"] = extra
    for method, arm in (("operator_data_only", "data_only"), ("operator_hybrid_1e-2", "hybrid_1e-2")):
        block[method] = _operator_block(protocol, root, arm, identifier)
    if (
        identifier == "noise0_sensors32_bursts4"
        and not _reference_block_matches(block["closed_form_ls"], published["sensors32_bursts"])
    ):
        raise ValueError("assembled reference cell left the Stage A closed-form table")
    return block


def _operator_block(protocol: dict[str, Any], root: Path, arm: str, identifier: str) -> dict[str, Any]:
    seeds = [int(seed) for seed in protocol["operator"]["arms"][arm]["seeds"]]
    payloads = [
        _read_condition(root / "operator" / arm / f"seed_{seed}" / f"{identifier}.json") for seed in seeds
    ]
    if any(item.get("applicable") is False for item in payloads):
        if not all(item.get("applicable") is False for item in payloads):
            raise ValueError(f"{arm} {identifier} is not inapplicable on every seed")
        return {"applicable": False, "reason": str(payloads[0].get("reason", "no_target_burst"))}
    per_seed = []
    for payload in payloads:
        score = recovery_from_operator_records(payload["records"], _baseline_from_records(payload), arm)
        summary = slice_summaries(score, protocol, grid=grid_from_protocol(protocol))
        for name in summary:
            summary[name]["seed"] = int(payload["seed"])
        per_seed.append(summary)
    slices = aggregate_seed_summaries(per_seed)
    examples = _operator_examples(payloads)
    ident_values = []
    hard_values = []
    threshold = float(protocol["hard_ood"]["threshold_nu"])
    for payload in payloads:
        ident_values.append(float(np.median([float(row["identifiability"]) for row in payload["records"]])))
        hard_rows = [row for row in payload["records"] if float(row["nu"]) <= threshold]
        hard_values.append(float(np.median([float(row["identifiability"]) for row in hard_rows])))
    return {
        "applicable": True,
        "slices": slices,
        "examples": examples,
        "median_identifiability": float(np.mean(ident_values)),
        "hard_ood_median_identifiability": float(np.mean(hard_values)),
        "selected_epochs": [int(item["selected_epoch"]) for item in payloads],
        "val_relative_l2": [float(item["val_relative_l2"]) for item in payloads],
    }


def _method_block(payload: dict[str, Any], protocol: dict[str, Any], pattern: str) -> dict[str, Any]:
    score = recovery_from_operator_records(payload["records"], _baseline_from_records(payload), pattern)
    summary = slice_summaries(score, protocol, grid=grid_from_protocol(protocol) if pattern != "closed_form_ls" else None)
    for name in summary:
        summary[name]["seed"] = int(payload["noise_seed"])
    return {
        "applicable": True,
        "noise_seed": int(payload["noise_seed"]),
        "slices": summary,
        "examples": reference_examples(payload["records"]),
    }


def _operator_examples(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed = []
    for payload in payloads:
        rows = {int(row["instance_id"]): row for row in payload["records"]}
        by_seed.append(rows)
    ids = sorted(by_seed[0])
    merged = []
    for instance_id in ids:
        rows = [table[instance_id] for table in by_seed]
        rel = float(np.mean([float(row["rel_error"]) for row in rows]))
        hats = [float(row["nu_hat"]) for row in rows]
        merged.append(
            {
                "instance_id": instance_id,
                "nu": float(rows[0]["nu"]),
                "nu_hat": float(np.mean(hats)),
                "nu_hat_per_seed": hats,
                "rel_error": rel,
                "failure": any(bool(row["failure"]) for row in rows),
                "nonpositive": any(bool(row["nonpositive"]) for row in rows),
            }
        )
    return reference_examples(merged)


def _checkpoint_meta(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    reference = condition_by_id(protocol, "noise0_sensors32_bursts4")
    for arm in ("data_only", "hybrid_1e-2"):
        rows = []
        for seed in protocol["operator"]["arms"][arm]["seeds"]:
            payload = _read_condition(
                root / "operator" / arm / f"seed_{int(seed)}" / f"{reference['id']}.json"
            )
            rows.append(
                {
                    "seed": int(seed),
                    "selected_epoch": int(payload["selected_epoch"]),
                    "val_relative_l2": float(payload["val_relative_l2"]),
                }
            )
        meta[arm] = rows
    return meta


def _public_score(score: Any) -> dict[str, Any]:
    if hasattr(score, "to_dict"):
        payload = score.to_dict(include_instances=False)
        payload.pop("worst", None)
        payload.pop("worse_than_baseline_ids", None)
        payload.pop("failure_rule", None)
        return payload
    return dict(score)


def _baseline_from_records(payload: dict[str, Any]) -> float:
    if "baseline_nu" in payload:
        return float(payload["baseline_nu"])
    raise ValueError("condition file is missing baseline_nu")


def _condition_payload(
    protocol: dict[str, Any],
    condition: dict[str, Any],
    digest: str,
    baseline: float,
    *,
    method: str,
    noise_seed: int,
    records: list[dict[str, Any]],
    applicable: bool,
    seed: int | None = None,
    reason: str | None = None,
    selected_epoch: int | None = None,
    val_relative_l2: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": CONDITION_FORMAT,
        "protocol_sha256": digest,
        "method": method,
        "condition_id": str(condition["id"]),
        "noise_fraction": float(condition["noise_fraction"]),
        "n_sensors": int(condition["n_sensors"]),
        "n_bursts": int(condition["n_bursts"]),
        "noise_seed": int(noise_seed),
        "applicable": applicable,
        "baseline_nu": float(baseline),
        "records": records,
    }
    if seed is not None:
        payload["seed"] = int(seed)
    if reason is not None:
        payload["reason"] = reason
    if selected_epoch is not None:
        payload["selected_epoch"] = int(selected_epoch)
    if val_relative_l2 is not None:
        payload["val_relative_l2"] = float(val_relative_l2)
    del protocol
    return payload


def _closed_form_hat(field: np.ndarray, spec: Any) -> float | None:
    from pinnforge.operator.stage_e import least_squares_or_none

    return least_squares_or_none(field, spec)


def _noisy(
    trajectories: list[Trajectory],
    epsilon: dict[int, np.ndarray],
    fraction: float,
) -> list[Trajectory]:
    noisy: list[Trajectory] = []
    for item in trajectories:
        field = apply_noise(item.u, epsilon[item.instance_id], fraction)
        noisy.append(Trajectory(instance_id=item.instance_id, split=item.split, nu=item.nu, u=field))
    return noisy


def _assert_reference(
    records: list[dict[str, Any]],
    published: dict[str, dict[str, Any]],
    baseline: float,
    n_times: int,
) -> None:
    score = recovery_from_operator_records(records, baseline, "sensors32_bursts")
    reference = published["sensors32_bursts"]["full_test"]
    if score.n_failures != reference.n_failures:
        raise ValueError("clean sensors32_bursts failure count left the Stage A record")
    if abs(score.mean_rel_error - reference.mean_rel_error) > 1e-12:
        raise ValueError("clean sensors32_bursts mean relative error left the Stage A record")
    if abs(score.mean_abs_error - reference.mean_abs_error) > 1e-12:
        raise ValueError("clean sensors32_bursts mean absolute error left the Stage A record")
    del n_times


def _reference_block_matches(block: dict[str, Any], published: dict[str, Any]) -> bool:
    for name, score in published.items():
        got = block["slices"][name]
        if int(got["n_failures"]) != int(score.n_failures):
            return False
        if abs(float(got["mean_rel_error"]) - float(score.mean_rel_error)) > 1e-12:
            return False
    return True


def _read_condition(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("format") != CONDITION_FORMAT:
        raise ValueError(f"{path} is not a stage E condition file")
    return payload


def _require_existing(path: Path, digest: str, method: str) -> None:
    payload = _read_condition(path)
    if payload.get("protocol_sha256") != digest:
        raise ValueError(f"{path} was scored under a different protocol")
    if method == "closed_form_ls" and payload.get("method") != "closed_form_ls":
        raise ValueError(f"{path} is not a closed-form file")


def _stage_d_view(protocol: dict[str, Any]) -> dict[str, Any]:
    """The checkpoint check reads arms and the window from a Stage D-shaped dict."""

    return {
        "arms": protocol["operator"]["arms"],
        "window": protocol["window"],
    }

