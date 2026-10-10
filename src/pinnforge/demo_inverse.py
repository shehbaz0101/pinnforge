"""CPU demo: recover Burgers viscosity from 32 sparse sensors.

``pinnforge demo burgers-inverse`` builds a small periodic Burgers set
with the Fourier solver, trains a small 1D FNO on the train and validation
splits, and scores two estimators on the held-out instances.

The closed form is the normal equation of the central Burgers residual on
32 equispaced sensors. The operator estimator lifts those sensors, rolls
the trained FNO forward, and picks the viscosity on a fixed grid whose
predicted sensors best match the later frames. The true viscosity is used
only to score the estimates.

This is a short illustration. It does not load the Stage D pilot, and it
does not use the ``sensors32_bursts`` mask. Importing this module does
not import torch.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from pinnforge.demo import DemoDependencyError
from pinnforge.figures import write_bars_png, write_field_png, write_scatter_png
from pinnforge.operator.inverse import (
    FAILURE_RELATIVE_ERROR,
    ObservationSpec,
    least_squares_viscosity,
)
from pinnforge.operator.stage_d import lift_sensors
from pinnforge.operator.windows import (
    FieldNorm,
    Trajectory,
    WindowSpec,
    build_window_datasets,
    window_starts,
)
from pinnforge.reference.numerical.initial import draw_initial_condition
from pinnforge.reference.numerical.solver import solve

DEMO_FORMAT = "pinnforge.burgers_inverse_demo.v1"
DEFAULT_INVERSE_OUTPUT = "runs/burgers-inverse"
_CLOSED = (31, 78, 121)


def _require_int(value: int, label: str, *, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


def _require_positive(value: float, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite and > 0")
    if float(value) <= 0.0:
        raise ValueError(f"{label} must be finite and > 0")

_OPERATOR = (184, 92, 56)
_BASELINE = (90, 90, 90)
_FIELD_LATER = (47, 107, 79)


@dataclass(frozen=True, slots=True)
class InverseDemoConfig:
    """Locked settings for one demo run.

    The published command uses :data:`PUBLISHED_INVERSE_DEMO`. Tests may
    pass a smaller config. The sample file on disk is not involved.
    """

    seed: int = 0
    n_space: int = 64
    n_modes: int = 4
    nu_min: float = 0.02
    nu_max: float = 0.08
    n_train: int = 16
    n_val: int = 4
    n_test: int = 4
    dt: float = 0.002
    t_final: float = 0.48
    save_dt: float = 0.02
    n_sensors: int = 32
    input_frames: int = 4
    output_frames: int = 4
    stride: int = 4
    width: int = 24
    modes: int = 8
    layers: int = 2
    epochs: int = 40
    batch_size: int = 16
    lr: float = 1e-3
    nu_grid: int = 61

    def __post_init__(self) -> None:
        _require_int(self.seed, "seed", minimum=0)
        _require_int(self.n_space, "n_space", minimum=4)
        if self.n_space % 2 != 0:
            raise ValueError("n_space must be even")
        _require_int(self.n_modes, "n_modes", minimum=1)
        if self.n_modes >= self.n_space // 2:
            raise ValueError("n_modes must be below the Nyquist mode n_space / 2")
        _require_positive(self.nu_min, "nu_min")
        _require_positive(self.nu_max, "nu_max")
        if self.nu_max < self.nu_min:
            raise ValueError("nu_max must be >= nu_min")
        _require_int(self.n_train, "n_train", minimum=2)
        _require_int(self.n_val, "n_val", minimum=1)
        _require_int(self.n_test, "n_test", minimum=1)
        _require_positive(self.dt, "dt")
        _require_positive(self.t_final, "t_final")
        _require_positive(self.save_dt, "save_dt")
        _require_int(self.n_sensors, "n_sensors", minimum=4)
        if self.n_sensors % 2 != 0 or self.n_space % self.n_sensors != 0:
            raise ValueError("n_sensors must be an even divisor of n_space")
        _require_int(self.input_frames, "input_frames", minimum=1)
        _require_int(self.output_frames, "output_frames", minimum=1)
        _require_int(self.stride, "stride", minimum=1)
        if self.stride != self.output_frames:
            raise ValueError("stride must equal output_frames so the rollout hands off one block")
        _require_int(self.width, "width", minimum=1)
        _require_int(self.modes, "modes", minimum=1)
        _require_int(self.layers, "layers", minimum=1)
        _require_int(self.epochs, "epochs", minimum=1)
        _require_int(self.batch_size, "batch_size", minimum=1)
        _require_positive(self.lr, "lr")
        _require_int(self.nu_grid, "nu_grid", minimum=2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "n_space": self.n_space,
            "n_modes": self.n_modes,
            "nu_min": self.nu_min,
            "nu_max": self.nu_max,
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
            "dt": self.dt,
            "t_final": self.t_final,
            "save_dt": self.save_dt,
            "n_sensors": self.n_sensors,
            "input_frames": self.input_frames,
            "output_frames": self.output_frames,
            "stride": self.stride,
            "width": self.width,
            "modes": self.modes,
            "layers": self.layers,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "nu_grid": self.nu_grid,
            "loss": "data",
            "device": "cpu",
            "torch_threads": 1,
        }


PUBLISHED_INVERSE_DEMO = InverseDemoConfig()


@dataclass(frozen=True, slots=True)
class InverseDemoResult:
    """Paths and the text the CLI prints."""

    output_dir: Path
    summary: str
    record: dict[str, Any]


def format_result_number(value: float) -> str:
    """Fixed scientific format shared by the JSON table and the README."""

    return f"{float(value):.8e}"


def results_table_markdown(record: dict[str, Any]) -> str:
    """Markdown table of the three estimators. Built only from ``record``."""

    sensors = int(record["config"]["n_sensors"])
    rows = (
        ("closed_form", f"closed form, {sensors} sensors"),
        ("operator", "FNO sensor match"),
        ("baseline", "training-mean constant"),
    )
    lines = [
        "| estimator | n | mean abs error | mean rel error | max rel error | failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label in rows:
        item = record["aggregates"][key]
        lines.append(
            "| {label} | {n} | {mae} | {rel} | {max_rel} | {failures} |".format(
                label=label,
                n=int(item["n"]),
                mae=format_result_number(item["mean_abs_error"]),
                rel=format_result_number(item["mean_rel_error"]),
                max_rel=format_result_number(item["max_rel_error"]),
                failures=int(item["n_failures"]),
            )
        )
    return "\n".join(lines) + "\n"


def run_burgers_inverse_demo(
    output_dir: Path,
    *,
    output_label: str,
    config: InverseDemoConfig | None = None,
) -> InverseDemoResult:
    """Train the small FNO, recover viscosity, and write the report files.

    ``output_dir`` is the resolved directory. ``output_label`` is the
    sandbox-relative path stored in the JSON. Torch is imported here.

    Raises:
        DemoDependencyError: the ``ml`` extra is not installed.
        ValueError: the output path is not a directory we can create, or
            a metric is not finite.
    """

    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a Path")
    if not isinstance(output_label, str) or output_label == "":
        raise ValueError("output_label must be a non-empty relative path")
    settings = config if config is not None else PUBLISHED_INVERSE_DEMO
    if not isinstance(settings, InverseDemoConfig):
        raise TypeError("config must be an InverseDemoConfig")
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path is not a directory: {output_label}")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    trajectories = _integrate(settings)
    windows = WindowSpec(settings.input_frames, settings.output_frames, settings.stride)
    labeled = _label(trajectories, settings)
    norm = _training_norm(labeled)
    datasets = build_window_datasets(labeled, windows, norm)
    model, history, parameter_count = _train(datasets, settings)
    from pinnforge.operator.checkpoint import save_fno_checkpoint
    from pinnforge.operator.residual import LossConfig

    selected = history[0]
    for row in history[1:]:
        if row.val_relative_l2 < selected.val_relative_l2:
            selected = row
    checkpoint = output_dir / "checkpoint.pt"
    save_fno_checkpoint(
        checkpoint,
        model,
        epoch=selected.epoch,
        spec=windows,
        norm=norm,
        seed=settings.seed,
        val_relative_l2=selected.val_relative_l2,
        loss=LossConfig(dt=settings.save_dt),
    )
    test = [item for item in labeled if item.split == "test"]
    grid = np.linspace(settings.nu_min, settings.nu_max, settings.nu_grid)
    instances = _recover(test, model, norm, windows, settings, float(norm.nu_mean), grid)
    wall_clock = time.perf_counter() - started
    record = _record(
        settings,
        output_label,
        norm,
        datasets,
        history,
        selected,
        parameter_count,
        instances,
        grid,
        wall_clock,
    )
    _write_plots(output_dir, test, instances, settings)
    _write_reports(output_dir, record)
    summary = _summary(output_label, record)
    return InverseDemoResult(output_dir=output_dir, summary=summary, record=record)


def _integrate(settings: InverseDemoConfig) -> list[tuple[int, float, np.ndarray, np.ndarray]]:
    total = settings.n_train + settings.n_val + settings.n_test
    rows: list[tuple[int, float, np.ndarray, np.ndarray]] = []
    shape: tuple[int, int] | None = None
    space: np.ndarray | None = None
    for instance_id in range(total):
        initial = draw_initial_condition(
            settings.seed,
            instance_id,
            settings.n_space,
            n_modes=settings.n_modes,
            nu_min=settings.nu_min,
            nu_max=settings.nu_max,
        )
        solved = solve(
            initial.u0,
            initial.nu,
            dt=settings.dt,
            t_final=settings.t_final,
            save_dt=settings.save_dt,
        )
        field = np.asarray(solved.u, dtype=np.float64)
        if shape is None:
            shape = (int(field.shape[0]), int(field.shape[1]))
            space = np.asarray(solved.x, dtype=np.float64)
        if (int(field.shape[0]), int(field.shape[1])) != shape:
            raise ValueError("solver frames do not share a shape")
        rows.append((instance_id, float(initial.nu), field, space))
    return rows


def _label(
    rows: list[tuple[int, float, np.ndarray, np.ndarray]],
    settings: InverseDemoConfig,
) -> list[Trajectory]:
    labeled: list[Trajectory] = []
    for instance_id, nu, field, _space in rows:
        if instance_id < settings.n_train:
            split = "train"
        elif instance_id < settings.n_train + settings.n_val:
            split = "val"
        else:
            split = "test"
        labeled.append(Trajectory(instance_id=instance_id, split=split, nu=nu, u=field))
    return labeled


def _training_norm(items: list[Trajectory]) -> FieldNorm:
    train = [item for item in items if item.split == "train"]
    values = np.concatenate([item.u.ravel() for item in train])
    viscosities = np.asarray([item.nu for item in train], dtype=np.float64)
    u_std = float(np.std(values, ddof=0))
    nu_std = float(np.std(viscosities, ddof=0))
    if u_std <= 0.0 or nu_std <= 0.0:
        raise ValueError("training fields and viscosities need a positive spread")
    return FieldNorm(
        u_mean=float(np.mean(values)),
        u_std=u_std,
        nu_mean=float(np.mean(viscosities)),
        nu_std=nu_std,
    )


def _train(datasets: dict[str, Any], settings: InverseDemoConfig) -> tuple[Any, list[Any], int]:
    from pinnforge.ml_import import InstallHint

    try:
        import torch

        from pinnforge.operator.train import fit_fno
    except InstallHint as exc:
        raise DemoDependencyError(str(exc)) from exc
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
        raise DemoDependencyError(str(InstallHint())) from exc
    torch.set_num_threads(1)
    model, history = fit_fno(
        datasets["train"],
        datasets["val"],
        width=settings.width,
        modes=settings.modes,
        n_layers=settings.layers,
        epochs=settings.epochs,
        batch_size=settings.batch_size,
        lr=settings.lr,
        seed=settings.seed,
    )
    return model, history, int(model.parameter_count())


def _recover(
    test: list[Trajectory],
    model: Any,
    norm: FieldNorm,
    windows: WindowSpec,
    settings: InverseDemoConfig,
    baseline_nu: float,
    grid: np.ndarray,
) -> list[dict[str, Any]]:
    import torch

    from pinnforge.operator.fno import pack_inputs

    spec = ObservationSpec(
        name=f"sensors{settings.n_sensors}",
        n_sensors=settings.n_sensors,
        series=(tuple(range(int(test[0].u.shape[0]))),),
        frame_dt=settings.save_dt,
    )
    starts = window_starts(int(test[0].u.shape[0]), windows)
    if not starts:
        raise ValueError("each trajectory must contain at least one FNO window")
    stride = settings.n_space // settings.n_sensors
    model.eval()
    rows: list[dict[str, Any]] = []
    for item in test:
        closed = least_squares_viscosity(item.u, spec)
        observed = np.ascontiguousarray(item.u[:, ::stride])
        lifted = lift_sensors(observed[: settings.input_frames], settings.n_space)
        scores = _grid_sensor_mse(
            model,
            pack_inputs,
            torch,
            norm,
            lifted,
            observed,
            windows,
            starts,
            stride,
            grid,
        )
        choice = int(np.argmin(scores))
        operator = float(grid[choice])
        rows.append(_instance_row(item, closed, operator, baseline_nu, float(scores[choice])))
    return rows


def _grid_sensor_mse(
    model: Any,
    pack_inputs: Any,
    torch: Any,
    norm: FieldNorm,
    lifted: np.ndarray,
    observed: np.ndarray,
    windows: WindowSpec,
    starts: tuple[int, ...],
    stride: int,
    grid: np.ndarray,
) -> np.ndarray:
    current = np.repeat(lifted[None, :, :], grid.shape[0], axis=0)
    total = np.zeros(grid.shape[0], dtype=np.float64)
    count = 0
    viscosities = np.asarray(grid, dtype=np.float64)
    for start in starts:
        predicted = _predict(model, pack_inputs, torch, norm, current, viscosities)
        future = observed[start + windows.input_frames : start + windows.span()]
        predicted_sensors = predicted[:, :, ::stride]
        residual = predicted_sensors - future[None, :, :]
        total += np.sum(residual * residual, axis=(1, 2))
        count += int(future.size)
        current = predicted
    if count < 1:
        raise ValueError("operator objective has no sensor samples")
    return total / count


def _predict(
    model: Any,
    pack_inputs: Any,
    torch: Any,
    norm: FieldNorm,
    windows: np.ndarray,
    viscosities: np.ndarray,
) -> np.ndarray:
    inputs = torch.as_tensor(norm.normalize_u(windows), dtype=torch.float32)
    nu = torch.as_tensor(norm.normalize_nu(viscosities), dtype=torch.float32)
    with torch.no_grad():
        predicted = model(pack_inputs(inputs, nu)).detach().cpu().numpy()
    physical = norm.denormalize_u(predicted)
    if not np.isfinite(physical).all():
        raise ValueError("FNO prediction is not finite")
    return np.asarray(physical, dtype=np.float64)


def _instance_row(
    item: Trajectory,
    closed: float,
    operator: float,
    baseline: float,
    sensor_mse: float,
) -> dict[str, Any]:
    estimates = {
        "closed_form": float(closed),
        "operator": float(operator),
        "baseline": float(baseline),
    }
    row: dict[str, Any] = {
        "instance_id": int(item.instance_id),
        "nu": float(item.nu),
        "operator_sensor_mse": float(sensor_mse),
    }
    for name, hat in estimates.items():
        absolute = abs(hat - float(item.nu))
        relative = absolute / float(item.nu)
        row[f"{name}_nu_hat"] = hat
        row[f"{name}_abs_error"] = absolute
        row[f"{name}_rel_error"] = relative
        row[f"{name}_failure"] = bool(hat <= 0.0 or relative > FAILURE_RELATIVE_ERROR)
    _require_finite_row(row)
    return row


def _record(
    settings: InverseDemoConfig,
    output_label: str,
    norm: FieldNorm,
    datasets: dict[str, Any],
    history: list[Any],
    selected: Any,
    parameter_count: int,
    instances: list[dict[str, Any]],
    grid: np.ndarray,
    wall_clock: float,
) -> dict[str, Any]:
    from pinnforge.training.manifest import environment

    if not math.isfinite(wall_clock) or wall_clock < 0.0:
        raise ValueError("wall clock is not finite")
    payload: dict[str, Any] = {
        "format": DEMO_FORMAT,
        "command": f"pinnforge demo burgers-inverse --output {output_label}",
        "output": output_label,
        "scope": {
            "illustration": (
                "Small CPU example of sparse viscosity recovery. "
                "Not a rerun of the Stage D sensors32_bursts study."
            ),
            "closed_form": (
                "Normal equation of the central residual on every saved frame "
                "at equispaced sensors. R = a - nu * b."
            ),
            "operator": (
                "Trained data-only FNO. The first window is lifted from the sensors. "
                "Later blocks are an autoregressive rollout. The grid point with the "
                "smallest sensor mean square is the estimate. Ties keep the smaller viscosity."
            ),
            "baseline": "Training-split mean viscosity, the same value for every test instance.",
            "failure_rule": "nu_hat <= 0 or rel_error > 0.5",
            "true_nu_used_in_estimator": False,
        },
        "config": settings.to_dict(),
        "normalization": norm.to_dict(),
        "n_train_windows": int(datasets["train"].n_windows()),
        "n_val_windows": int(datasets["val"].n_windows()),
        "n_test_windows": int(datasets["test"].n_windows()) if "test" in datasets else 0,
        "parameter_count": int(parameter_count),
        "selected_epoch": int(selected.epoch),
        "val_relative_l2": float(selected.val_relative_l2),
        "nu_grid_step": float(grid[1] - grid[0]),
        "wall_clock_seconds": float(wall_clock),
        "environment": environment(),
        "history": [
            {
                "epoch": int(row.epoch),
                "train_mse": float(row.train_mse),
                "val_relative_l2": float(row.val_relative_l2),
            }
            for row in history
        ],
        "instances": instances,
        "aggregates": {
            "closed_form": _aggregate(instances, "closed_form"),
            "operator": _aggregate(instances, "operator"),
            "baseline": _aggregate(instances, "baseline"),
        },
        "files": {
            "result_json": "result.json",
            "result_markdown": "result.md",
            "report_html": "index.html",
            "viscosity_scatter": "viscosity_scatter.png",
            "relative_error": "relative_error.png",
            "field_sensors": "field_sensors.png",
            "checkpoint": "checkpoint.pt",
        },
    }
    _require_finite_row(
        {
            "val_relative_l2": payload["val_relative_l2"],
            "wall_clock_seconds": payload["wall_clock_seconds"],
            "nu_grid_step": payload["nu_grid_step"],
        }
    )
    return payload


def _aggregate(instances: list[dict[str, Any]], name: str) -> dict[str, Any]:
    absolute = np.asarray([row[f"{name}_abs_error"] for row in instances], dtype=np.float64)
    relative = np.asarray([row[f"{name}_rel_error"] for row in instances], dtype=np.float64)
    return {
        "n": len(instances),
        "mean_abs_error": float(np.mean(absolute)),
        "mean_rel_error": float(np.mean(relative)),
        "max_rel_error": float(np.max(relative)),
        "n_failures": int(sum(1 for row in instances if row[f"{name}_failure"])),
    }


def _write_plots(
    output_dir: Path,
    test: list[Trajectory],
    instances: list[dict[str, Any]],
    settings: InverseDemoConfig,
) -> None:
    truth = [row["nu"] for row in instances]
    write_scatter_png(
        output_dir / "viscosity_scatter.png",
        truth,
        [
            ("closed", [row["closed_form_nu_hat"] for row in instances], _CLOSED),
            ("fno", [row["operator_nu_hat"] for row in instances], _OPERATOR),
        ],
        xlabel="true nu",
        ylabel="estimate",
    )
    write_bars_png(
        output_dir / "relative_error.png",
        ["closed", "fno", "base"],
        [
            _aggregate(instances, "closed_form")["mean_rel_error"],
            _aggregate(instances, "operator")["mean_rel_error"],
            _aggregate(instances, "baseline")["mean_rel_error"],
        ],
        [_CLOSED, _OPERATOR, _BASELINE],
        ylabel="rel error",
    )
    shown = test[0]
    stride = settings.n_space // settings.n_sensors
    from pinnforge.reference.numerical.solver import grid

    space = grid(settings.n_space)
    write_field_png(
        output_dir / "field_sensors.png",
        [float(value) for value in space],
        [
            ("t0", [float(value) for value in shown.u[0]], _CLOSED),
            ("tend", [float(value) for value in shown.u[-1]], _FIELD_LATER),
        ],
        [float(value) for value in space[::stride]],
        [float(value) for value in shown.u[0, ::stride]],
        xlabel="x",
        ylabel="u",
    )


def _write_reports(output_dir: Path, record: dict[str, Any]) -> None:
    (output_dir / "result.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    markdown = _markdown(record)
    (output_dir / "result.md").write_text(markdown, encoding="utf-8")
    (output_dir / "index.html").write_text(_html(record), encoding="utf-8")


def _summary(output_label: str, record: dict[str, Any]) -> str:
    lines = [
        "pinnforge demo burgers-inverse",
        f"output: {output_label}",
        f"seed: {record['config']['seed']}",
        f"epochs: {record['config']['epochs']}",
        f"selected_epoch: {record['selected_epoch']}",
        f"val_relative_l2: {format_result_number(record['val_relative_l2'])}",
        f"wall_clock_seconds: {format_result_number(record['wall_clock_seconds'])}",
        f"nu_grid_step: {format_result_number(record['nu_grid_step'])}",
        "",
        results_table_markdown(record).rstrip("\n"),
        "",
        "json: result.json",
        "markdown: result.md",
        "html: index.html",
        "plots: viscosity_scatter.png, relative_error.png, field_sensors.png",
        "",
    ]
    return "\n".join(lines)


def _markdown(record: dict[str, Any]) -> str:
    config = record["config"]
    scope = record["scope"]
    lines = [
        "# Burgers inverse demo",
        "",
        f"Command: `{record['command']}`",
        "",
        scope["illustration"],
        "",
        "## Results",
        "",
        results_table_markdown(record).rstrip("\n"),
        "",
        (
            f"Data-only FNO, width {config['width']}, modes {config['modes']}, "
            f"layers {config['layers']}, {config['epochs']} Adam epochs, seed {config['seed']}. "
            f"Selected epoch {record['selected_epoch']} by validation relative L2 "
            f"{format_result_number(record['val_relative_l2'])}. "
            f"Wall clock {format_result_number(record['wall_clock_seconds'])} seconds on CPU "
            f"with torch threads set to 1. Viscosity grid step "
            f"{format_result_number(record['nu_grid_step'])}."
        ),
        "",
        scope["closed_form"],
        "",
        scope["operator"],
        "",
        scope["baseline"],
        "",
        f"Failure rule: {scope['failure_rule']}.",
        "",
        "## Plots",
        "",
        "Blue points in `viscosity_scatter.png` are the closed form. Orange points are the FNO. The gray line is equality.",
        "",
        "`relative_error.png` is the mean relative error of the closed form, the FNO, and the training-mean constant. The axis is log10 when the bars span more than a factor of eight. The numbers on the bars are the relative errors.",
        "",
        "`field_sensors.png` is the first test trajectory at the first and last saved frames. Black dots are the sensors on the first frame.",
        "",
        "Per-instance rows are in `result.json`.",
        "",
    ]
    return "\n".join(lines)


def _html(record: dict[str, Any]) -> str:
    table_rows = []
    sensors = int(record["config"]["n_sensors"])
    labels = {
        "closed_form": f"closed form, {sensors} sensors",
        "operator": "FNO sensor match",
        "baseline": "training-mean constant",
    }
    for key, label in labels.items():
        item = record["aggregates"][key]
        table_rows.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td>{int(item['n'])}</td>"
            f"<td>{format_result_number(item['mean_abs_error'])}</td>"
            f"<td>{format_result_number(item['mean_rel_error'])}</td>"
            f"<td>{format_result_number(item['max_rel_error'])}</td>"
            f"<td>{int(item['n_failures'])}</td>"
            "</tr>"
        )
    body = "\n".join(table_rows)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PINNForge Burgers inverse demo</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; max-width: 52rem; color: #1c1c1c; }}
table {{ border-collapse: collapse; }}
td, th {{ border: 1px solid #ccc; padding: 0.3rem 0.6rem; text-align: right; }}
td:first-child, th:first-child {{ text-align: left; }}
img {{ max-width: 100%; height: auto; }}
</style>
</head>
<body>
<h1>Burgers inverse demo</h1>
<p>{escape(record["scope"]["illustration"])}</p>
<p>Command: <code>{escape(record["command"])}</code></p>
<table>
<thead>
<tr><th>estimator</th><th>n</th><th>mean abs error</th><th>mean rel error</th><th>max rel error</th><th>failures</th></tr>
</thead>
<tbody>
{body}
</tbody>
</table>
<p>Selected epoch {int(record["selected_epoch"])}, validation relative L2 {format_result_number(record["val_relative_l2"])}, wall clock {format_result_number(record["wall_clock_seconds"])} seconds.</p>
<h2>True viscosity against the estimate</h2>
<img src="viscosity_scatter.png" alt="True viscosity versus closed-form and FNO estimates">
<h2>Mean relative error</h2>
<img src="relative_error.png" alt="Mean relative error of the closed form, the FNO, and the training mean">
<h2>One test field and the sensors</h2>
<img src="field_sensors.png" alt="First test field at the first and last frames, with sensor markers">
</body>
</html>
"""


def _require_finite_row(row: dict[str, Any]) -> None:
    for key, value in row.items():
        if isinstance(value, bool) or isinstance(value, int):
            continue
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{key} is not finite")
