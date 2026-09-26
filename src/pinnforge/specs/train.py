"""Training hyperparameters for one built-in equation.

The counts, widths, and weights are explicit so a checkpoint can rebuild
the same run. ``device`` is ``cpu`` only. Checkpoint and log paths are
relative strings; :func:`pinnforge.specs.paths.resolve_inside_cwd`
checks them when the loop writes. The training package re-exports this
module. Importing :mod:`pinnforge.training` still needs torch; importing
:mod:`pinnforge.specs` does not.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from pinnforge.sampling import SampleMethod, resolve_equation_id
from pinnforge.sampling.defaults import CLI_COUNT_DEFAULTS, default_spec

# Keep these aligned with ``pinnforge.models.ACTIVATIONS``. That tuple
# lives in a torch-backed module, so the config cannot import it.
ACTIVATION_NAMES: tuple[str, ...] = ("relu", "silu", "tanh")

ActivationName = Literal["relu", "silu", "tanh"]


def _reject_bool(value: object) -> object:
    if isinstance(value, bool):
        raise ValueError("must be an integer")
    return value


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("must be a finite number")
    return number


def _hidden_widths(value: object) -> tuple[int, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise ValueError("hidden_widths must be a sequence of positive integers")
    widths = tuple(value)
    if len(widths) < 1:
        raise ValueError("hidden_widths must contain at least one layer")
    for width in widths:
        if isinstance(width, bool) or not isinstance(width, int) or width < 1:
            raise ValueError("hidden width must be a positive integer")
    return widths


def _relative_directory(value: object) -> str:
    return _relative_text(value, suffix=None, label="checkpoint_dir")


def _relative_log(value: object) -> str:
    return _relative_text(value, suffix=".jsonl", label="log_path")


def _relative_text(value: object, *, suffix: str | None, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if value == "" or value != value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty relative path")
    raw = Path(value)
    if raw.is_absolute() or value.startswith("~"):
        raise ValueError(f"{label} must be a relative path")
    if raw.parts == () or raw == Path("."):
        raise ValueError(f"{label} must name a path under the working directory")
    if suffix is not None and raw.suffix.lower() != suffix:
        raise ValueError(f"{label} must end in {suffix}")
    return value


def _canonical_equation(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("equation_id must be a string")
    return resolve_equation_id(value)


Count = Annotated[int, BeforeValidator(_reject_bool)]
Finite = Annotated[float, BeforeValidator(_finite_number)]
HiddenWidths = Annotated[tuple[int, ...], BeforeValidator(_hidden_widths)]
RelativeDir = Annotated[str, BeforeValidator(_relative_directory)]
RelativeLog = Annotated[str, BeforeValidator(_relative_log)]
EquationId = Annotated[str, BeforeValidator(_canonical_equation)]


class TrainConfig(BaseModel):
    """One CPU training run on a built-in Day 1 equation.

    ``equation_id`` accepts the registry id or a CLI alias such as
    ``harmonic``. Omitted ``n_ic`` and ``n_bc`` are filled from the Day 2
    per-equation defaults (harmonic has no boundary points, Poisson has
    no initial condition). ``n_interior`` defaults to 32 so a laptop CPU
    can finish the demo quickly.

    ``hidden_widths`` is the MLP width list passed to ``mlp_from_spec``.
    ``epochs`` is the number of Adam steps. Metrics also record epoch 0,
    the loss of the initial weights, so a run of 50 epochs writes 51
    lines. ``w_pde``, ``w_ic``, and ``w_bc`` scale the residual MSE and
    the two soft-penalty terms. At least one weight must be positive.

    ``checkpoint_dir`` and ``log_path`` are relative to the working
    directory. The log path must end in ``.jsonl``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    equation_id: EquationId = "harmonic_oscillator"
    n_interior: Count = Field(default=32, ge=1)
    n_ic: Count = Field(default=16, ge=0)
    n_bc: Count = Field(default=0, ge=0)
    hidden_widths: HiddenWidths = (16, 16)
    activation: ActivationName = "tanh"
    epochs: Count = Field(default=50, ge=1)
    lr: Finite = Field(default=1e-3, gt=0)
    w_pde: Finite = Field(default=1.0, ge=0)
    w_ic: Finite = Field(default=1.0, ge=0)
    w_bc: Finite = Field(default=1.0, ge=0)
    seed: Count = Field(default=0, ge=0)
    device: Literal["cpu"] = "cpu"
    method: SampleMethod = "uniform"
    checkpoint_dir: RelativeDir = "checkpoints"
    log_path: RelativeLog = "metrics.jsonl"

    @model_validator(mode="before")
    @classmethod
    def _fill_condition_counts(cls, data: object) -> object:
        """Use the Day 2 count defaults when ``n_ic`` or ``n_bc`` is omitted.

        Keys that are present, including an explicit zero, are left
        alone. A checkpoint therefore reloads the counts it stored.
        """

        if not isinstance(data, dict):
            return data
        raw = dict(data)
        equation = raw.get("equation_id", "harmonic_oscillator")
        if not isinstance(equation, str):
            return raw
        try:
            equation_id = resolve_equation_id(equation)
        except ValueError:
            return raw
        defaults = CLI_COUNT_DEFAULTS[equation_id]
        raw.setdefault("n_ic", defaults["n_ic"])
        raw.setdefault("n_bc", defaults["n_bc"])
        return raw

    @model_validator(mode="after")
    def _matches_builtin_spec(self) -> Self:
        if self.w_pde == 0 and self.w_ic == 0 and self.w_bc == 0:
            raise ValueError("at least one of w_pde, w_ic, and w_bc must be positive")
        spec = default_spec(self.equation_id)
        if self.equation_id == "poisson_toy" and self.n_ic != 0:
            raise ValueError("poisson_toy has no initial condition; n_ic must be 0")
        boundaries = getattr(spec, "boundary_conditions", ())
        if self.n_bc > 0 and len(boundaries) == 0:
            raise ValueError(f"{self.equation_id} has no boundary conditions; n_bc must be 0")
        return self
