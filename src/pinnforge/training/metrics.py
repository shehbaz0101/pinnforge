"""One JSON object per epoch.

Epoch 0 is the initial network. Later epochs are the loss after that
many Adam steps. Component fields are unweighted mean squares. ``loss``
is ``w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc``. ``lr`` is the
Adam learning rate, which this loop does not schedule.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

METRICS_FORMAT = "pinnforge.metrics.v1"

_FIELDS = ("epoch", "loss", "loss_pde", "loss_ic", "loss_bc", "lr")


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Loss components and learning rate for one epoch."""

    epoch: int
    loss: float
    loss_pde: float
    loss_ic: float
    loss_bc: float
    lr: float

    def as_dict(self) -> dict[str, object]:
        """JSON-ready mapping, including the format tag."""

        return {
            "format": METRICS_FORMAT,
            "epoch": self.epoch,
            "loss": self.loss,
            "loss_pde": self.loss_pde,
            "loss_ic": self.loss_ic,
            "loss_bc": self.loss_bc,
            "lr": self.lr,
        }


def write_metrics(path: Path, history: tuple[EpochMetrics, ...] | list[EpochMetrics]) -> None:
    """Replace ``path`` with one JSON line per epoch.

    Parent directories are created. The file is the whole history, not
    an append to a previous run.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row.as_dict(), allow_nan=False) for row in history]
    text = "\n".join(lines)
    if text:
        text += "\n"
    destination.write_text(text, encoding="utf-8")


def append_metrics(path: Path, row: EpochMetrics) -> None:
    """Append one epoch, creating parent directories if needed."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row.as_dict(), allow_nan=False) + "\n")


def read_metrics(path: Path) -> tuple[EpochMetrics, ...]:
    """Read a file written by :func:`append_metrics` or :func:`write_metrics`.

    Epoch numbers must be ``0, 1, 2, ...`` with no gaps.

    Raises:
        ValueError: the file is empty, a line is not the metrics format,
            or the epoch sequence is broken.
    """

    text = Path(path).read_text(encoding="utf-8")
    rows: list[EpochMetrics] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line.strip() == "":
            continue
        payload = json.loads(line)
        rows.append(_row(payload, line_number))
    if not rows:
        raise ValueError("metrics file is empty")
    for index, row in enumerate(rows):
        if row.epoch != index:
            raise ValueError(f"metrics epochs must be 0, 1, 2, ...; line {index + 1} has {row.epoch}")
    return tuple(rows)


def _row(payload: object, line_number: int) -> EpochMetrics:
    if not isinstance(payload, dict):
        raise ValueError(f"metrics line {line_number} must be an object")
    if payload.get("format") != METRICS_FORMAT:
        raise ValueError(f"metrics line {line_number} has format {payload.get('format')!r}")
    missing = [key for key in _FIELDS if key not in payload]
    if missing:
        names = ", ".join(missing)
        raise ValueError(f"metrics line {line_number} is missing {names}")
    epoch = payload["epoch"]
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"metrics line {line_number} epoch must be a non-negative integer")
    values = {key: _finite(payload[key], line_number=line_number, label=key) for key in _FIELDS if key != "epoch"}
    return EpochMetrics(epoch=epoch, **values)


def _finite(value: object, *, line_number: int, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"metrics line {line_number} {label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"metrics line {line_number} {label} must be a finite number")
    return number
