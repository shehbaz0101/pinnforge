"""JSON manifest for one training run.

The file sits next to ``checkpoint.pt``. It records the train,
validation, and test RNG identities and the package versions that
produced the run. It does not store weights.
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np

from pinnforge import __version__
from pinnforge.ml_import import require_torch

torch = require_torch()

MANIFEST_FORMAT = "pinnforge.manifest.v1"


def environment() -> dict[str, str]:
    """Versions and the Python platform. No host name and no secrets."""

    return {
        "pinnforge": __version__,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def write_manifest(directory: Path, payload: dict[str, object]) -> Path:
    """Write ``manifest.json`` under ``directory``."""

    path = directory / "manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
