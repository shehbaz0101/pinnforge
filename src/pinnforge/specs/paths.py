"""Relative paths that stay inside the working directory.

Checkpoints and ``metrics.jsonl`` are local files. Absolute paths are
rejected. ``..`` and symlinks are resolved before the containment check,
matching :func:`pinnforge.sampling.resolve_output_path`. The training
package re-exports this helper. Importing :mod:`pinnforge.specs` does
not import torch.
"""

from __future__ import annotations

from pathlib import Path


def resolve_inside_cwd(path: str | Path, *, suffix: str | None = None) -> Path:
    """Resolve ``path`` and require it to stay inside the working directory.

    ``suffix``, when set, is compared case-insensitively (``.jsonl`` or
    ``.pt``). The path must not be absolute and must not be the working
    directory itself.

    Raises:
        ValueError: the path is absolute, escapes the working directory,
            or has the wrong suffix.
    """

    raw = Path(path)
    if raw.is_absolute():
        raise ValueError("path must be a relative path")
    if raw.parts == () or raw == Path("."):
        raise ValueError("path must name a file or directory under the working directory")
    cwd = Path.cwd().resolve()
    resolved = (cwd / raw).resolve()
    if resolved == cwd or cwd not in resolved.parents:
        raise ValueError("path must stay inside the working directory")
    if suffix is not None and resolved.suffix.lower() != suffix:
        raise ValueError(f"path must end in {suffix}")
    return resolved
