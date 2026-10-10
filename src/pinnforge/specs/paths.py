"""Paths that stay inside the sandbox root.

The root is the process working directory, unless ``PINNFORGE_DATA_ROOT``
is set or a caller passes ``root``. ``pinnforge sample``, ``train``,
``eval``, ``run``, and ``serve`` accept ``--data-root``, which sets that
variable for the command. Config files, checkpoints, metrics, eval JSON,
and sample output must be relative to the root. Absolute paths and ``~``
are rejected for those inputs. ``..`` and symlinks are resolved before the
containment check, so a path that leaves the root is rejected even when
the text looks nested. :func:`resolve_cli_output` is the local exception
for ``pinnforge demo burgers-inverse --output``: a relative directory
still stays inside the root, and an absolute directory is accepted.
The HTTP API does not call that function. Importing this module does not
import torch.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT_ENV = "PINNFORGE_DATA_ROOT"


class PathSandboxError(ValueError):
    """A user path resolved outside the sandbox root."""


def configure_sandbox_root(value: str | Path) -> Path:
    """Resolve ``value`` to an existing directory and return that path.

    A relative root is resolved from the working directory. ``~`` is
    expanded here because the root is chosen by the operator, not read
    from a user file path.

    Raises:
        PathSandboxError: ``value`` is empty, contains a null byte, or
            is not an existing directory.
    """

    if isinstance(value, Path):
        raw = value.as_posix().strip()
    elif isinstance(value, str):
        raw = value.strip()
    else:
        raise PathSandboxError("data root must be a path")
    if raw == "" or "\x00" in raw:
        raise PathSandboxError("data root must be a non-empty path")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        resolved = candidate.resolve()
    except (OSError, ValueError) as exc:
        raise PathSandboxError(f"data root could not be resolved: {raw}") from exc
    if not resolved.is_dir():
        raise PathSandboxError(f"data root does not exist or is not a directory: {resolved}")
    return resolved


def sandbox_root() -> Path:
    """Return the allowed root for this process.

    An unset or blank ``PINNFORGE_DATA_ROOT`` means the working directory.
    """

    raw = os.environ.get(DATA_ROOT_ENV)
    if raw is None or not raw.strip():
        return Path.cwd().resolve()
    return configure_sandbox_root(raw)


def resolve_inside_cwd(
    path: str | Path,
    *,
    suffix: str | None = None,
    root: Path | None = None,
    label: str = "path",
) -> Path:
    """Resolve ``path`` and require it to stay inside the sandbox root.

    ``suffix``, when set, is compared case-insensitively (``.jsonl`` or
    ``.pt``). The path must be relative, must not be the root itself, and
    must still sit inside the root after ``..`` and symlinks are resolved.
    The target does not have to exist yet.

    ``root`` defaults to :func:`sandbox_root`. ``label`` is the noun used
    in the error (``path``, ``output path``, ``checkpoint``).

    Raises:
        PathSandboxError: the path is absolute, expands ``~``, escapes
            the root, or has the wrong suffix.
    """

    if isinstance(path, Path):
        text = path.as_posix()
    elif isinstance(path, str):
        text = path
    else:
        raise PathSandboxError(f"{label} must be a relative path")
    if text == "" or text != text.strip() or "\x00" in text:
        raise PathSandboxError(f"{label} must be a non-empty relative path")
    if text.startswith("~"):
        raise PathSandboxError(f"{label} must be a relative path")
    raw = Path(text)
    if raw.is_absolute():
        raise PathSandboxError(f"{label} must be a relative path")
    if raw.parts == () or raw == Path("."):
        raise PathSandboxError(
            f"{label} must name a file or directory under the working directory"
        )
    base = sandbox_root() if root is None else Path(root).resolve()
    try:
        resolved = (base / raw).resolve()
    except (OSError, ValueError) as exc:
        raise PathSandboxError(f"{label} could not be resolved: {text}") from exc
    if resolved == base or not _is_inside(resolved, base):
        message = f"{label} must stay inside the working directory"
        cwd = Path.cwd().resolve()
        if base != cwd:
            message = f"{message} (sandbox root {base})"
        raise PathSandboxError(message)
    if suffix is not None and resolved.suffix.lower() != suffix:
        raise PathSandboxError(f"{label} must end in {suffix}")
    return resolved


def resolve_cli_output(path: str | Path, *, label: str = "output") -> tuple[Path, str]:
    """Resolve a directory from the local ``--output`` flag.

    A relative path must stay inside the sandbox root, the same rule as
    :func:`resolve_inside_cwd`. An absolute path, or ``~`` expanded to
    one, is accepted because the CLI is a local process. It is resolved,
    it must name a directory other than the filesystem root or the home
    directory, and an existing path must already be a directory. The
    directory does not have to exist yet.

    The returned label is the relative text the user passed, or the
    resolved POSIX path when the input was absolute. Request bodies for
    the HTTP API stay on :func:`resolve_inside_cwd` and remain relative.

    Raises:
        PathSandboxError: the path is empty, escapes the sandbox root,
            or is an absolute location this command will not write.
    """

    if isinstance(path, Path):
        text = path.as_posix()
    elif isinstance(path, str):
        text = path
    else:
        raise PathSandboxError(f"{label} must be a path")
    if text == "" or text != text.strip() or "\x00" in text:
        raise PathSandboxError(f"{label} must be a non-empty path")
    expanded = Path(text).expanduser()
    if expanded.is_absolute():
        resolved = _resolve_absolute_directory(expanded, label=label)
        return resolved, resolved.as_posix()
    resolved = resolve_inside_cwd(text, label=label)
    return resolved, Path(text).as_posix()


def _resolve_absolute_directory(path: Path, *, label: str) -> Path:
    try:
        resolved = path.resolve()
    except (OSError, ValueError) as exc:
        raise PathSandboxError(f"{label} could not be resolved: {path.as_posix()}") from exc
    if resolved == Path(resolved.anchor) or resolved == Path.home():
        raise PathSandboxError(f"{label} must name a directory under a parent, not {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise PathSandboxError(f"{label} is not a directory: {resolved}")
    return resolved


def relative_to_sandbox(path: str | Path) -> str:
    """Return ``path`` relative to :func:`sandbox_root`, in POSIX form.

    Raises:
        PathSandboxError: the resolved path is outside the sandbox root.
    """

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(sandbox_root()).as_posix()
    except ValueError as exc:
        raise PathSandboxError("path must stay inside the working directory") from exc


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
