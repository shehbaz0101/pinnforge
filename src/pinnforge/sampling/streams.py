"""Independent NumPy streams for training, validation, and test draws.

``numpy.random.default_rng(seed)`` is one stream. A training draw of
``n`` uniform points and a longer evaluation draw with the same integer
seed share a prefix: the training points are the first rows of the
evaluation points. These helpers mix the seed with a stream name so the
three draws stay reproducible and are not prefixes of each other, even
when the integer seed and the count are the same.

The stream identity is the list passed to ``numpy.random.SeedSequence``.
Callers log that list. It is not a secret.
"""

from __future__ import annotations

import hashlib

import numpy as np

STREAM_NAMES: tuple[str, ...] = ("train", "validation", "test")

# Fixed mix-in so a named stream is not ``default_rng(seed)``.
_SALT = 0x50494E4E
_TAGS = {"train": 1, "validation": 2, "test": 3}


def stream_identity(seed: int, stream: str) -> dict[str, object]:
    """Return the logged identity of one named stream.

    ``seed_sequence`` is the entropy ``stream_generator`` passes to
    ``numpy.random.SeedSequence``. Equal seeds with different stream
    names produce different sequences.
    """

    sequence = _sequence(seed, stream)
    return {"name": stream, "seed": seed, "seed_sequence": sequence}


def stream_generator(seed: int, stream: str) -> np.random.Generator:
    """Return the Generator for ``stream`` at ``seed``.

    This is not ``numpy.random.default_rng(seed)``. Use that only when
    the caller wants the historical single stream, such as
    ``pinnforge sample``.
    """

    return np.random.default_rng(np.random.SeedSequence(_sequence(seed, stream)))


def points_sha256(points: np.ndarray) -> str:
    """Hex digest of a float64 copy of ``points``.

    The digest identifies a draw in a manifest. It is not a hash of the
    seed, so two streams can be compared by the points they produced.
    """

    array = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    if not np.isfinite(array).all():
        raise ValueError("points must be finite")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _sequence(seed: int, stream: str) -> list[int]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    try:
        tag = _TAGS[stream]
    except KeyError:
        known = ", ".join(STREAM_NAMES)
        raise ValueError(f"unknown rng stream {stream!r}; known: {known}") from None
    return [seed, tag, _SALT]
