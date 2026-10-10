"""Small PNG charts with no plotting dependency.

The drawings are illustrations of numbers computed elsewhere. Axis ticks
use a 3x5 bitmap font. Series names are drawn with the same font, so the
caption set is the letters that font contains.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

# 3-wide by 5-tall glyphs. Each row is three bits, left to right.
_GLYPHS: dict[str, tuple[int, ...]] = {
    " ": (0, 0, 0, 0, 0),
    "0": (0b111, 0b101, 0b101, 0b101, 0b111),
    "1": (0b010, 0b110, 0b010, 0b010, 0b111),
    "2": (0b111, 0b001, 0b111, 0b100, 0b111),
    "3": (0b111, 0b001, 0b111, 0b001, 0b111),
    "4": (0b101, 0b101, 0b111, 0b001, 0b001),
    "5": (0b111, 0b100, 0b111, 0b001, 0b111),
    "6": (0b111, 0b100, 0b111, 0b101, 0b111),
    "7": (0b111, 0b001, 0b001, 0b010, 0b010),
    "8": (0b111, 0b101, 0b111, 0b101, 0b111),
    "9": (0b111, 0b101, 0b111, 0b001, 0b111),
    ".": (0b000, 0b000, 0b000, 0b000, 0b010),
    "-": (0b000, 0b000, 0b111, 0b000, 0b000),
    "+": (0b000, 0b010, 0b111, 0b010, 0b000),
    "e": (0b111, 0b100, 0b110, 0b100, 0b111),
    "a": (0b000, 0b110, 0b001, 0b111, 0b101),
    "b": (0b100, 0b100, 0b111, 0b101, 0b111),
    "c": (0b111, 0b100, 0b100, 0b100, 0b111),
    "d": (0b001, 0b001, 0b111, 0b101, 0b111),
    "f": (0b011, 0b100, 0b110, 0b100, 0b100),
    "i": (0b010, 0b000, 0b010, 0b010, 0b010),
    "l": (0b110, 0b010, 0b010, 0b010, 0b111),
    "m": (0b000, 0b111, 0b101, 0b101, 0b101),
    "n": (0b000, 0b110, 0b101, 0b101, 0b101),
    "o": (0b000, 0b111, 0b101, 0b101, 0b111),
    "r": (0b000, 0b110, 0b101, 0b100, 0b100),
    "s": (0b111, 0b100, 0b111, 0b001, 0b111),
    "t": (0b111, 0b010, 0b010, 0b010, 0b010),
    "u": (0b000, 0b101, 0b101, 0b101, 0b111),
    "v": (0b000, 0b101, 0b101, 0b101, 0b010),
    "x": (0b000, 0b101, 0b010, 0b101, 0b000),
}

_WHITE = (255, 255, 255)
_INK = (32, 32, 32)
_GRID = (230, 230, 230)
_AXIS = (40, 40, 40)


def write_png(path: Path, image: np.ndarray) -> None:
    """Write an RGB uint8 image as a PNG file."""

    values = np.asarray(image)
    if values.ndim != 3 or values.shape[2] != 3 or values.dtype != np.uint8:
        raise ValueError("image must have shape (height, width, 3) and dtype uint8")
    height, width = int(values.shape[0]), int(values.shape[1])
    raw = b"".join(b"\x00" + values[row].tobytes() for row in range(height))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_png_bytes(width, height, raw))


def write_scatter_png(
    path: Path,
    xs: list[float],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    *,
    xlabel: str,
    ylabel: str,
) -> None:
    """Scatter ``xs`` against each series, with a y = x guide line."""

    if len(xs) < 1 or len(series) < 1:
        raise ValueError("scatter needs at least one x and one series")
    ys = [value for _, points, _ in series for value in points]
    if any(len(points) != len(xs) for _, points, _ in series):
        raise ValueError("each series must have one y per x")
    low = min(min(xs), min(ys))
    high = max(max(xs), max(ys))
    if high == low:
        high = low + 1.0
    pad = 0.08 * (high - low)
    low -= pad
    high += pad
    image, box = _canvas()
    _axes(image, box, low, high, low, high, xlabel=xlabel, ylabel=ylabel, ylog=False)
    _guide(image, box, low, high, low, high, ylog=False)
    for index, (name, points, color) in enumerate(series):
        for x_value, y_value in zip(xs, points, strict=True):
            px, py = _map_point(x_value, y_value, box, low, high, low, high, ylog=False)
            _disc(image, px, py, 5, color)
        _legend_row(image, name, color, index)
    write_png(path, image)


def write_bars_png(
    path: Path,
    labels: list[str],
    values: list[float],
    colors: list[tuple[int, int, int]],
    *,
    ylabel: str,
) -> None:
    """Vertical bars. A wide positive range uses a log10 axis."""

    if not labels or len(labels) != len(values) or len(labels) != len(colors):
        raise ValueError("bar labels, values, and colors must be the same non-empty length")
    if any(value <= 0.0 for value in values):
        raise ValueError("bar values must be > 0")
    ylog = max(values) / min(values) > 8.0
    image, box = _canvas()
    left, top, width, height = box
    if ylog:
        y_low = min(np.log10(values)) - 0.15
        y_high = max(np.log10(values)) + 0.25
    else:
        y_low = 0.0
        y_high = max(values) * 1.25
    _axes(
        image,
        box,
        0.0,
        float(len(values)),
        y_low,
        y_high,
        xlabel="",
        ylabel=ylabel,
        ylog=ylog,
        x_ticks=False,
    )
    slot = width / len(values)
    for index, (label, value, color) in enumerate(zip(labels, values, colors, strict=True)):
        if ylog:
            y0 = _y_at(y_low, box, y_low, y_high, ylog=False)
            y1 = _y_at(float(np.log10(value)), box, y_low, y_high, ylog=False)
        else:
            y0 = _y_at(0.0, box, y_low, y_high, ylog=False)
            y1 = _y_at(value, box, y_low, y_high, ylog=False)
        x0 = int(left + index * slot + slot * 0.22)
        x1 = int(left + index * slot + slot * 0.78)
        _fill_rect(image, x0, int(y1), x1, int(y0), color)
        text = f"{value:.3f}"
        _text(image, x0, max(top + 2, int(y1) - 16), text, scale=2, color=_INK)
        _text(image, x0, top + height + 8, label, scale=2, color=_INK)
    write_png(path, image)


def write_field_png(
    path: Path,
    x: list[float],
    series: list[tuple[str, list[float], tuple[int, int, int]]],
    sensors_x: list[float],
    sensors_y: list[float],
    *,
    xlabel: str,
    ylabel: str,
) -> None:
    """Line chart of one or more fields, with sensor markers on the first series."""

    if len(x) < 2 or not series:
        raise ValueError("field plot needs an x axis and one series")
    ys = [value for _, points, _ in series for value in points]
    ys.extend(sensors_y)
    x_low, x_high = min(x), max(x)
    y_low, y_high = min(ys), max(ys)
    if x_high == x_low:
        x_high = x_low + 1.0
    if y_high == y_low:
        y_high = y_low + 1.0
    y_pad = 0.08 * (y_high - y_low)
    y_low -= y_pad
    y_high += y_pad
    image, box = _canvas()
    _axes(image, box, x_low, x_high, y_low, y_high, xlabel=xlabel, ylabel=ylabel, ylog=False)
    for index, (name, points, color) in enumerate(series):
        if len(points) != len(x):
            raise ValueError("field series must match x")
        coords = [
            _map_point(x_value, y_value, box, x_low, x_high, y_low, y_high, ylog=False)
            for x_value, y_value in zip(x, points, strict=True)
        ]
        for start, end in zip(coords, coords[1:], strict=False):
            _line(image, start[0], start[1], end[0], end[1], color, thickness=2)
        _legend_row(image, name, color, index)
    for x_value, y_value in zip(sensors_x, sensors_y, strict=True):
        px, py = _map_point(x_value, y_value, box, x_low, x_high, y_low, y_high, ylog=False)
        _disc(image, px, py, 3, (20, 20, 20))
    write_png(path, image)


def _png_bytes(width: int, height: int, raw: bytes) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def _canvas() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    image = np.full((400, 640, 3), 255, dtype=np.uint8)
    return image, (72, 28, 540, 300)


def _axes(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
    *,
    xlabel: str,
    ylabel: str,
    ylog: bool,
    x_ticks: bool = True,
) -> None:
    left, top, width, height = box
    _fill_rect(image, left, top, left + width, top + height, _WHITE)
    for tick in range(5):
        frac = tick / 4
        x = int(left + frac * width)
        y = int(top + frac * height)
        _line(image, x, top, x, top + height, _GRID, thickness=1)
        _line(image, left, y, left + width, y, _GRID, thickness=1)
        y_value = y_high - frac * (y_high - y_low)
        if ylog:
            y_value = 10.0**y_value
        if x_ticks:
            x_value = x_low + frac * (x_high - x_low)
            _text(image, x - 12, top + height + 6, _tick(x_value), scale=2, color=_INK)
        _text(image, 4, y - 4, _tick(y_value), scale=2, color=_INK)
    _line(image, left, top, left, top + height, _AXIS, thickness=1)
    _line(image, left, top + height, left + width, top + height, _AXIS, thickness=1)
    if xlabel:
        _text(image, left, top + height + 28, xlabel, scale=2, color=_INK)
    if ylabel:
        _text(image, 4, 8, ylabel, scale=2, color=_INK)


def _guide(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
    *,
    ylog: bool,
) -> None:
    start = _map_point(x_low, x_low, box, x_low, x_high, y_low, y_high, ylog=ylog)
    end = _map_point(x_high, x_high, box, x_low, x_high, y_low, y_high, ylog=ylog)
    _line(image, start[0], start[1], end[0], end[1], (170, 170, 170), thickness=1)


def _map_point(
    x_value: float,
    y_value: float,
    box: tuple[int, int, int, int],
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
    *,
    ylog: bool,
) -> tuple[int, int]:
    left, top, width, height = box
    px = left + (x_value - x_low) / (x_high - x_low) * width
    mapped = float(np.log10(y_value)) if ylog else y_value
    py = _y_at(mapped, box, y_low, y_high, ylog=False)
    return int(round(px)), int(round(py))


def _y_at(y_value: float, box: tuple[int, int, int, int], y_low: float, y_high: float, *, ylog: bool) -> float:
    _, top, _, height = box
    mapped = float(np.log10(y_value)) if ylog else y_value
    return top + (y_high - mapped) / (y_high - y_low) * height


def _tick(value: float) -> str:
    if abs(value) >= 100 or (abs(value) < 0.01 and value != 0.0):
        return f"{value:.1e}".replace("e+0", "e+").replace("e-0", "e-")
    text = f"{value:.2f}"
    return text


def _legend_row(image: np.ndarray, name: str, color: tuple[int, int, int], index: int) -> None:
    y = 36 + index * 16
    _fill_rect(image, 470, y, 490, y + 8, color)
    _text(image, 496, y - 2, name, scale=2, color=_INK)


def _text(
    image: np.ndarray,
    x: int,
    y: int,
    text: str,
    *,
    scale: int,
    color: tuple[int, int, int],
) -> None:
    cursor = x
    for char in text:
        glyph = _GLYPHS.get(char, _GLYPHS.get(char.lower()))
        if glyph is None:
            cursor += 4 * scale
            continue
        for row, bits in enumerate(glyph):
            for col in range(3):
                if bits & (1 << (2 - col)):
                    _fill_rect(
                        image,
                        cursor + col * scale,
                        y + row * scale,
                        cursor + (col + 1) * scale - 1,
                        y + (row + 1) * scale - 1,
                        color,
                    )
        cursor += 4 * scale


def _disc(image: np.ndarray, x: int, y: int, radius: int, color: tuple[int, int, int]) -> None:
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy <= radius * radius:
                _pixel(image, x + dx, y + dy, color)


def _line(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
    *,
    thickness: int,
) -> None:
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    for step in range(steps):
        fraction = step / max(steps - 1, 1)
        x = int(round(x0 + fraction * (x1 - x0)))
        y = int(round(y0 + fraction * (y1 - y0)))
        radius = max(0, thickness - 1)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                _pixel(image, x + dx, y + dy, color)


def _fill_rect(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    height, width = image.shape[0], image.shape[1]
    left, right = sorted((int(x0), int(x1)))
    top, bottom = sorted((int(y0), int(y1)))
    left = max(0, left)
    top = max(0, top)
    right = min(width - 1, right)
    bottom = min(height - 1, bottom)
    if right < left or bottom < top:
        return
    image[top : bottom + 1, left : right + 1] = color


def _pixel(image: np.ndarray, x: int, y: int, color: tuple[int, int, int]) -> None:
    height, width = image.shape[0], image.shape[1]
    if 0 <= x < width and 0 <= y < height:
        image[y, x] = color
