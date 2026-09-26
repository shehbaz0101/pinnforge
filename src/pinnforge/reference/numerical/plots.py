"""Small SVG line charts for the convergence study.

The figures are written with the standard library so a convergence run
does not need Matplotlib. They are illustrations of the saved JSON, not
a separate numerical result.
"""

from __future__ import annotations

import math
from pathlib import Path


def write_chart(
    path: Path,
    series: list[dict[str, object]],
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    xlog: bool = False,
    ylog: bool = False,
) -> None:
    """Write one SVG chart. Each series has ``name``, ``x``, and ``y`` lists."""

    prepared = [_prepare(item, xlog=xlog, ylog=ylog) for item in series]
    width, height = 720, 440
    left, right, top, bottom = 78, 24, 46, 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [value for item in prepared for value in item["px"]]
    ys = [value for item in prepared for value in item["py"]]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    colors = ("#1f4e79", "#b85c38", "#2f6b4f", "#6b4c9a")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="28" font-family="sans-serif" font-size="16">{_escape(title)}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        x = left + fraction * plot_w
        y = top + fraction * plot_h
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#e6e6e6"/>')
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e6e6e6"/>')
        x_label = _tick_label(x_min + fraction * (x_max - x_min), log=xlog)
        y_label = _tick_label(y_max - fraction * (y_max - y_min), log=ylog)
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 18}" text-anchor="middle" font-family="sans-serif" font-size="11">{x_label}</text>'
        )
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="11">{y_label}</text>'
        )
    parts.append(
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222222"/>'
    )
    for index, item in enumerate(prepared):
        color = colors[index % len(colors)]
        points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(item["px"], item["py"], strict=True))
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        for x, y in zip(item["px"], item["py"], strict=True):
            parts.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3.5" fill="{color}"/>')
        legend_y = top + 16 + index * 18
        parts.append(f'<line x1="{left + plot_w - 180}" y1="{legend_y}" x2="{left + plot_w - 150}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>')
        parts.append(
            f'<text x="{left + plot_w - 144}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{_escape(str(item["name"]))}</text>'
        )
    parts.append(
        f'<text x="{left + plot_w / 2}" y="{height - 12}" text-anchor="middle" font-family="sans-serif" font-size="13">{_escape(xlabel)}</text>'
    )
    parts.append(
        f'<text x="18" y="{top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 18 {top + plot_h / 2})" font-family="sans-serif" font-size="13">{_escape(ylabel)}</text>'
    )
    parts.append("</svg>")
    Path(path).write_text("\n".join(parts) + "\n", encoding="utf-8")


def _prepare(item: dict[str, object], *, xlog: bool, ylog: bool) -> dict[str, object]:
    raw_x = [float(value) for value in item["x"]]  # type: ignore[index]
    raw_y = [float(value) for value in item["y"]]  # type: ignore[index]
    if len(raw_x) != len(raw_y) or not raw_x:
        raise ValueError("chart series need matching non-empty x and y")
    px = [_log10(value) if xlog else value for value in raw_x]
    py = [_log10(value) if ylog else value for value in raw_y]
    return {"name": item["name"], "px": px, "py": py}


def _log10(value: float) -> float:
    if value <= 0.0 or not math.isfinite(value):
        raise ValueError("log axes require finite positive values")
    return math.log10(value)


def _tick_label(value: float, *, log: bool) -> str:
    shown = 10**value if log else value
    if shown == 0:
        return "0"
    absolute = abs(shown)
    if absolute >= 1000 or absolute < 0.01:
        return f"{shown:.1e}"
    return f"{shown:.3g}"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
