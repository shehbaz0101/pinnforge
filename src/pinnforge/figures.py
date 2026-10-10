"""PNG charts for the Burgers inverse demo.

Text is Matplotlib's bundled DejaVu Sans at :data:`FIGURE_DPI`. The previous
renderer was a 3x5 bitmap whose glyphs mixed uppercase and lowercase shapes,
so labels such as ``closed``, ``base``, and ``rel error`` drew as jagged
mixed-case pixels on a fresh install.
"""

from __future__ import annotations

from pathlib import Path

FIGURE_DPI = 120
_FIGSIZE = (6.4, 4.0)
_FONT_FILE: Path | None = None
_BACKEND = False


def dejavu_sans_file() -> Path:
    """Return the DejaVu Sans file shipped with Matplotlib.

    The family is selected by file name, so a missing system font cannot
    substitute a different face.

    Raises:
        RuntimeError: Matplotlib did not resolve its DejaVu Sans file.
    """

    global _FONT_FILE
    if _FONT_FILE is not None:
        return _FONT_FILE
    _pyplot()
    from matplotlib import font_manager

    found = Path(
        font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans"),
            fallback_to_default=False,
        )
    )
    compact = found.name.lower().replace(" ", "")
    if "dejavusans" not in compact:
        raise RuntimeError(f"expected Matplotlib's DejaVu Sans, found {found}")
    _FONT_FILE = found
    return found


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
    fig, ax = _figure()
    font = _font(11)
    ax.plot([low, high], [low, high], color=(0.67, 0.67, 0.67), linewidth=1.0, zorder=1)
    for name, points, color in series:
        ax.scatter(
            xs,
            points,
            s=42,
            color=_rgb(color),
            label=name,
            zorder=2,
            linewidths=0,
        )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel(xlabel, fontproperties=font)
    ax.set_ylabel(ylabel, fontproperties=font)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.legend(frameon=False, prop=_font(11), loc="upper left")
    _apply_font(ax, font)
    _save(fig, path)


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
    fig, ax = _figure()
    font = _font(11)
    positions = list(range(len(values)))
    ax.bar(
        positions,
        values,
        width=0.62,
        color=[_rgb(color) for color in colors],
        zorder=2,
    )
    if ylog:
        ax.set_yscale("log")
        ax.set_ylim(min(values) / 2.2, max(values) * 2.8)
    else:
        ax.set_ylim(0.0, max(values) * 1.28)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontproperties=font)
    ax.set_ylabel(ylabel, fontproperties=font)
    for position, value in zip(positions, values, strict=True):
        ax.text(
            position,
            value,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontproperties=_font(10),
            color=(0.13, 0.13, 0.13),
            zorder=3,
        )
    _apply_font(ax, font)
    _save(fig, path)


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
    if len(sensors_x) != len(sensors_y):
        raise ValueError("sensor coordinates must be paired")
    for _name, points, _color in series:
        if len(points) != len(x):
            raise ValueError("field series must match x")
    fig, ax = _figure()
    font = _font(11)
    for name, points, color in series:
        ax.plot(x, points, color=_rgb(color), linewidth=1.8, label=name, zorder=2)
    if sensors_x:
        ax.scatter(
            sensors_x,
            sensors_y,
            s=16,
            c="black",
            linewidths=0,
            zorder=3,
        )
    ax.set_xlabel(xlabel, fontproperties=font)
    ax.set_ylabel(ylabel, fontproperties=font)
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    ax.legend(frameon=False, prop=_font(11), loc="upper right")
    _apply_font(ax, font)
    _save(fig, path)


def _pyplot():
    global _BACKEND
    import matplotlib

    if not _BACKEND:
        matplotlib.use("Agg")
        _BACKEND = True
    import matplotlib.pyplot as plt

    return plt


def _font(size: float):
    from matplotlib import font_manager

    return font_manager.FontProperties(fname=str(dejavu_sans_file()), size=size)


def _figure():
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=_FIGSIZE, dpi=FIGURE_DPI)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(True, color="#e6e6e6", linewidth=0.6, zorder=0)
    for spine in ax.spines.values():
        spine.set_color("#282828")
        spine.set_linewidth(0.8)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.16, top=0.94)
    return fig, ax


def _apply_font(ax, font) -> None:
    ax.xaxis.label.set_fontproperties(font)
    ax.yaxis.label.set_fontproperties(font)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontproperties(font)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontproperties(font)


def _rgb(color: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(channel / 255.0 for channel in color)


def _save(fig, path: Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=FIGURE_DPI, facecolor="white")
    _pyplot().close(fig)
