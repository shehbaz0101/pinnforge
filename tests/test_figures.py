"""Demo plots use Matplotlib's bundled DejaVu Sans."""

from __future__ import annotations

import struct
from pathlib import Path

from pinnforge.figures import (
    FIGURE_DPI,
    dejavu_sans_file,
    write_bars_png,
    write_field_png,
    write_scatter_png,
)


def _size(path: Path) -> tuple[int, int]:
    blob = path.read_bytes()
    assert blob.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", blob[16:24])
    assert len(blob) < 200_000
    return width, height


def test_font_file_is_matplotlib_dejavu_sans() -> None:
    path = dejavu_sans_file()
    assert path.is_file()
    assert "dejavusans" in path.name.lower().replace(" ", "")
    assert "mpl-data" in path.as_posix() or "matplotlib" in path.as_posix()


def test_plots_use_the_bumped_dpi_and_normal_case_labels(tmp_path: Path) -> None:
    scatter = tmp_path / "viscosity_scatter.png"
    bars = tmp_path / "relative_error.png"
    field = tmp_path / "field_sensors.png"
    write_scatter_png(
        scatter,
        [0.03, 0.05],
        [
            ("Closed", [0.031, 0.049], (31, 78, 121)),
            ("FNO", [0.03, 0.06], (184, 92, 56)),
        ],
        xlabel="True viscosity",
        ylabel="Estimate",
    )
    write_bars_png(
        bars,
        ["Closed", "FNO", "Baseline"],
        [0.004, 0.05, 0.26],
        [(31, 78, 121), (184, 92, 56), (90, 90, 90)],
        ylabel="Relative error",
    )
    write_field_png(
        field,
        [0.0, 0.5, 1.0],
        [
            ("t = 0", [0.0, 1.0, 0.0], (31, 78, 121)),
            ("t = end", [0.1, 0.4, 0.1], (47, 107, 79)),
        ],
        [0.0, 1.0],
        [0.0, 0.0],
        xlabel="x",
        ylabel="u",
    )
    expected = (int(_FIG_W), int(_FIG_H))
    for path in (scatter, bars, field):
        assert _size(path) == expected


_FIG_W = 6.4 * FIGURE_DPI
_FIG_H = 4.0 * FIGURE_DPI
