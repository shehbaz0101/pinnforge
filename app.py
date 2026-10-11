"""Gradio entry point for a Hugging Face Space.

The Space SDK runs this file. It trains only the three small problems in
:mod:`pinnforge.live_demo`. It does not mount the localhost FastAPI service.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gradio as gr  # noqa: E402

from pinnforge.live_demo import (  # noqa: E402
    HEADER_MARKDOWN,
    equation_blurb,
    equation_choices,
    prediction_figure,
    prepare_runtime,
    run_quick_pinn,
    solve_status,
    stage_d_headline,
    stage_d_markdown,
    viscosity_figure,
)


def _run(equation: str) -> tuple[object, str]:
    try:
        solved = run_quick_pinn(equation)
    except Exception as exc:
        return None, f"The solve stopped: {type(exc).__name__}: {exc}"
    return prediction_figure(solved), solve_status(solved)


def _open_page() -> tuple[object, str]:
    return viscosity_figure(), stage_d_markdown()


def build_demo() -> gr.Blocks:
    """Assemble the page. Training happens only on the Run button."""

    choices = equation_choices()
    headline = stage_d_headline()
    with gr.Blocks(title="PINNForge") as demo:
        gr.Markdown(HEADER_MARKDOWN)
        with gr.Row():
            with gr.Column(scale=4):
                equation = gr.Radio(
                    choices=choices,
                    value="harmonic",
                    label="Equation",
                    info="The fit uses a fixed small budget so it finishes in seconds on a free CPU.",
                )
                blurb = gr.Markdown(equation_blurb("harmonic"))
                run = gr.Button("Run PINN", variant="primary")
                status = gr.Markdown(
                    "Click **Run PINN**. Training happens on this CPU when you click, "
                    "and the plot below is that run."
                )
            with gr.Column(scale=5):
                caption = gr.Markdown(stage_d_markdown(headline))
                viscosity = gr.Plot(
                    value=viscosity_figure(headline),
                    label="Precomputed Stage D result",
                )
        plot = gr.Plot(label="Prediction and reference")
        equation.change(equation_blurb, inputs=equation, outputs=blurb)
        run.click(_run, inputs=equation, outputs=[plot, status], concurrency_limit=1)
        demo.load(_open_page, outputs=[viscosity, caption])
    return demo


def main() -> None:
    prepare_runtime()
    demo = build_demo()
    demo.queue(max_size=8, default_concurrency_limit=1)
    demo.launch()


if __name__ == "__main__":
    main()
