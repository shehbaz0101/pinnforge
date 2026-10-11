"""Gradio entry point for the Render free web service.

The process binds ``0.0.0.0`` and ``PORT`` (7860 when unset). It trains only
the three small problems in :mod:`pinnforge.live_demo`. Torch stays
unimported until the first Run click. This file does not mount the
localhost FastAPI service.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Thread pools and glibc arenas are fixed before torch is imported on a click.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import gradio as gr  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402

from pinnforge.live_demo import (  # noqa: E402
    HEADER_MARKDOWN,
    HEALTH_PATH,
    equation_blurb,
    equation_choices,
    health_body,
    prediction_figure,
    release_heap,
    release_other_figures,
    run_quick_pinn,
    server_bind,
    solve_status,
    stage_d_headline,
    stage_d_markdown,
    viscosity_figure,
)


def _run(equation: str) -> tuple[object, str]:
    figure: object = None
    try:
        solved = run_quick_pinn(equation)
        figure = prediction_figure(solved)
        release_other_figures(figure)
        status = solve_status(solved)
    except Exception as exc:
        return None, f"The solve stopped: {type(exc).__name__}: {exc}"
    finally:
        release_heap()
    return figure, status


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
                    info="The fit uses a fixed small budget sized for a 512 MB instance.",
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


async def _health(_request: object) -> JSONResponse:
    return JSONResponse(health_body())


def app_kwargs() -> dict[str, list[Route]]:
    """Routes passed into Gradio's FastAPI app. ``/health`` skips the solver."""

    return {"routes": [Route(HEALTH_PATH, _health, methods=["GET"])]}


def main() -> None:
    host, port = server_bind()
    demo = build_demo()
    demo.queue(max_size=2, default_concurrency_limit=1)
    demo.launch(
        server_name=host,
        server_port=port,
        max_threads=1,
        app_kwargs=app_kwargs(),
        show_error=True,
    )


if __name__ == "__main__":
    main()
