"""Gradio entry point for the Render free web service.

The process binds ``0.0.0.0`` and ``PORT`` (7860 when unset). The page opens
on saved prediction plots and the Stage D chart. Torch stays unimported
until the live button. This file does not mount the localhost FastAPI service.
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
    landing_markdown,
    live_button_label,
    live_section_markdown,
    precomputed_images,
    prediction_figure,
    release_heap,
    release_other_figures,
    run_quick_pinn,
    server_bind,
    solve_status,
    stage_d_headline,
    stage_d_markdown,
    viscosity_image_path,
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


def _gallery_image(caption: str, path: Path) -> gr.Image:
    return gr.Image(
        value=str(path),
        label=caption,
        interactive=False,
        show_label=True,
        height=320,
        type="filepath",
    )


def build_demo() -> gr.Blocks:
    """Assemble the page. The gallery is static. Training waits for the button."""

    choices = equation_choices()
    headline = stage_d_headline()
    images = precomputed_images()
    with gr.Blocks(title="PINNForge") as demo:
        gr.Markdown(HEADER_MARKDOWN)
        gr.Markdown(landing_markdown(headline))
        with gr.Row():
            for caption, path in images:
                with gr.Column():
                    _gallery_image(caption, path)
        gr.Markdown(stage_d_markdown(headline))
        gr.Image(
            value=str(viscosity_image_path()),
            label="Precomputed Stage D viscosity chart",
            interactive=False,
            show_label=True,
            height=320,
            type="filepath",
        )
        gr.Markdown(live_section_markdown())
        with gr.Row():
            with gr.Column():
                equation = gr.Radio(
                    choices=choices,
                    value="harmonic",
                    label="Equation",
                    info="Optional. One torch thread, sized for a 512 MB instance.",
                )
                blurb = gr.Markdown(equation_blurb("harmonic"))
                run = gr.Button(live_button_label("harmonic"), variant="secondary")
                status = gr.Markdown(
                    "The live plot stays empty until you click. The curves above are already the result."
                )
            with gr.Column():
                plot = gr.Plot(label="Live prediction and reference")
        equation.change(equation_blurb, inputs=equation, outputs=blurb)
        equation.change(live_button_label, inputs=equation, outputs=run)
        run.click(_run, inputs=equation, outputs=[plot, status], concurrency_limit=1)
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
