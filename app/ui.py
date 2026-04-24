"""
Gradio UI — Two-phase layout:
  Phase 1: Upload page (full screen, no chat)
  Phase 2: Chat page with sidebar (after indexing)
"""

import os
import gradio as gr
from app.rag_engine import RAGEngine
from app.handbook_generator import HandbookGenerator
from app.handlers import handle_upload, handle_chat_message

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_lg,
).set(
    body_background_fill="#f4f6fb",
    block_background_fill="white",
    block_border_width="0px",
    block_shadow="0 1px 4px rgba(0,0,0,0.06)",
    button_primary_background_fill="linear-gradient(135deg, #4f46e5, #7c3aed)",
    button_primary_background_fill_hover="linear-gradient(135deg, #4338ca, #6d28d9)",
    button_primary_text_color="white",
    button_primary_border_color="transparent",
    button_primary_shadow="0 4px 14px rgba(79,70,229,0.25)",
    button_primary_shadow_hover="0 6px 20px rgba(79,70,229,0.35)",
    input_background_fill="white",
    input_border_color="#e2e8f0",
    input_border_color_focus="#6366f1",
    input_border_width="1.5px",
    input_shadow_focus="0 0 0 3px rgba(99,102,241,0.1)",
)

CSS = """
.gradio-container { max-width: 100% !important; overflow: hidden !important; }
body { overflow: hidden !important; }

/* Hide Gradio's default progress bars (we use our own) */
.progress-bar, .wrap.default, .generating {
    display: none !important;
}
.progress-text {
    display: none !important;
}

/* ── Phase 1: Upload page ── */
#upload-page {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 85vh;
}
#upload-card {
    max-width: 800px;
    width: 90%;
    margin: 0 auto;
    padding: 40px;
}
#upload-hero {
    text-align: center;
    padding: 0 0 24px;
}
#upload-hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
#upload-hero p {
    color: #64748b;
    margin: 8px 0 0;
    font-size: 1rem;
}

/* Progress bar */
#progress-bar-container { margin: 12px 0 8px; }
#progress-bar-container .progress-track {
    width: 100%; height: 10px; background: #e2e8f0;
    border-radius: 999px; overflow: hidden;
}
#progress-bar-container .progress-fill {
    height: 100%;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    border-radius: 999px; transition: width 0.4s ease;
}
@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(79,70,229,0.3); }
    50% { box-shadow: 0 0 8px 2px rgba(79,70,229,0.3); }
}
#progress-bar-container .progress-fill.active {
    animation: pulse-glow 1.5s ease-in-out infinite;
}

/* ── Phase 2: Chat page ── */
#download-btn, #download-file {
    min-height: 48px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    border-radius: 12px !important;
    margin-top: 8px !important;
}
"""


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(rag_engine: RAGEngine, handbook_gen: HandbookGenerator) -> gr.Blocks:
    """Two-phase app: upload page → chat page with sidebar."""

    with gr.Blocks(title="PaperLens", theme=THEME, css=CSS, fill_height=True) as app:

        # ════════════════════════════════════════════
        # PHASE 1: Upload Page (visible by default)
        # ════════════════════════════════════════════
        with gr.Column(visible=True, elem_id="upload-page") as upload_page:
            with gr.Column(elem_id="upload-card"):
                gr.HTML(
                    '<div id="upload-hero">'
                    "<h1>PaperLens</h1>"
                    "<p>Upload research papers to get started</p>"
                    "</div>"
                )

                file_upload = gr.File(
                    label="Research Papers",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath",
                )

                with gr.Row():
                    index_btn = gr.Button(
                        "Index Documents", variant="primary", visible=False, scale=3,
                    )
                    cancel_btn = gr.Button(
                        "Cancel", variant="secondary", visible=False, scale=1,
                    )

                progress_bar = gr.HTML("", elem_id="progress-bar-container", visible=False)
                upload_status = gr.Markdown("")

        # ════════════════════════════════════════════
        # PHASE 2: Chat Page (hidden until indexed)
        # ════════════════════════════════════════════
        with gr.Column(visible=False) as chat_page:

            # Chat interface
            chatbot = gr.Chatbot(
                placeholder=(
                    "<h2 style='font-weight:700;color:#4f46e5;'>Ready to chat!</h2>"
                    "<p style='color:#64748b;'>Ask questions about your papers or type "
                    "'Create a handbook on [topic]'</p>"
                ),
                height="75vh",
            )

            import functools
            chat_fn = functools.partial(_chat_fn, rag_engine=rag_engine, handbook_gen=handbook_gen)

            gr.ChatInterface(
                fn=chat_fn,
                chatbot=chatbot,
                autofocus=True,
                save_history=False,
                title=None,
                fill_height=True,
            )

            # Download button — hidden, appears after handbook generation
            download_btn = gr.DownloadButton(
                label="Download Handbook PDF",
                visible=False,
                variant="primary",
                elem_id="download-btn",
            )

            # Timer polls every 2s to check if a handbook was generated
            timer = gr.Timer(value=2, active=True)

            def _check_handbook():
                from app import handlers
                path = handlers._last_handbook_path
                start = handlers._session_start_time
                if not path or not start or not os.path.exists(path):
                    return gr.DownloadButton(label="Download Handbook PDF", visible=False)
                if os.path.getmtime(path) < start:
                    return gr.DownloadButton(label="Download Handbook PDF", visible=False)
                size_kb = os.path.getsize(path) // 1024
                return gr.DownloadButton(
                    label=f"Download Handbook PDF ({size_kb}KB)",
                    value=path,
                    visible=True,
                    variant="primary",
                )

            timer.tick(fn=_check_handbook, outputs=[download_btn])



        # ════════════════════════════════════════════
        # WIRING
        # ════════════════════════════════════════════

        def _progress_html(frac: float) -> str:
            pct = max(0, min(100, int(frac * 100)))
            active = "active" if pct < 100 else ""
            return (
                f'<div class="progress-track">'
                f'<div class="progress-fill {active}" style="width:{pct}%"></div>'
                f'</div>'
            )

        # Show index button when files selected
        file_upload.change(
            fn=lambda files: gr.update(visible=bool(files)),
            inputs=[file_upload],
            outputs=[index_btn],
        )

        # Index and switch to chat page
        async def process_and_switch(files):
            if not files:
                yield gr.update(), "", gr.update(), gr.update(), gr.update(), gr.update()
                return
            async for status_text, frac in handle_upload(files, rag_engine):
                is_done = frac >= 1.0
                yield (
                    gr.update(value=_progress_html(frac), visible=not is_done),
                    status_text,
                    gr.update(visible=False) if is_done else gr.update(),  # upload_page
                    gr.update(visible=True) if is_done else gr.update(),   # chat_page
                    gr.update(visible=False) if is_done else gr.update(),  # cancel
                    [] if is_done else gr.update(),  # clear chatbot
                )

        def on_index_start():
            return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True, value=_progress_html(0))

        index_event = index_btn.click(
            fn=on_index_start,
            outputs=[index_btn, cancel_btn, progress_bar],
        ).then(
            fn=process_and_switch,
            inputs=[file_upload],
            outputs=[progress_bar, upload_status, upload_page, chat_page, cancel_btn, chatbot],
        )

        cancel_btn.click(fn=None, cancels=[index_event]).then(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "Indexing cancelled."),
            outputs=[index_btn, cancel_btn, progress_bar, upload_status],
        )

    return app


async def _chat_fn(message, history, *, rag_engine, handbook_gen):
    """Chat function for ChatInterface — yields response strings."""
    async for response, _dl in handle_chat_message(message, history, rag_engine, handbook_gen):
        yield response
