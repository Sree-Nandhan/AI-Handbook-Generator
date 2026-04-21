"""
Gradio UI — step-by-step flow:
1. User sees only PDF upload area
2. After upload + indexing, chat area appears
"""

import gradio as gr
from app.rag_engine import RAGEngine
from app.handbook_generator import HandbookGenerator
from app.handlers import handle_upload, handle_chat

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
#app-container { max-width: 900px; margin: 0 auto; }

#hero {
    text-align: center;
    padding: 28px 0 16px;
}
#hero h1 {
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
#hero p {
    color: #64748b;
    margin: 6px 0 0;
    font-size: 0.95rem;
}

#footer {
    text-align: center;
    padding: 16px 0 8px;
    color: #94a3b8;
    font-size: 0.78rem;
}

#send-btn, #upload-btn, #cancel-btn, #generate-btn {
    min-height: 42px !important;
    font-weight: 600 !important;
}

/* Ensure progress bar is always visible above other elements */
.progress-bar, .wrap, .progress-text {
    z-index: 1000 !important;
    position: relative !important;
}

/* Progress bar styling */
.uploading-note {
    text-align: center;
    padding: 12px;
    color: #4f46e5;
    font-weight: 500;
    font-size: 0.95rem;
    background: #eef2ff;
    border-radius: 8px;
    margin: 8px 0;
}
"""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(rag_engine: RAGEngine, handbook_gen: HandbookGenerator) -> gr.Blocks:
    """Create the Gradio app with two-phase flow."""

    with gr.Blocks(title="AI Handbook Generator") as app:

        with gr.Column(elem_id="app-container"):

            gr.HTML(
                '<div id="hero">'
                "<h1>AI Handbook Generator</h1>"
                "<p>Upload PDFs, ask questions, generate 20,000+ word handbooks</p>"
                "</div>"
            )

            # ── Phase 1: Upload (visible by default) ──
            with gr.Column(visible=True) as upload_section:
                file_upload = gr.File(
                    label="Upload PDF Documents",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath",
                )
                with gr.Row():
                    index_btn = gr.Button(
                        "Index Documents", variant="primary", elem_id="upload-btn",
                        scale=3, visible=False,
                    )
                    cancel_btn = gr.Button(
                        "Cancel", variant="secondary", elem_id="cancel-btn",
                        scale=1, visible=False,
                    )
                upload_status = gr.Markdown("")

            # ── Phase 2: Chat (hidden until upload) ──
            with gr.Column(visible=False) as chat_section:
                chatbot = gr.Chatbot(
                    height=420,
                    show_label=False,
                    avatar_images=(
                        None,
                        "https://api.dicebear.com/9.x/bottts-neutral/svg?seed=handbook&backgroundColor=6366f1",
                    ),
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask a question, or type 'Create a handbook on [topic]'...",
                        show_label=False,
                        scale=5,
                        lines=1,
                        submit_btn=True,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1, elem_id="send-btn")

                download = gr.File(label="Download Handbook", visible=False)

            gr.HTML(
                '<div id="footer">'
                "Built with Grok 4.1 &bull; LightRAG &bull; Supabase &bull; Gradio"
                "</div>"
            )

        # ── Wiring ──

        async def process_upload(files, progress=gr.Progress(track_tqdm=True)):
            """Upload files with progress, then show chat section."""
            if not files:
                return (
                    "Please select PDF files to upload.",
                    gr.update(),
                    gr.update(),
                    gr.update(),
                )

            status = await handle_upload(files, rag_engine, progress)

            return (
                status,
                gr.update(visible=False),  # hide upload
                gr.update(visible=True),   # show chat
                gr.update(visible=False),  # hide cancel
            )

        # Show Index button when files are selected
        file_upload.change(
            fn=lambda files: gr.update(visible=bool(files)),
            inputs=[file_upload],
            outputs=[index_btn],
        )

        # Index button hides, cancel appears
        def on_index_start():
            return gr.update(visible=False), gr.update(visible=True)

        index_event = index_btn.click(
            fn=on_index_start,
            outputs=[index_btn, cancel_btn],
        ).then(
            fn=process_upload,
            inputs=[file_upload],
            outputs=[upload_status, upload_section, chat_section, cancel_btn],
        )

        # Cancel stops indexing, restores Index button
        cancel_btn.click(
            fn=None,
            cancels=[index_event],
        ).then(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False), "Indexing cancelled."),
            outputs=[index_btn, cancel_btn, upload_status],
        )

        # Chat wiring
        async def on_chat(message, history):
            async for h, dl in handle_chat(message, history, rag_engine, handbook_gen):
                yield h, dl

        send_btn.click(
            fn=on_chat,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, download],
            show_progress="minimal",
        ).then(fn=lambda: "", outputs=[msg_input])

        msg_input.submit(
            fn=on_chat,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, download],
            show_progress="minimal",
        ).then(fn=lambda: "", outputs=[msg_input])

    return app
