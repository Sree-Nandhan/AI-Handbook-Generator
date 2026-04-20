import os
import asyncio
import gradio as gr
from gradio.themes.utils import colors, sizes
from rag_engine import RAGEngine
from pdf_processor import extract_text_from_pdf, save_uploaded_file
from handbook_generator import HandbookGenerator
from config import OUTPUT_DIR

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
rag_engine = RAGEngine()
handbook_gen: HandbookGenerator | None = None


async def startup():
    global handbook_gen
    await rag_engine.initialize()
    handbook_gen = HandbookGenerator(rag_engine)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
THEME = gr.themes.Base(
    primary_hue=colors.indigo,
    secondary_hue=colors.purple,
    neutral_hue=colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    radius_size=sizes.radius_lg,
    spacing_size=sizes.spacing_md,
    text_size=sizes.text_md,
).set(
    body_background_fill="#f1f5f9",
    block_background_fill="white",
    block_border_width="0px",
    block_shadow="0 1px 3px rgba(0,0,0,0.08)",
    block_radius="14px",
    button_primary_background_fill="linear-gradient(135deg, #4f46e5, #7c3aed)",
    button_primary_background_fill_hover="linear-gradient(135deg, #4338ca, #6d28d9)",
    button_primary_text_color="white",
    button_primary_border_color="transparent",
    button_primary_shadow="0 4px 14px rgba(79,70,229,0.3)",
    button_primary_shadow_hover="0 6px 20px rgba(79,70,229,0.4)",
    input_background_fill="#f8fafc",
    input_border_color="#e2e8f0",
    input_border_color_focus="#6366f1",
    input_border_width="1.5px",
    input_shadow_focus="0 0 0 3px rgba(99,102,241,0.12)",
    input_radius="10px",
    input_placeholder_color="#94a3b8",
)

CSS = """
.gradio-container { max-width: 900px !important; margin: 0 auto !important; }
#hero { text-align: center; padding: 28px 0 8px; }
#hero h1 {
    font-size: 2rem; font-weight: 800; margin: 0;
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
#hero p { color: #64748b; margin: 6px 0 0; font-size: 0.95rem; }
.status {
    font-size: 0.88rem; padding: 8px 12px; border-radius: 8px;
    background: #f8fafc; border-left: 3px solid #6366f1; color: #334155;
}
#handbook-out { max-height: 500px; overflow-y: auto; line-height: 1.7; }
#footer { text-align: center; padding: 14px 0; color: #94a3b8; font-size: 0.78rem; }
button, input, textarea { transition: all 0.15s ease !important; }
"""


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------
async def handle_upload_and_index(files):
    if not files:
        return "Upload a PDF to get started."
    parts = []
    for f in files:
        path = save_uploaded_file(f)
        text = extract_text_from_pdf(path)
        if len(text) < 100:
            parts.append(f"**{os.path.basename(f)}** — too little text")
            continue
        await rag_engine.insert_document(text)
        parts.append(f"**{os.path.basename(f)}** — {len(text):,} chars indexed")
    return "**Ready!** " + " | ".join(parts)


async def handle_chat(msg, history):
    if not msg.strip():
        return "", history
    history = history or []
    history.append({"role": "user", "content": msg})
    try:
        resp = await rag_engine.query(msg)
        history.append({"role": "assistant", "content": resp})
    except Exception as e:
        history.append({"role": "assistant", "content": f"Error: {e}"})
    return "", history


async def handle_generate(topic):
    if not topic.strip():
        yield "Enter a topic.", "", gr.update(visible=False)
        return
    if not handbook_gen:
        yield "Loading...", "", gr.update(visible=False)
        return
    if not rag_engine._chunks:
        yield "Upload and index a PDF first.", "", gr.update(visible=False)
        return

    yield "**Planning** handbook structure...", "*Planning...*", gr.update(visible=False)

    try:
        # Phase 1: Plan
        broad_ctx = await rag_engine.query_for_handbook_context(topic)
        plan = await handbook_gen.generate_plan(topic, broad_ctx[:500])

        if not plan:
            yield "**Error:** Could not generate a plan. Try a different topic.", "", gr.update(visible=False)
            return

        total = len(plan)
        yield f"**Plan ready** — {total} sections, writing...", f"*{total} sections planned. Writing...*", gr.update(visible=False)

        # Phase 2: Write each paragraph with live progress
        paragraphs = []
        written_text = ""

        for i, step in enumerate(plan):
            # Retry up to 3 times per section with increasing delay
            para = None
            for attempt in range(3):
                try:
                    ctx = await rag_engine.query_for_handbook_context(step.main_point)
                    para = await handbook_gen.write_paragraph(topic, plan, written_text, step, ctx)
                    break
                except Exception:
                    if attempt < 2:
                        await asyncio.sleep(5)

            if not para:
                para = f"*[Section {step.number}: {step.main_point} — generation skipped due to rate limits]*"

            paragraphs.append(para)
            written_text += "\n\n" + para
            wc = len(written_text.split())

            yield (
                f"**Writing** section {i+1}/{total} — {wc:,} words so far...",
                written_text.strip(),
                gr.update(visible=False),
            )

            await asyncio.sleep(5)

        # Phase 3: Compile
        final = handbook_gen.compile_handbook(topic, paragraphs, plan)
        path = handbook_gen.save_handbook(final)
        wc = len(final.split())

        yield f"**Done!** {wc:,} words generated.", final, gr.update(value=path, visible=True)

    except Exception as e:
        yield f"**Error:** {e}", "", gr.update(visible=False)


# ---------------------------------------------------------------------------
# UI — simple, clean, 3 sections stacked vertically
# ---------------------------------------------------------------------------
def build_ui():
    with gr.Blocks(title="AI Handbook Generator") as app:

        # Header
        gr.HTML("""
        <div id="hero">
            <h1>AI Handbook Generator</h1>
            <p>Upload a research paper, ask questions, generate a 20,000+ word handbook</p>
        </div>
        """)

        # ── Section 1: Upload ──
        gr.Markdown("### Upload & Index")
        with gr.Row():
            file_upload = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                file_count="multiple",
                type="filepath",
                scale=3,
            )
            index_btn = gr.Button("Upload & Index", variant="primary", scale=1)
        index_status = gr.Markdown("Upload a PDF to get started.", elem_classes=["status"])

        # ── Section 2: Chat ──
        gr.Markdown("### Chat with your Document")
        chatbot = gr.Chatbot(height=350, show_label=False, placeholder="Index a PDF first, then ask questions...")
        with gr.Row():
            msg_input = gr.Textbox(placeholder="Ask a question...", show_label=False, scale=5, lines=1)
            send_btn = gr.Button("Send", variant="primary", scale=1)

        # ── Section 3: Generate ──
        gr.Markdown("### Generate Handbook")
        with gr.Row():
            topic_input = gr.Textbox(
                placeholder="e.g. Create a handbook on Retrieval-Augmented Generation...",
                show_label=False, scale=4, lines=1,
            )
            gen_btn = gr.Button("Generate", variant="primary", scale=1)
        progress = gr.Markdown("", elem_classes=["status"])
        handbook_out = gr.Markdown("", elem_id="handbook-out")
        download = gr.File(label="Download Handbook", visible=False)

        # Footer
        gr.HTML('<div id="footer">Built with Cerebras &bull; Sentence Transformers &bull; Gradio</div>')

        # ── Wiring ──
        index_btn.click(fn=handle_upload_and_index, inputs=[file_upload], outputs=[index_status])
        send_btn.click(fn=handle_chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
        msg_input.submit(fn=handle_chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
        gen_btn.click(fn=handle_generate, inputs=[topic_input], outputs=[progress, handbook_out, download])

    return app


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    asyncio.run(startup())
    app = build_ui()
    app.queue()
    app.launch(server_name="0.0.0.0", server_port=7860, css=CSS, theme=THEME, js="() => { document.querySelector('body').classList.remove('dark'); }")
