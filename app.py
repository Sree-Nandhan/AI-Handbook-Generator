import os
import asyncio
import gradio as gr
from rag_engine import RAGEngine
from pdf_processor import extract_text_from_pdf, save_uploaded_file
from handbook_generator import HandbookGenerator, HandbookProgress
from config import OUTPUT_DIR

# Global state
rag_engine = RAGEngine()
handbook_gen: HandbookGenerator | None = None


async def startup():
    global handbook_gen
    await rag_engine.initialize()
    handbook_gen = HandbookGenerator(rag_engine)


async def handle_file_upload(files, current_files):
    if files is None:
        return "No files uploaded yet.", current_files or []
    new_files = []
    for f in files:
        saved_path = save_uploaded_file(f)
        new_files.append(saved_path)
    all_files = (current_files or []) + new_files
    display = "\n".join([f"- {os.path.basename(f)}" for f in all_files])
    return f"**Uploaded files:**\n{display}", all_files


async def handle_index(file_paths):
    if not file_paths:
        return "No files to index. Please upload PDFs first."

    status_parts = []
    for path in file_paths:
        try:
            text = extract_text_from_pdf(path)
            char_count = len(text)
            if char_count < 100:
                status_parts.append(
                    f"**Warning:** {os.path.basename(path)} has very little text ({char_count} chars). "
                    f"It may be image-based. Consider using an OCR tool."
                )
                continue
            await rag_engine.insert_document(text)
            status_parts.append(f"Indexed **{os.path.basename(path)}** ({char_count:,} chars)")
        except Exception as e:
            status_parts.append(f"Error indexing {os.path.basename(path)}: {str(e)}")

    return "**Indexing complete:**\n\n" + "\n\n".join(status_parts)


async def handle_chat(message, history):
    if not message.strip():
        return "", history

    history = history or []
    history.append({"role": "user", "content": message})

    try:
        response = await rag_engine.query(message, mode="hybrid")
        history.append({"role": "assistant", "content": response})
    except Exception as e:
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})

    return "", history


async def handle_generate_handbook(topic, history):
    if not topic.strip():
        yield (
            "Please enter a handbook topic.",
            "",
            gr.update(visible=False),
            history or [],
        )
        return

    if handbook_gen is None:
        yield (
            "System not initialized. Please wait and try again.",
            "",
            gr.update(visible=False),
            history or [],
        )
        return

    history = history or []
    history.append({"role": "user", "content": f"Generate handbook: {topic}"})

    # Streaming progress updates
    progress_info = {"text": "Starting..."}

    def on_progress(p: HandbookProgress):
        progress_info["text"] = (
            f"**{p.status.upper()}** | "
            f"Section {p.completed_paragraphs}/{p.total_paragraphs} | "
            f"{p.total_words_written:,} words written | "
            f"Current: {p.current_section}"
        )

    yield (
        "Starting handbook generation...",
        "",
        gr.update(visible=False),
        history,
    )

    try:
        handbook_md = await handbook_gen.generate_handbook(topic, on_progress)

        output_path = handbook_gen.save_handbook(handbook_md)
        word_count = len(handbook_md.split())

        history.append({
            "role": "assistant",
            "content": f"Handbook generated successfully! **{word_count:,} words**. You can download it below.",
        })

        yield (
            f"**Complete!** {word_count:,} words generated.",
            handbook_md,
            gr.update(value=output_path, visible=True),
            history,
        )

    except Exception as e:
        history.append({"role": "assistant", "content": f"Error generating handbook: {str(e)}"})
        yield (
            f"**Error:** {str(e)}",
            "",
            gr.update(visible=False),
            history,
        )


def build_ui():
    with gr.Blocks(
        title="AI Handbook Generator",
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 0; }
        .subtitle { text-align: center; color: #666; margin-top: 0; }
        """,
    ) as app:

        uploaded_files_state = gr.State([])

        gr.Markdown("# AI Handbook Generator", elem_classes=["main-title"])
        gr.Markdown(
            "Upload PDFs, ask questions about your documents, and generate comprehensive 20,000+ word handbooks.",
            elem_classes=["subtitle"],
        )

        with gr.Row():
            # --- LEFT SIDEBAR ---
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("### Documents")
                file_upload = gr.File(
                    label="Upload PDFs",
                    file_types=[".pdf"],
                    file_count="multiple",
                    type="filepath",
                )
                uploaded_list = gr.Markdown("No files uploaded yet.")
                index_btn = gr.Button("Index Documents", variant="primary")
                index_status = gr.Markdown("")

            # --- MAIN AREA ---
            with gr.Column(scale=3):
                gr.Markdown("### Chat with your Documents")
                chatbot = gr.Chatbot(label="Chat", height=350, type="messages")

                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="Ask about your documents...",
                        show_label=False,
                        scale=4,
                    )
                    send_btn = gr.Button("Send", scale=1)

                gr.Markdown("---")
                gr.Markdown("### Generate Handbook")

                handbook_topic = gr.Textbox(
                    label="Handbook Topic / Instructions",
                    placeholder='e.g., "Create a comprehensive handbook on Retrieval-Augmented Generation covering architectures, implementations, and best practices"',
                    lines=3,
                )

                generate_btn = gr.Button("Generate Handbook (20,000+ words)", variant="primary")
                progress_text = gr.Markdown("")

                with gr.Accordion("Generated Handbook", open=False):
                    handbook_output = gr.Markdown("", label="Handbook Content")

                download_btn = gr.File(label="Download Handbook", visible=False)

        # --- EVENT HANDLERS ---
        file_upload.change(
            fn=handle_file_upload,
            inputs=[file_upload, uploaded_files_state],
            outputs=[uploaded_list, uploaded_files_state],
        )

        index_btn.click(
            fn=handle_index,
            inputs=[uploaded_files_state],
            outputs=[index_status],
        )

        send_btn.click(
            fn=handle_chat,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )
        msg_input.submit(
            fn=handle_chat,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )

        generate_btn.click(
            fn=handle_generate_handbook,
            inputs=[handbook_topic, chatbot],
            outputs=[progress_text, handbook_output, download_btn, chatbot],
        )

    return app


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    asyncio.run(startup())
    app = build_ui()
    app.queue()
    app.launch(server_name="0.0.0.0", server_port=7860)
