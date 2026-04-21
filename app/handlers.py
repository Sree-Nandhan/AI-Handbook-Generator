"""
Chat event handlers — upload, Q&A, and handbook generation.
"""

import os
import asyncio
import logging
import gradio as gr
from app.rag_engine import RAGEngine
from app.pdf_processor import extract_text, save_uploaded_file
from app.handbook_generator import HandbookGenerator

logger = logging.getLogger("handbook.handlers")

HANDBOOK_KEYWORDS = [
    "handbook", "generate", "create a handbook",
    "write a handbook", "20,000", "20000",
]


# ---------------------------------------------------------------------------
# Phase 1: Upload & index PDFs (with progress)
# ---------------------------------------------------------------------------

async def handle_upload(files: list, rag_engine: RAGEngine, progress=gr.Progress()):
    """Process uploaded PDFs and index into LightRAG knowledge graph."""
    if not files:
        return "Please select PDF files to upload."

    total = len(files)
    results = []

    for idx, f in enumerate(files):
        fname = os.path.basename(f)

        # Step 1: Extract text
        progress((idx * 3) / (total * 3), desc=f"Extracting text from {fname}...")
        try:
            path = save_uploaded_file(f)
            text = extract_text(path)
            if len(text) < 100:
                results.append(f"**{fname}** — too little text extracted")
                continue
        except Exception as e:
            logger.error(f"Failed to extract {fname}: {e}")
            results.append(f"**{fname}** — extraction error: {e}")
            continue

        # Step 2: Index into knowledge graph
        progress((idx * 3 + 1) / (total * 3), desc=f"Indexing {fname} into knowledge graph...")
        try:
            await rag_engine.insert(text)
            results.append(f"**{fname}** — {len(text):,} chars indexed")
        except Exception as e:
            logger.error(f"Failed to index {fname}: {e}")
            results.append(f"**{fname}** — indexing error: {e}")
            continue

        progress((idx * 3 + 2) / (total * 3), desc=f"Finished {fname}")

    progress(1.0, desc="Done!")

    return (
        "**Documents indexed!**\n\n"
        + "\n".join(f"- {r}" for r in results)
        + "\n\nYou can now ask questions or request a handbook."
    )


# ---------------------------------------------------------------------------
# Phase 2: Chat — Q&A or handbook generation
# ---------------------------------------------------------------------------

async def handle_chat(
    message: str,
    history: list,
    rag_engine: RAGEngine,
    handbook_gen: HandbookGenerator,
):
    """Handle chat messages — Q&A or handbook generation."""
    history = history or []
    user_text = message.strip() if message else ""

    if not user_text:
        yield history, gr.update()
        return

    history.append({"role": "user", "content": user_text})
    yield history, gr.update()

    # Detect handbook request
    is_handbook = any(kw in user_text.lower() for kw in HANDBOOK_KEYWORDS)

    if is_handbook and not rag_engine.has_documents:
        history.append({
            "role": "assistant",
            "content": "Please upload PDF documents first.",
        })
        yield history, gr.update()
        return

    if is_handbook:
        async for h, dl in _generate_handbook(user_text, history, rag_engine, handbook_gen):
            yield h, dl
        return

    # Regular Q&A
    try:
        response = await rag_engine.query(user_text)
        history.append({"role": "assistant", "content": response})
    except Exception as e:
        logger.error(f"Query failed: {e}")
        history.append({"role": "assistant", "content": f"Error: {e}"})
    yield history, gr.update()


# ---------------------------------------------------------------------------
# Handbook generation pipeline
# ---------------------------------------------------------------------------

async def _generate_handbook(
    topic: str,
    history: list,
    rag_engine: RAGEngine,
    handbook_gen: HandbookGenerator,
):
    """Run AgentWrite/LongWriter pipeline with live progress."""
    history.append({"role": "assistant", "content": "**Planning** handbook structure..."})
    yield history, gr.update()

    try:
        # Phase 1: Plan
        logger.info(f"Handbook started: {topic[:80]}")
        broad_ctx = await rag_engine.get_context(topic)
        plan = await handbook_gen.create_plan(topic, broad_ctx[:2000])

        if not plan:
            history[-1]["content"] = "Could not generate a plan. Try rephrasing."
            yield history, gr.update()
            return

        total = len(plan)
        history[-1]["content"] = f"**Plan ready** — {total} sections. Writing..."
        yield history, gr.update()

        # Phase 2: Write each section
        paragraphs = []
        written_text = ""

        for i, section in enumerate(plan):
            para = None
            for attempt in range(3):
                try:
                    ctx = await rag_engine.get_context(section.main_point)
                    para = await handbook_gen.write_section(
                        topic, plan, written_text, section, ctx
                    )
                    break
                except Exception as e:
                    logger.warning(f"Section {section.number} attempt {attempt+1}: {e}")
                    if attempt < 2:
                        await asyncio.sleep(3)

            if not para:
                para = f"*[Section {section.number}: {section.main_point} — skipped]*"
                logger.error(f"Section {section.number} skipped")

            paragraphs.append(para)
            written_text += "\n\n" + para
            wc = len(written_text.split())

            history[-1]["content"] = (
                f"**Writing** section {i+1}/{total} — {wc:,} words so far..."
            )
            yield history, gr.update()
            await asyncio.sleep(1)

        # Phase 3: Compile and save
        final = handbook_gen.compile(topic, paragraphs, plan)
        path = handbook_gen.save(final, topic)
        wc = len(final.split())
        fname = os.path.basename(path)

        history[-1]["content"] = (
            f"**Handbook complete!** {wc:,} words generated.\n\n"
            f"**{fname}** is ready for download.\n\n"
            f"---\n\n{final[:2000]}...\n\n---\n\n"
            f"*Showing first 2,000 chars. Full {wc:,}-word handbook available via download button below.*"
        )
        logger.info(f"Handbook complete: {wc:,} words -> {path}")
        yield history, gr.update(value=path, visible=True)

    except Exception as e:
        logger.error(f"Handbook failed: {e}")
        history[-1]["content"] = f"**Error:** {e}"
        yield history, gr.update()
