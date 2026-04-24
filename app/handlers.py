"""
Chat event handlers — upload, Q&A, and handbook generation.
"""

import os
import asyncio
import logging
import gradio as gr
from app.rag_engine import RAGEngine
from app.pdf_processor import extract_text, extract_title, extract_references, save_uploaded_file
from app.handbook_generator import HandbookGenerator

logger = logging.getLogger("handbook.handlers")

HANDBOOK_KEYWORDS = [
    "handbook",
    "create a handbook",
    "write a handbook",
    "generate a handbook",
    "generate handbook",
    "make a handbook",
]

REGENERATE_KEYWORDS = ["same handbook", "regenerate", "same topic", "regenerate handbook"]

HISTORY_KEYWORDS = ["show handbooks", "list handbooks", "previous handbooks", "my handbooks"]

_last_handbook_topic = ""


# ---------------------------------------------------------------------------
# Phase 1: Upload & index PDFs (with progress)
# ---------------------------------------------------------------------------

async def handle_upload(files: list, rag_engine: RAGEngine):
    """Process uploaded PDFs and index into LightRAG knowledge graph.

    This is an async generator that yields (status_markdown, progress_fraction)
    tuples so the UI can show live progress.
    """
    if not files:
        yield "Please select PDF files to upload.", 0
        return

    total = len(files)
    results = []

    for idx, f in enumerate(files):
        fname = os.path.basename(f)
        file_num = f"[{idx + 1}/{total}]"

        # ── Extract text (fast) ──
        frac = idx / total
        yield f"**{file_num} Extracting** `{fname}`...", frac
        try:
            path = save_uploaded_file(f)
            paper_title = extract_title(path)
            rag_engine.add_source_filename(paper_title if paper_title else fname)
            text = extract_text(path)
            if len(text) < 100:
                results.append(f"**{fname}** — too little text extracted")
                continue
        except Exception as e:
            logger.error(f"Failed to extract {fname}: {e}")
            results.append(f"**{fname}** — extraction error: {e}")
            continue

        # ── Extract references ──
        import pdfplumber
        try:
            with pdfplumber.open(path) as pdf:
                raw = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
                num_pages = len(pdf.pages)
            refs = extract_references(raw)
            if refs:
                rag_engine.add_references(fname, refs)
                logger.info(f"Extracted references from {fname}: {len(refs)} chars")
        except Exception as e:
            logger.warning(f"Reference extraction failed for {fname}: {e}")
            num_pages = len(text) // 3000  # rough estimate

        # Estimate indexing time based on text size (~5s per 10K chars)
        est_seconds = max(10, int(len(text) / 10000 * 5))
        est_min = est_seconds // 60
        est_str = f"~{est_min}m {est_seconds % 60}s" if est_min > 0 else f"~{est_seconds}s"

        # ── Index (slow — show estimated time) ──
        frac = (idx + 0.4) / total
        yield f"**{file_num} Indexing** `{fname}` ({num_pages} pages, {len(text):,} chars) — est. {est_str}", frac
        try:
            await rag_engine.insert(text)
            results.append(f"**{fname}** — {len(text):,} chars indexed")
        except Exception as e:
            logger.error(f"Failed to index {fname}: {e}")
            results.append(f"**{fname}** — indexing error: {e}")
            continue

        frac = (idx + 1) / total
        yield f"**{file_num} Done** — `{fname}` indexed successfully!", frac

    yield (
        "**All documents indexed!**\n\n"
        + "\n".join(f"- {r}" for r in results)
        + "\n\nYou can now ask questions or request a handbook."
    ), 1.0


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
    global _last_handbook_topic
    history = history or []
    user_text = message.strip() if message else ""

    if not user_text:
        yield history, gr.update()
        return

    history.append({"role": "user", "content": user_text})
    yield history, gr.update()

    lower_text = user_text.lower()

    # Detect handbook history/listing request
    is_history = any(kw in lower_text for kw in HISTORY_KEYWORDS)
    if is_history:
        handbooks = HandbookGenerator.list_handbooks()
        if handbooks:
            lines = ["**Previously generated handbooks:**\n"]
            for hb in handbooks:
                lines.append(
                    f"- **{hb['topic']}** — {hb['word_count']:,} words — {hb['timestamp']}"
                )
            content = "\n".join(lines)
        else:
            content = "No handbooks have been generated yet."
        history.append({"role": "assistant", "content": content})
        yield history, gr.update()
        return

    # Detect regenerate / same topic request
    is_regenerate = any(kw in lower_text for kw in REGENERATE_KEYWORDS)
    if is_regenerate and _last_handbook_topic:
        topic = _last_handbook_topic
        if not rag_engine.has_documents:
            history.append({
                "role": "assistant",
                "content": "Please upload PDF documents first.",
            })
            yield history, gr.update()
            return
        history.append({
            "role": "assistant",
            "content": f"Regenerating handbook with the same topic: **{topic}**",
        })
        yield history, gr.update()
        async for h, dl in _generate_handbook(topic, history, rag_engine, handbook_gen):
            yield h, dl
        return
    elif is_regenerate and not _last_handbook_topic:
        history.append({
            "role": "assistant",
            "content": "No previous handbook topic found. Please specify a topic for the handbook.",
        })
        yield history, gr.update()
        return

    # Detect handbook request
    is_handbook = any(kw in lower_text for kw in HANDBOOK_KEYWORDS)

    if is_handbook and not rag_engine.has_documents:
        history.append({
            "role": "assistant",
            "content": "Please upload PDF documents first.",
        })
        yield history, gr.update()
        return

    if is_handbook:
        # If user just says "handbook" without a topic, and we have a previous one, ask
        bare_handbook = lower_text.strip() in ["handbook", "create handbook", "generate handbook", "make handbook"]
        if bare_handbook and _last_handbook_topic:
            history.append({
                "role": "assistant",
                "content": (
                    f"I previously generated a handbook on **{_last_handbook_topic}**.\n\n"
                    f"Would you like to:\n"
                    f"1. **Specify a new topic** — e.g., 'Create a handbook on Machine Learning'\n"
                    f"2. **Regenerate the same** — just say 'same' or 'regenerate'\n"
                ),
            })
            yield history, gr.update()
            return
        async for h, dl in _generate_handbook(user_text, history, rag_engine, handbook_gen):
            yield h, dl
        return

    # Regular Q&A
    import random
    loading_msgs = [
        "Diving into the research papers...",
        "Analyzing the knowledge graph...",
        "Cross-referencing findings...",
        "Consulting the research corpus...",
        "Mining insights from your papers...",
        "Connecting the dots across documents...",
        "Synthesizing information...",
        "Searching through the literature...",
    ]
    history.append({"role": "assistant", "content": f"*{random.choice(loading_msgs)}*"})
    yield history, gr.update()

    try:
        response = await rag_engine.query(user_text)
        history[-1]["content"] = response
    except Exception as e:
        logger.error(f"Query failed: {e}")
        history[-1]["content"] = f"Error: {e}"
    yield history, gr.update()


# ---------------------------------------------------------------------------
# Handbook generation — parallel batch pipeline
# ---------------------------------------------------------------------------

async def _generate_handbook(
    topic: str,
    history: list,
    rag_engine: RAGEngine,
    handbook_gen: HandbookGenerator,
):
    """Run parallel AgentWrite/LongWriter pipeline with live progress."""
    global _last_handbook_topic
    history.append({"role": "assistant", "content": "**Planning** handbook structure..."})
    yield history, gr.update()

    # Progress state shared with callback
    status = {"stage": "planning", "total": 0, "done": 0}

    def on_progress(stage, total, done):
        status["stage"] = stage
        status["total"] = total
        status["done"] = done

    try:
        # Run generation with progress callback
        gen_task = asyncio.create_task(
            handbook_gen.generate_parallel(topic, rag_engine, on_progress)
        )

        # Poll progress while generation runs
        while not gen_task.done():
            stage = status["stage"]
            total = status["total"]
            done = status["done"]

            if stage == "planning":
                msg = "**Planning** handbook structure..."
            elif stage == "fetching_context":
                msg = f"**Fetching** RAG context for {total} sections..."
            elif stage == "writing":
                msg = f"**Writing** sections — {done}/{total} complete..."
            else:
                msg = "**Processing...**"

            history[-1]["content"] = msg
            yield history, gr.update()
            await asyncio.sleep(0.5)

        # Get result
        final, plan = await gen_task
        pdf_path, md_path = handbook_gen.save(final, topic)
        _last_handbook_topic = topic
        wc = len(final.split())
        fname = os.path.basename(pdf_path)

        history[-1]["content"] = (
            f"**Handbook complete!** {wc:,} words generated.\n\n"
            f"**{fname}** is ready for download.\n\n"
            f"---\n\n{final[:2000]}...\n\n---\n\n"
            f"*Showing first 2,000 chars. Full {wc:,}-word handbook available via download button below.*"
        )
        logger.info(f"Handbook complete: {wc:,} words -> {pdf_path}")
        yield history, gr.DownloadButton(
            label=f"Download Handbook ({wc:,} words)",
            value=pdf_path,
            visible=True,
        )

    except Exception as e:
        logger.error(f"Handbook failed: {e}")
        history[-1]["content"] = f"**Error:** {e}"
        yield history, gr.update()
