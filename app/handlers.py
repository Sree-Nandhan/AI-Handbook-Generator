"""
Chat event handlers — upload, Q&A, and handbook generation.
Designed for gr.ChatInterface (yields response strings, not history lists).
"""

import os
import asyncio
import logging
import random
import gradio as gr
from app.rag_engine import RAGEngine
from app.pdf_processor import extract_text, extract_title, extract_references, save_uploaded_file
from app.handbook_generator import HandbookGenerator

logger = logging.getLogger("handbook.handlers")

HANDBOOK_KEYWORDS = [
    "handbook", "create a handbook", "write a handbook",
    "generate a handbook", "generate handbook", "make a handbook",
]
REGENERATE_KEYWORDS = ["same handbook", "regenerate", "same topic", "regenerate handbook"]
HISTORY_KEYWORDS = ["show handbooks", "list handbooks", "previous handbooks", "my handbooks"]

LOADING_MSGS = [
    "Diving into the research papers...",
    "Analyzing the knowledge graph...",
    "Cross-referencing findings...",
    "Consulting the research corpus...",
    "Mining insights from your papers...",
    "Connecting the dots across documents...",
    "Synthesizing information...",
    "Searching through the literature...",
]

_last_handbook_topic = ""
_last_handbook_path = ""


# ---------------------------------------------------------------------------
# Phase 1: Upload & index PDFs
# ---------------------------------------------------------------------------

async def handle_upload(files: list, rag_engine: RAGEngine):
    """Process uploaded PDFs and index into LightRAG knowledge graph.
    Yields (status_markdown, progress_fraction) tuples.
    """
    if not files:
        yield "Please select PDF files to upload.", 0
        return

    total = len(files)
    results = []

    for idx, f in enumerate(files):
        fname = os.path.basename(f)
        file_num = f"[{idx + 1}/{total}]"

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

        import pdfplumber
        try:
            with pdfplumber.open(path) as pdf:
                raw = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
                num_pages = len(pdf.pages)
            refs = extract_references(raw)
            if refs:
                rag_engine.add_references(fname, refs)
        except Exception as e:
            logger.warning(f"Reference extraction failed for {fname}: {e}")
            num_pages = len(text) // 3000

        # Estimate based on observed rates: ~1.7s per 1K chars, ~7s per page
        # Use whichever gives a higher estimate for safety
        import time as _time
        est_by_chars = int(len(text) / 1000 * 1.7)
        est_by_pages = num_pages * 7
        est_seconds = max(15, max(est_by_chars, est_by_pages))
        if est_seconds >= 60:
            est_str = f"~{est_seconds // 60}m {est_seconds % 60}s"
        else:
            est_str = f"~{est_seconds}s"

        frac = (idx + 0.4) / total
        yield f"**{file_num} Indexing** `{fname}` ({num_pages} pages, {len(text)//1000}K chars) — est. {est_str}", frac
        t_index_start = _time.time()
        try:
            await rag_engine.insert(text)
            t_index_end = _time.time()
            actual = int(t_index_end - t_index_start)
            results.append(f"**{fname}** — {len(text):,} chars indexed in {actual}s")
            logger.info(f"Indexed {fname}: {len(text):,} chars, {num_pages} pages in {actual}s")
        except Exception as e:
            logger.error(f"Failed to index {fname}: {e}")
            results.append(f"**{fname}** — indexing error: {e}")
            continue

        frac = (idx + 1) / total
        yield f"**{file_num} Done** — `{fname}` indexed!", frac

    yield (
        "**All documents indexed!**\n\n"
        + "\n".join(f"- {r}" for r in results)
        + "\n\nYou can now ask questions or request a handbook."
    ), 1.0


# ---------------------------------------------------------------------------
# Phase 2: Chat — for gr.ChatInterface (yields response strings)
# ---------------------------------------------------------------------------

async def handle_chat_message(
    message: str,
    history: list,
    rag_engine: RAGEngine,
    handbook_gen: HandbookGenerator,
):
    """Handle a single chat message. Yields response string chunks for streaming.
    Returns tuple (response, download_update) for additional_outputs.
    """
    global _last_handbook_topic, _last_handbook_path
    user_text = message.strip() if message else ""

    if not user_text:
        yield "", gr.update()
        return

    lower_text = user_text.lower()

    # ── Handbook history ──
    if any(kw in lower_text for kw in HISTORY_KEYWORDS):
        handbooks = HandbookGenerator.list_handbooks()
        if handbooks:
            lines = ["**Previously generated handbooks:**\n"]
            for hb in handbooks:
                lines.append(f"- **{hb['topic']}** — {hb['word_count']:,} words — {hb['timestamp']}")
            yield "\n".join(lines), gr.update()
        else:
            yield "No handbooks have been generated yet.", gr.update()
        return

    # ── Regenerate ──
    if any(kw in lower_text for kw in REGENERATE_KEYWORDS):
        if _last_handbook_topic:
            async for resp, dl in _generate_handbook(_last_handbook_topic, rag_engine, handbook_gen):
                yield resp, dl
        else:
            yield "No previous handbook topic found. Please specify a topic.", gr.update()
        return

    # ── Handbook generation ──
    if any(kw in lower_text for kw in HANDBOOK_KEYWORDS):
        bare = lower_text.strip() in ["handbook", "create handbook", "generate handbook", "make handbook"]
        if bare and _last_handbook_topic:
            yield (
                f"I previously generated a handbook on **{_last_handbook_topic}**.\n\n"
                f"Would you like to:\n"
                f"1. **Specify a new topic** — e.g., 'Create a handbook on Machine Learning'\n"
                f"2. **Regenerate the same** — just say 'same' or 'regenerate'"
            ), gr.update()
            return
        async for resp, dl in _generate_handbook(user_text, rag_engine, handbook_gen):
            yield resp, dl
        return

    # ── Simple greetings — don't hit RAG ──
    greetings = ["hello", "hi", "hey", "yo", "sup", "howdy", "greetings", "good morning",
                 "good afternoon", "good evening", "thanks", "thank you", "ok", "okay",
                 "bye", "goodbye", "help"]
    if lower_text.strip().rstrip("!?.") in greetings:
        yield (
            "Hello! I'm your Research Master AI. Here's what I can do:\n\n"
            "- **Ask questions** about your uploaded papers\n"
            "- **Create a handbook** — e.g., 'Create a handbook on Machine Learning'\n"
            "- **Show handbooks** — view previously generated handbooks\n\n"
            "What would you like to know?"
        ), gr.update()
        return

    # ── Regular Q&A ──
    yield f"*{random.choice(LOADING_MSGS)}*", gr.update()

    try:
        response = await rag_engine.query(user_text)
        yield response, gr.update()
    except Exception as e:
        logger.error(f"Query failed: {e}")
        yield f"Error: {e}", gr.update()


# ---------------------------------------------------------------------------
# Handbook generation pipeline
# ---------------------------------------------------------------------------

async def _generate_handbook(
    topic: str,
    rag_engine: RAGEngine,
    handbook_gen: HandbookGenerator,
):
    """Run handbook generation. Yields (response_text, download_update) tuples."""
    global _last_handbook_topic, _last_handbook_path

    yield "**Planning** handbook structure...", gr.update()

    status = {"stage": "planning", "total": 0, "done": 0}

    def on_progress(stage, total, done):
        status["stage"] = stage
        status["total"] = total
        status["done"] = done

    try:
        gen_task = asyncio.create_task(
            handbook_gen.generate_parallel(topic, rag_engine, on_progress)
        )

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

            yield msg, gr.update()
            await asyncio.sleep(0.3)

        final, plan = await gen_task
        pdf_path, md_path = handbook_gen.save(final, topic)
        _last_handbook_topic = topic
        _last_handbook_path = pdf_path
        wc = len(final.split())

        yield (
            f"**Handbook complete!** {wc:,} words generated.\n\n"
            f"---\n\n{final[:2000]}...\n\n---\n\n"
            f"*Full {wc:,}-word handbook available via the download button below.*"
        ), gr.update()

    except Exception as e:
        logger.error(f"Handbook failed: {e}")
        yield f"**Error:** {e}", gr.update()
