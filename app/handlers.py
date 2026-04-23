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
        yield f"**{file_num} Extracting text** from `{fname}`...", frac
        try:
            path = save_uploaded_file(f)
            # Extract research paper title from first page
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

        # ── Extract references before indexing ──
        import pdfplumber
        frac = (idx + 0.2) / total
        yield f"**{file_num} Extracting references** from `{fname}`...", frac
        try:
            with pdfplumber.open(path) as pdf:
                raw = "\n\n".join(p.extract_text() or "" for p in pdf.pages)
            refs = extract_references(raw)
            if refs:
                rag_engine.add_references(fname, refs)
                logger.info(f"Extracted references from {fname}: {len(refs)} chars")
        except Exception as e:
            logger.warning(f"Reference extraction failed for {fname}: {e}")

        # ── Index (slow — show animated status) ──
        frac = (idx + 0.4) / total
        yield f"**{file_num} Indexing** `{fname}` into knowledge graph — this may take a moment...", frac
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
# Handbook generation — parallel batch pipeline
# ---------------------------------------------------------------------------

async def _generate_handbook(
    topic: str,
    history: list,
    rag_engine: RAGEngine,
    handbook_gen: HandbookGenerator,
):
    """Run parallel AgentWrite/LongWriter pipeline with live progress."""
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
