"""
Handbook Generator — AgentWrite/LongWriter pipeline for generating
20,000+ word structured handbooks using Grok 4.1.

Uses parallel batch writing for speed: sections within each batch
are written concurrently, with batch summaries maintaining coherence.
"""

import re
import os
import asyncio
import logging
import time
from openai import AsyncOpenAI
from dataclasses import dataclass
from tenacity import (
    retry, wait_exponential, stop_after_attempt,
    retry_if_exception_type, before_sleep_log,
)
from app.config import (
    XAI_API_KEY, XAI_BASE_URL, XAI_MODEL,
    MIN_WORDS_PER_PARAGRAPH, MAX_WORDS_PER_PARAGRAPH,
    TARGET_HANDBOOK_WORDS, OUTPUT_DIR,
)

logger = logging.getLogger("handbook.generator")

BATCH_SIZE = 6  # Sections per parallel batch


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SectionPlan:
    """A single planned section of the handbook."""
    number: int
    main_point: str
    word_count: int


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PLAN_PROMPT = """You are a professional handbook architect. Given the following instruction, \
break it down into a detailed writing plan for a comprehensive handbook.

Each paragraph should be between {min_words} and {max_words} words.
The total handbook should be approximately {target_words} words (not more, not less).
Create exactly {num_sections} paragraphs to reach this target.
Aim for an average of 500 words per paragraph. Do NOT create more than {num_sections} paragraphs.

Format each entry EXACTLY as:
Paragraph [N] - Main Point: [Detailed description of what this paragraph covers] - Word Count: [target word count]

Instruction: {instruction}

Context from source documents:
{rag_context}

Output ONLY the paragraph plan entries, nothing else."""


WRITE_PROMPT = """You are an excellent writing assistant creating a professional handbook.

Original instruction: {instruction}

Full writing plan (your section is marked with >>>):
{full_plan}

Relevant context from source documents:
{rag_context}

{prior_context}

Now write the section marked with >>> above.

IMPORTANT:
- Write ONLY this section, do not repeat other sections
- Use the source document context to ensure accuracy and include specific details
- Cite or reference the source documents where appropriate
- Target exactly {target_word_count} words for this section
- Use proper markdown headings (### for section titles)
- Maintain professional tone
- Do not write a conclusion or summary unless this is the final section
- Output ONLY the paragraph text, nothing else"""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class HandbookGenerator:
    """Generates 20,000+ word handbooks using parallel AgentWrite/LongWriter."""

    def __init__(self, rag_engine):
        self.rag = rag_engine
        self.client = AsyncOpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL)
        logger.info(f"HandbookGenerator ready — model: {XAI_MODEL}")

    @retry(
        wait=wait_exponential(min=5, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _call_llm(self, prompt: str, max_tokens: int = 4096) -> str:
        """Call Grok 4.1 with automatic retry on failure."""
        response = await self.client.chat.completions.create(
            model=XAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response from LLM")
        return content

    # --- Planning ---

    async def create_plan(self, instruction: str, context: str = "") -> list[SectionPlan]:
        """Generate a structured writing plan for the handbook."""
        num_sections = TARGET_HANDBOOK_WORDS // (
            (MIN_WORDS_PER_PARAGRAPH + MAX_WORDS_PER_PARAGRAPH) // 2
        )
        logger.info(f"Planning: {num_sections} sections for ~{TARGET_HANDBOOK_WORDS} words")

        prompt = PLAN_PROMPT.format(
            min_words=MIN_WORDS_PER_PARAGRAPH,
            max_words=MAX_WORDS_PER_PARAGRAPH,
            target_words=TARGET_HANDBOOK_WORDS,
            num_sections=num_sections,
            instruction=instruction,
            rag_context=context or "No source documents available.",
        )

        raw = await self._call_llm(prompt, max_tokens=4096)
        plan = self._parse_plan(raw)

        if len(plan) < 10:
            logger.warning(f"Plan too short ({len(plan)} sections), retrying...")
            prompt += "\n\nIMPORTANT: You MUST create at least 20 paragraphs. Try again."
            raw = await self._call_llm(prompt, max_tokens=4096)
            plan = self._parse_plan(raw)

        logger.info(f"Plan ready: {len(plan)} sections")
        return plan

    def _parse_plan(self, raw: str) -> list[SectionPlan]:
        """Parse LLM plan output into SectionPlan objects."""
        patterns = [
            r"Paragraph\s+(\d+)\s*[-\u2013\u2014]\s*Main Point:\s*(.+?)\s*[-\u2013\u2014]\s*Word Count:\s*(\d+)",
            r"Paragraph\s+(\d+)\s*[-\u2013\u2014]\s*(.+?):\s*\n.*?[-\u2013\u2014]\s*Word Count:\s*(\d+)",
            r"Paragraph\s+(\d+)\s*[-\u2013\u2014]\s*(.+?)\s*[-\u2013\u2014]\s*Word Count:\s*(\d+)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, raw)
            if matches:
                return [SectionPlan(int(n), p.strip(), int(w)) for n, p, w in matches]

        logger.error("Failed to parse plan from LLM output")
        return []

    # --- Parallel batch writing ---

    async def write_section_parallel(
        self,
        instruction: str,
        plan: list[SectionPlan],
        section: SectionPlan,
        context: str,
        prior_summary: str = "",
    ) -> str:
        """Write a single section with plan context and prior batch summary."""
        # Mark current section in plan with >>>
        plan_lines = []
        for s in plan:
            prefix = ">>> " if s.number == section.number else "    "
            plan_lines.append(
                f"{prefix}Paragraph {s.number} - Main Point: {s.main_point} - Word Count: {s.word_count}"
            )
        plan_text = "\n".join(plan_lines)

        if prior_summary:
            prior_context = (
                f"Summary of previously written sections (for coherence):\n{prior_summary}"
            )
        else:
            prior_context = "This is the beginning of the handbook."

        prompt = WRITE_PROMPT.format(
            instruction=instruction,
            full_plan=plan_text,
            rag_context=context or "No additional context.",
            prior_context=prior_context,
            target_word_count=section.word_count,
        )

        return await self._call_llm(prompt, max_tokens=2048)

    async def write_batch(
        self,
        instruction: str,
        plan: list[SectionPlan],
        batch: list[SectionPlan],
        contexts: dict[int, str],
        prior_summary: str = "",
    ) -> list[str]:
        """Write a batch of sections in parallel."""
        tasks = [
            self.write_section_parallel(
                instruction, plan, section,
                contexts.get(section.number, ""),
                prior_summary,
            )
            for section in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        paragraphs = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Section {batch[i].number} failed: {result}")
                paragraphs.append(
                    f"*[Section {batch[i].number}: {batch[i].main_point} — generation failed]*"
                )
            else:
                paragraphs.append(result)

        return paragraphs

    def _summarize_text(self, text: str, max_words: int = 500) -> str:
        """Create a brief summary of written text for next batch context."""
        words = text.split()
        if len(words) <= max_words:
            return text
        # Take first 200 and last 300 words
        return " ".join(words[:200]) + "\n...\n" + " ".join(words[-300:])

    async def generate_parallel(
        self,
        instruction: str,
        rag_engine,
        progress_callback=None,
    ) -> tuple[str, list[SectionPlan]]:
        """Generate a full handbook using parallel batch writing."""
        t_start = time.time()

        # Phase 1: Plan
        if progress_callback:
            progress_callback("planning", 0, 0)

        broad_ctx = await rag_engine.get_context(instruction)
        plan = await self.create_plan(instruction, broad_ctx[:2000])
        if not plan:
            raise ValueError("Failed to generate plan")

        total = len(plan)
        t_plan = time.time()
        logger.info(f"Plan generated in {t_plan - t_start:.1f}s")

        # Phase 2: Fetch all RAG contexts in parallel
        if progress_callback:
            progress_callback("fetching_context", total, 0)

        context_tasks = {
            s.number: rag_engine.get_context(s.main_point)
            for s in plan
        }
        context_results = await asyncio.gather(*context_tasks.values())
        contexts = dict(zip(context_tasks.keys(), context_results))

        t_ctx = time.time()
        logger.info(f"RAG contexts fetched in {t_ctx - t_plan:.1f}s")

        # Phase 3: Write in batches
        all_paragraphs = []
        prior_summary = ""
        batches = [plan[i:i + BATCH_SIZE] for i in range(0, total, BATCH_SIZE)]

        for batch_idx, batch in enumerate(batches):
            if progress_callback:
                done = len(all_paragraphs)
                progress_callback("writing", total, done)

            logger.info(
                f"Writing batch {batch_idx + 1}/{len(batches)}: "
                f"sections {batch[0].number}-{batch[-1].number}"
            )

            batch_results = await self.write_batch(
                instruction, plan, batch, contexts, prior_summary
            )
            all_paragraphs.extend(batch_results)

            # Build summary of everything written so far for next batch
            written_so_far = "\n\n".join(all_paragraphs)
            prior_summary = self._summarize_text(written_so_far)

            wc = len(written_so_far.split())
            logger.info(f"Batch {batch_idx + 1} done: {wc:,} words total")

        t_write = time.time()
        logger.info(f"All sections written in {t_write - t_ctx:.1f}s")

        # Phase 4: Compile
        final = self.compile(instruction, all_paragraphs, plan)
        t_end = time.time()
        logger.info(
            f"Handbook generated in {t_end - t_start:.1f}s total "
            f"(plan: {t_plan-t_start:.0f}s, ctx: {t_ctx-t_plan:.0f}s, "
            f"write: {t_write-t_ctx:.0f}s, compile: {t_end-t_write:.0f}s)"
        )

        return final, plan

    # --- Compilation & export ---

    def _extract_title(self, instruction: str) -> str:
        """Always returns 'Handbook' as the title."""
        return "Handbook"

    def compile(
        self, instruction: str, paragraphs: list[str], plan: list[SectionPlan]
    ) -> str:
        """Assemble all sections into a final handbook with TOC and references."""
        title = self._extract_title(instruction)

        toc = ["## Table of Contents\n"]
        for s in plan:
            toc.append(f"{s.number}. {s.main_point}")

        body = "\n\n".join(paragraphs)
        wc = len(body.split())
        logger.info(f"Handbook compiled: {wc:,} words")

        return (
            f"# {title}\n\n"
            f"{chr(10).join(toc)}\n\n---\n\n"
            f"{body}\n\n---\n\n"
            f"## References\n\n"
            f"*This handbook was generated from uploaded PDF source documents "
            f"using Retrieval-Augmented Generation (RAG). All content is derived "
            f"from and grounded in the source materials.*\n"
        )

    def save(self, handbook_md: str, topic: str = "handbook") -> str:
        """Save handbook as Markdown and PDF with unique timestamped name."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Extract a clean short name from the topic
        # Remove common filler words to get the core subject
        filler = ["create", "a", "the", "on", "about", "for", "of", "handbook", "generate", "write", "make"]
        words = topic.lower().split()
        core = [w for w in words if w not in filler and w.isalpha()]
        name = "_".join(core[:4]) if core else "handbook"
        name = name.title().replace("_", "-")
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"Handbook_{name}_{ts}"

        md_path = os.path.join(OUTPUT_DIR, f"{base}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(handbook_md)
        logger.info(f"Markdown saved: {md_path}")

        pdf_path = os.path.join(OUTPUT_DIR, f"{base}.pdf")
        try:
            _markdown_to_pdf(handbook_md, pdf_path)
            logger.info(f"PDF saved: {pdf_path}")
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            return md_path

        return pdf_path


# ---------------------------------------------------------------------------
# PDF export helpers
# ---------------------------------------------------------------------------

def _sanitize(text: str) -> str:
    """Replace unicode characters that fpdf can't render."""
    replacements = {
        "\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2022": "-",
        "\u00a0": " ", "\u200b": "", "\u2032": "'", "\u2033": '"',
        "\u2192": "->", "\u2190": "<-", "\u2265": ">=", "\u2264": "<=",
        "\u2023": ">",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _markdown_to_pdf(md_text: str, pdf_path: str) -> None:
    """Convert markdown to a professionally formatted PDF."""
    from fpdf import FPDF

    # Colors — professional black/gray scheme
    HEADING = (20, 20, 20)
    DARK = (30, 41, 59)
    GRAY = (100, 116, 139)
    ACCENT = (60, 60, 60)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.set_margins(25, 25, 25)

    # --- Title page ---
    pdf.add_page()
    pdf.ln(60)
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_text_color(*HEADING)

    # Extract title from first # heading
    title = "Handbook"
    for line in md_text.split("\n"):
        if line.strip().startswith("# "):
            title = line.strip()[2:]
            break
    pdf.multi_cell(0, 14, _sanitize(title), align="C")

    pdf.ln(10)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.8)
    center_x = pdf.w / 2
    pdf.line(center_x - 40, pdf.get_y(), center_x + 40, pdf.get_y())

    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 8, "Generated using AI Handbook Generator", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, "Powered by Grok 4.1 | LightRAG | Supabase", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)
    import time
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 8, time.strftime("Generated on %B %d, %Y"), align="C", new_x="LMARGIN", new_y="NEXT")

    # --- Content pages ---
    pdf.add_page()
    in_toc = False
    skip_first_title = True

    for line in md_text.split("\n"):
        stripped = line.strip()

        # Skip empty lines (add small spacing)
        if not stripped:
            pdf.ln(2)
            continue

        # Main title (skip, already on title page)
        if stripped.startswith("# ") and skip_first_title:
            skip_first_title = False
            continue

        # Skip lines that look like plan text leaking through
        if re.match(r"^Paragraph\s+\d+\s*[-\u2013\u2014]", stripped):
            continue

        # H1
        if stripped.startswith("# "):
            pdf.ln(6)
            pdf.set_font("Helvetica", "B", 22)
            pdf.set_text_color(*HEADING)
            pdf.multi_cell(0, 12, _sanitize(stripped[2:]))
            pdf.ln(4)

        # H2 — Table of Contents / References / Section headers
        elif stripped.startswith("## "):
            heading = stripped[3:]
            pdf.ln(8)

            pdf.set_draw_color(*ACCENT)
            pdf.set_line_width(0.5)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + 25, pdf.get_y())
            pdf.ln(3)

            pdf.set_font("Helvetica", "B", 16)
            pdf.set_text_color(*HEADING)
            pdf.multi_cell(0, 10, _sanitize(heading))
            pdf.ln(3)

            in_toc = "table of contents" in heading.lower()

        # H3 — Section titles
        elif stripped.startswith("### "):
            pdf.ln(6)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(*HEADING)
            pdf.multi_cell(0, 9, _sanitize(stripped[5:] if stripped.startswith("#### ") else stripped[4:]))
            pdf.set_text_color(*DARK)
            pdf.ln(2)

        # H4
        elif stripped.startswith("#### "):
            pdf.ln(4)
            pdf.set_font("Helvetica", "BI", 12)
            pdf.set_text_color(*DARK)
            pdf.multi_cell(0, 8, _sanitize(stripped[5:]))
            pdf.ln(2)

        # Horizontal rule
        elif stripped == "---":
            pdf.ln(4)
            pdf.set_draw_color(200, 200, 200)
            pdf.set_line_width(0.3)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(4)

        # Italic text
        elif stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**"):
            pdf.set_font("Helvetica", "I", 10)
            pdf.set_text_color(*GRAY)
            pdf.multi_cell(0, 6, _sanitize(stripped.strip("*")))
            pdf.set_text_color(*DARK)
            pdf.ln(2)

        # Bold text
        elif stripped.startswith("**") and stripped.endswith("**"):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*DARK)
            pdf.multi_cell(0, 7, _sanitize(stripped.strip("*")))
            pdf.ln(2)

        # Numbered list items (TOC or regular)
        elif re.match(r"^\d+\.\s", stripped):
            if in_toc:
                pdf.set_font("Helvetica", "", 10)
                pdf.set_text_color(*DARK)
                pdf.cell(8)  # indent
                pdf.multi_cell(0, 6, _sanitize(stripped))
                pdf.ln(1)
            else:
                pdf.set_font("Helvetica", "", 11)
                pdf.set_text_color(*DARK)
                pdf.cell(6)  # indent
                pdf.multi_cell(0, 7, _sanitize(stripped))
                pdf.ln(2)

        # Bullet points
        elif stripped.startswith("- ") or stripped.startswith("* "):
            pdf.set_font("Helvetica", "", 11)
            pdf.set_text_color(*DARK)
            pdf.cell(6)  # indent
            pdf.multi_cell(0, 7, _sanitize("  " + stripped))
            pdf.ln(1)

        # Regular paragraph — handle inline **bold**
        else:
            pdf.set_text_color(*DARK)
            _render_paragraph(pdf, stripped)
            pdf.ln(3)

    pdf.output(pdf_path)


def _render_paragraph(pdf, text: str) -> None:
    """Render a paragraph with inline **bold** support."""
    import re
    parts = re.split(r"(\*\*.*?\*\*)", text)

    if len(parts) == 1:
        # No bold markers, simple render
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 7, _sanitize(text))
        return

    # Has bold parts — write as flowing text
    line = ""
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            # Flush normal text first
            if line:
                pdf.set_font("Helvetica", "", 11)
                pdf.write(7, _sanitize(line))
                line = ""
            # Write bold
            pdf.set_font("Helvetica", "B", 11)
            pdf.write(7, _sanitize(part.strip("*")))
        else:
            line += part

    # Flush remaining
    if line:
        pdf.set_font("Helvetica", "", 11)
        pdf.write(7, _sanitize(line))

    pdf.ln(7)
