"""
Handbook Generator — AgentWrite/LongWriter pipeline for generating
20,000+ word structured handbooks from research papers using Grok 4.1.

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

BATCH_SIZE = 12  # Sections per parallel batch


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

PLAN_PROMPT = """You are a professional handbook architect. Given the following instruction and \
source material from uploaded research papers, create a chapter plan for a comprehensive \
{target_words}-word handbook.

Each chapter should have a SHORT title (5-10 words max).
Each chapter should be between {min_words} and {max_words} words.
Create exactly {num_sections} chapters.
Do NOT create more than {num_sections} chapters.

Structure the handbook logically:
- Start with introduction and foundational concepts
- Progress through core methodologies and technical details
- Cover key findings, results, and evidence
- Include practical applications and implementation guidance
- End with challenges, future directions, and conclusions

Format each entry EXACTLY as:
Paragraph [N] - Main Point: [Short chapter title] - Word Count: [target word count]

IMPORTANT: Keep chapter titles SHORT. Good: "Core Methodology and Approach"
Bad: "Detailed exploration of the methodology used including all sub-components and their relationships"

Instruction: {instruction}

Context from source documents:
{rag_context}

Output ONLY the chapter plan entries, nothing else."""


WRITE_PROMPT = """You are writing Chapter {section_number} of a professional handbook.

Chapter title: {section_title}
Handbook topic: {instruction}

Full chapter plan (your chapter is marked with >>>):
{full_plan}

Relevant context from source documents:
{rag_context}

{prior_context}

Write this chapter following this EXACT structure:

1. Start with: ## Chapter {section_number}: {section_title}
2. Use ### for subsections (numbered like {section_number}.1, {section_number}.2, etc.)
3. Use #### for sub-subsections if needed

CRITICAL formatting rules:
- Use the chapter/subsection numbering system (e.g., "### 3.1 Subsection Title")
- Use bullet points and numbered lists for key information
- Cite source papers by author name (e.g., "According to Smith et al., ...")
- Include specific data, numbers, statistics from the sources
- Write in authoritative, professional tone
- Do NOT include "(Word count: ...)" or any meta-commentary
- Do NOT include "References" — those go at the end of the full handbook
- Target exactly {target_word_count} words
- Output ONLY the chapter content, nothing else"""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class HandbookGenerator:
    """Generates 20,000+ word structured handbooks using parallel AgentWrite/LongWriter."""

    def __init__(self, rag_engine):
        self.rag = rag_engine
        self.client = AsyncOpenAI(api_key=XAI_API_KEY, base_url=XAI_BASE_URL)
        logger.info(f"HandbookGenerator ready — model: {XAI_MODEL}")

    @retry(
        wait=wait_exponential(min=1, max=15),
        stop=stop_after_attempt(4),
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
            prompt += f"\n\nIMPORTANT: You MUST create at least {num_sections} sections. Try again."
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
            section_number=section.number,
            section_title=section.main_point,
            instruction=instruction,
            full_plan=plan_text,
            rag_context=context or "No additional context.",
            prior_context=prior_context,
            target_word_count=section.word_count,
        )

        return await self._call_llm(prompt, max_tokens=4096)

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
        refs = rag_engine.get_references()
        source_title = rag_engine.get_source_title()
        final = self.compile(instruction, all_paragraphs, plan, refs, source_title)
        t_end = time.time()
        logger.info(
            f"Handbook generated in {t_end - t_start:.1f}s total "
            f"(plan: {t_plan-t_start:.0f}s, ctx: {t_ctx-t_plan:.0f}s, "
            f"write: {t_write-t_ctx:.0f}s, compile: {t_end-t_write:.0f}s)"
        )

        return final, plan

    # --- Compilation & export ---

    def _extract_title(self, instruction: str) -> str:
        """Extract a meaningful title from the instruction."""
        strip_words = {
            "create", "generate", "write", "make", "build", "produce",
            "a", "an", "the", "handbook", "on", "about", "for", "of",
            "from", "using", "based", "regarding", "concerning",
        }
        words = instruction.strip().split()
        core = [w for w in words if w.lower().strip(".,;:!?") not in strip_words and w.isalpha()]
        if not core:
            return "Handbook"
        topic = " ".join(w.capitalize() for w in core[:6])
        return f"{topic} \u2014 Handbook"

    def compile(
        self, instruction: str, paragraphs: list[str], plan: list[SectionPlan],
        references: str = "", source_title: str = "",
    ) -> str:
        """Assemble all sections into a final handbook with TOC and references."""
        # Use source paper name as title, fall back to extracted topic
        if source_title:
            title = source_title
        else:
            title = self._extract_title(instruction)

        # Build clean TOC with dotted leaders
        toc = ["## Table of Contents\n"]
        for s in plan:
            toc.append(f"**Chapter {s.number}** — {s.main_point}")

        # Clean up body — strip word count meta-text and stray "References" mid-doc
        body = "\n\n".join(paragraphs)
        body = re.sub(r"\n*\(Word [Cc]ount:?\s*\d+\w*\)\n*", "\n", body)
        body = re.sub(r"\n*\*?\*?Word [Cc]ount:?\*?\*?\s*~?\d[\d,]*\+?\s*words?\n*", "\n", body)
        # Remove mid-document "References" sections (keep only the final one we add)
        body = re.sub(
            r"\n+#{1,3}\s*References\s*\n+((?:\[\d+\].*\n*)+)",
            "", body
        )
        body = re.sub(r"\n{3,}", "\n\n", body)

        wc = len(body.split())
        logger.info(f"Handbook compiled: {wc:,} words")

        # Build references section from source documents
        if references:
            refs_section = f"## References\n\n{references}\n"
        else:
            refs_section = (
                "## References\n\n"
                "No references section found in the source documents.\n"
            )

        return (
            f"# {title}\n\n"
            f"{chr(10).join(toc)}\n\n---\n\n"
            f"{body}\n\n---\n\n"
            f"{refs_section}"
        )

    def save(self, handbook_md: str, topic: str = "handbook") -> tuple[str, str]:
        """Save handbook as Markdown and PDF. Returns (pdf_path, md_path)."""
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Extract a clean short name from the topic
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
            return md_path, md_path

        return pdf_path, md_path


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
    """Convert markdown to a professional handbook-style PDF."""
    from fpdf import FPDF
    import time as _time

    BLACK = (0, 0, 0)
    DARK = (30, 30, 30)
    GRAY = (100, 100, 100)

    class HandbookPDF(FPDF):
        """Custom PDF with page headers and footers."""
        def __init__(self):
            super().__init__()
            self._handbook_title = "Handbook"
            self._page_started = False

        def header(self):
            if self._page_started and self.page_no() > 1:
                self.set_font("Times", "I", 9)
                self.set_text_color(*GRAY)
                self.cell(0, 8, self._handbook_title, align="R")
                self.ln(4)
                self.set_draw_color(180, 180, 180)
                self.set_line_width(0.3)
                self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
                self.ln(4)

        def footer(self):
            if self._page_started and self.page_no() > 1:
                self.set_y(-20)
                self.set_font("Times", "", 9)
                self.set_text_color(*GRAY)
                self.cell(0, 10, str(self.page_no() - 1), align="C")

    pdf = HandbookPDF()
    pdf.set_auto_page_break(auto=True, margin=25)
    pdf.set_margins(38, 25, 25)  # wider left margin like the reference

    # Extract title
    title = "Handbook"
    for line in md_text.split("\n"):
        if line.strip().startswith("# "):
            title = line.strip()[2:]
            break
    pdf._handbook_title = title

    # --- Title page (no header/footer) ---
    pdf.add_page()
    pdf.ln(80)
    pdf.set_font("Times", "B", 24)
    pdf.set_text_color(*BLACK)
    pdf.multi_cell(0, 12, _sanitize(title), align="C")
    pdf.ln(6)
    pdf.set_draw_color(*BLACK)
    pdf.set_line_width(1.0)
    center = pdf.w / 2
    pdf.line(center - 50, pdf.get_y(), center + 50, pdf.get_y())
    pdf.ln(2)
    pdf.set_line_width(0.4)
    pdf.line(center - 50, pdf.get_y(), center + 50, pdf.get_y())
    pdf._page_started = True

    # --- Content pages ---
    pdf.add_page()
    in_toc = False
    skip_first_title = True
    section_count = 0
    lines = md_text.split("\n")
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()

        if not stripped:
            pdf.ln(2)
            i += 1
            continue

        if stripped.startswith("# ") and skip_first_title:
            skip_first_title = False
            i += 1
            continue

        # Skip plan text and word count leaking through
        if re.match(r"^Paragraph\s+\d+\s*[-\u2013\u2014]", stripped):
            i += 1
            continue
        if re.match(r"^\(?Word [Cc]ount:?\s*\d", stripped):
            i += 1
            continue

        # --- Markdown table detection ---
        if "|" in stripped and stripped.startswith("|"):
            table_lines = []
            while i < len(lines) and "|" in lines[i].strip() and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            _render_table(pdf, table_lines)
            pdf.ln(4)
            continue

        # H1
        if stripped.startswith("# "):
            pdf.add_page()
            pdf.ln(10)
            pdf.set_font("Times", "B", 20)
            pdf.set_text_color(*BLACK)
            pdf.multi_cell(0, 10, _sanitize(stripped[2:]))
            pdf.ln(4)

        # H2 — chapter headings / major sections
        elif stripped.startswith("## "):
            heading = stripped[3:]
            in_toc = "table of contents" in heading.lower()

            if not in_toc:
                section_count += 1
                # Start each chapter on a new page
                if section_count > 1 or pdf.get_y() > 80:
                    pdf.add_page()

            pdf.ln(8)
            pdf.set_font("Times", "B", 16)
            pdf.set_text_color(*BLACK)
            pdf.multi_cell(0, 10, _sanitize(heading))
            pdf.ln(4)

        # H3 — subsections
        elif stripped.startswith("### "):
            pdf.ln(5)
            pdf.set_font("Times", "B", 12)
            pdf.set_text_color(*BLACK)
            pdf.multi_cell(0, 8, _sanitize(stripped[4:]))
            pdf.ln(2)

        # H4 — sub-subsections
        elif stripped.startswith("#### "):
            pdf.ln(3)
            pdf.set_font("Times", "BI", 11)
            pdf.set_text_color(*DARK)
            pdf.multi_cell(0, 7, _sanitize(stripped[5:]))
            pdf.ln(2)

        # Horizontal rule
        elif stripped == "---":
            pdf.ln(3)
            pdf.set_draw_color(180, 180, 180)
            pdf.set_line_width(0.3)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(3)

        # Italic text (standalone line)
        elif stripped.startswith("*") and stripped.endswith("*") and not stripped.startswith("**"):
            pdf.set_font("Times", "I", 10)
            pdf.set_text_color(*GRAY)
            pdf.multi_cell(0, 6, _sanitize(stripped.strip("*")))
            pdf.set_text_color(*BLACK)
            pdf.ln(2)

        # Bold text (standalone line)
        elif stripped.startswith("**") and stripped.endswith("**"):
            pdf.set_font("Times", "B", 11)
            pdf.set_text_color(*BLACK)
            pdf.multi_cell(0, 7, _sanitize(stripped.strip("*")))
            pdf.ln(2)

        # Numbered list (TOC)
        elif re.match(r"^\d+\.\s", stripped) and in_toc:
            pdf.set_font("Times", "", 10)
            pdf.set_text_color(*BLACK)
            pdf.cell(10)
            pdf.multi_cell(0, 6, _sanitize(stripped))
            pdf.ln(1)

        # Numbered list (body)
        elif re.match(r"^\d+\.\s", stripped):
            pdf.set_font("Times", "", 11)
            pdf.set_text_color(*BLACK)
            pdf.cell(10)
            pdf.multi_cell(0, 7, _sanitize(stripped))
            pdf.ln(2)

        # Bullet points
        elif stripped.startswith("- ") or stripped.startswith("* "):
            pdf.set_font("Times", "", 11)
            pdf.set_text_color(*BLACK)
            bullet_text = stripped[2:]
            pdf.cell(10)
            pdf.cell(5, 7, _sanitize("-"))
            pdf.multi_cell(0, 7, _sanitize(bullet_text))
            pdf.ln(1)

        # Regular paragraph with inline bold
        else:
            pdf.set_text_color(*BLACK)
            _render_paragraph(pdf, stripped)
            pdf.ln(4)

        i += 1

    pdf.output(pdf_path)


def _render_table(pdf, table_lines: list[str]) -> None:
    """Render markdown table lines as a proper PDF table with borders and auto-sized columns."""
    if not table_lines:
        return

    # Parse cells from each row
    rows = []
    for line in table_lines:
        # Skip separator lines (|---|---|)
        if re.match(r"^\|[\s\-:| ]+\|$", line):
            continue
        cells = [c.strip() for c in line.split("|")]
        # Remove empty first/last from leading/trailing |
        if cells and not cells[0]:
            cells = cells[1:]
        if cells and not cells[-1]:
            cells = cells[:-1]
        if cells:
            rows.append(cells)

    if not rows:
        return

    num_cols = max(len(r) for r in rows)
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    font_size = 9

    # Calculate column widths proportional to max content length
    max_lens = [0] * num_cols
    for row in rows:
        for j, cell in enumerate(row):
            clean = cell.strip("*")
            max_lens[j] = max(max_lens[j], len(clean))

    # Ensure minimum width and compute proportional widths
    total_chars = sum(max(l, 3) for l in max_lens) or 1
    col_widths = [max(usable_width * max(l, 3) / total_chars, 15) for l in max_lens]

    # Scale to fit exactly
    scale = usable_width / sum(col_widths)
    col_widths = [w * scale for w in col_widths]

    row_height = 7

    # Use smaller font if table is wide
    if num_cols > 4:
        font_size = 8

    pdf.set_draw_color(100, 100, 100)
    pdf.set_line_width(0.3)

    # Render header row with bold + gray background
    header = rows[0]
    pdf.set_font("Times", "B", font_size)
    pdf.set_fill_color(230, 230, 230)
    x_start = pdf.get_x()
    for j in range(num_cols):
        cell = header[j] if j < len(header) else ""
        cell = cell.strip("*")
        pdf.cell(col_widths[j], row_height, _sanitize(cell), border=1, fill=True, align="C")
    pdf.ln(row_height)

    # Render data rows
    for row in rows[1:]:
        for j in range(num_cols):
            cell = row[j] if j < len(row) else ""
            # Bold cell if wrapped in **
            if cell.startswith("**") and cell.endswith("**"):
                pdf.set_font("Times", "B", font_size)
                cell = cell.strip("*")
            else:
                pdf.set_font("Times", "", font_size)
            pdf.cell(col_widths[j], row_height, _sanitize(cell), border=1, align="C")
        pdf.ln(row_height)


def _render_paragraph(pdf, text: str) -> None:
    """Render a paragraph with inline **bold** support, using Times font."""
    parts = re.split(r"(\*\*.*?\*\*)", text)

    if len(parts) == 1:
        pdf.set_font("Times", "", 11)
        # First-line indent
        pdf.cell(8)
        pdf.multi_cell(0, 7, _sanitize(text))
        return

    # First-line indent for paragraph with mixed bold
    pdf.cell(8)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            pdf.set_font("Times", "B", 11)
            pdf.write(7, _sanitize(part.strip("*")))
        elif part:
            pdf.set_font("Times", "", 11)
            pdf.write(7, _sanitize(part))

    pdf.ln(7)
