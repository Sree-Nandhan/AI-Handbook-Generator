"""
Handbook Generator — AgentWrite/LongWriter pipeline for generating
20,000+ word structured handbooks using Grok 4.1.
"""

import re
import os
import logging
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
from app.rag_engine import RAGEngine

logger = logging.getLogger("handbook.generator")


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

Planned writing steps:
{full_plan}

Relevant context from source documents:
{rag_context}

Text written so far:
{written_text}

Now write the next section:
{current_step}

IMPORTANT:
- Write ONLY this section, do not repeat previously written text
- Use the source document context to ensure accuracy and include specific details
- Cite or reference the source documents where appropriate (e.g. "According to the research..." or "As described in the source material...")
- Target exactly {target_word_count} words for this section
- Use proper markdown headings (### for section titles)
- Maintain professional tone and logical flow from previous sections
- Do not write a conclusion or summary unless this is the final section
- Output ONLY the paragraph text, nothing else"""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class HandbookGenerator:
    """Generates 20,000+ word handbooks using AgentWrite/LongWriter."""

    def __init__(self, rag_engine: RAGEngine):
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
            prompt += "\n\nIMPORTANT: You MUST create at least 30 paragraphs. Try again."
            raw = await self._call_llm(prompt, max_tokens=4096)
            plan = self._parse_plan(raw)

        logger.info(f"Plan ready: {len(plan)} sections")
        return plan

    def _parse_plan(self, raw: str) -> list[SectionPlan]:
        """Parse LLM plan output into SectionPlan objects."""
        patterns = [
            # Paragraph N - Main Point: ... - Word Count: N
            r"Paragraph\s+(\d+)\s*[-\u2013\u2014]\s*Main Point:\s*(.+?)\s*[-\u2013\u2014]\s*Word Count:\s*(\d+)",
            # Multi-line: Paragraph N - Title:\n... - Word Count: N
            r"Paragraph\s+(\d+)\s*[-\u2013\u2014]\s*(.+?):\s*\n.*?[-\u2013\u2014]\s*Word Count:\s*(\d+)",
            # Paragraph N - Title - Word Count: N
            r"Paragraph\s+(\d+)\s*[-\u2013\u2014]\s*(.+?)\s*[-\u2013\u2014]\s*Word Count:\s*(\d+)",
        ]
        for pattern in patterns:
            matches = re.findall(pattern, raw)
            if matches:
                return [SectionPlan(int(n), p.strip(), int(w)) for n, p, w in matches]

        logger.error("Failed to parse plan from LLM output")
        return []

    # --- Section writing ---

    async def write_section(
        self,
        instruction: str,
        plan: list[SectionPlan],
        written_so_far: str,
        section: SectionPlan,
        context: str,
    ) -> str:
        """Write a single handbook section using the AgentWrite technique."""
        plan_text = "\n".join(
            f"Paragraph {s.number} - Main Point: {s.main_point} - Word Count: {s.word_count}"
            for s in plan
        )

        # Truncate earlier text if it exceeds context window budget
        display_text = written_so_far
        wc = len(written_so_far.split())
        if wc > 8000:
            words = written_so_far.split()
            truncated = " ".join(words[-3000:])
            covered = [s.main_point for s in plan if s.number < section.number]
            display_text = (
                f"[Earlier sections omitted. {wc} words written covering: "
                f"{', '.join(covered[:10])}{'...' if len(covered) > 10 else ''}]\n\n"
                f"...{truncated}"
            )

        step_text = (
            f"Paragraph {section.number} - Main Point: "
            f"{section.main_point} - Word Count: {section.word_count}"
        )

        prompt = WRITE_PROMPT.format(
            instruction=instruction,
            full_plan=plan_text,
            rag_context=context or "No additional context.",
            written_text=display_text or "[This is the beginning of the handbook]",
            current_step=step_text,
            target_word_count=section.word_count,
        )

        return await self._call_llm(prompt, max_tokens=2048)

    # --- Compilation & export ---

    def compile(
        self, instruction: str, paragraphs: list[str], plan: list[SectionPlan]
    ) -> str:
        """Assemble all sections into a final handbook with TOC, headings, and citations."""
        toc = ["## Table of Contents\n"]
        for s in plan:
            toc.append(f"{s.number}. {s.main_point}")

        body = "\n\n".join(paragraphs)
        wc = len(body.split())
        logger.info(f"Handbook compiled: {wc:,} words")

        return (
            f"# Handbook: {instruction}\n\n"
            f"{chr(10).join(toc)}\n\n---\n\n"
            f"{body}\n\n---\n\n"
            f"## References\n\n"
            f"*This handbook was generated from uploaded PDF source documents "
            f"using RAG (Retrieval-Augmented Generation). All content is derived "
            f"from and grounded in the source materials.*\n\n"
            f"---\n\n"
            f"*Total word count: approximately {wc:,} words*\n"
            f"*Generated using AgentWrite/LongWriter pipeline with Grok 4.1*"
        )

    def save(self, handbook_md: str, topic: str = "handbook") -> str:
        """Save handbook as Markdown and PDF with unique timestamped name."""
        import time
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        safe = "".join(c if c.isalnum() or c in " -_" else "" for c in topic)
        safe = safe.strip().replace(" ", "_")[:50] or "handbook"
        ts = time.strftime("%Y%m%d_%H%M%S")
        base = f"{safe}_{ts}"

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
# PDF export helpers (module-level to keep class clean)
# ---------------------------------------------------------------------------

def _sanitize(text: str) -> str:
    """Replace unicode characters that fpdf can't render."""
    replacements = {
        "\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2022": "*",
        "\u00a0": " ", "\u200b": "", "\u2032": "'", "\u2033": '"',
        "\u2192": "->", "\u2190": "<-", "\u2265": ">=", "\u2264": "<=",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _markdown_to_pdf(md_text: str, pdf_path: str) -> None:
    """Convert markdown text to a basic PDF document."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    for line in md_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 20)
            pdf.cell(0, 12, _sanitize(stripped[2:]), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)
        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, _sanitize(stripped[3:]), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
        elif stripped.startswith("### "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 9, _sanitize(stripped[4:]), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
        elif stripped == "---":
            pdf.ln(4)
            pdf.set_draw_color(200, 200, 200)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(4)
        elif stripped.startswith("*") and stripped.endswith("*") and len(stripped) > 2:
            pdf.set_font("Helvetica", "I", 10)
            pdf.multi_cell(0, 6, _sanitize(stripped.strip("*")))
            pdf.ln(2)
        else:
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, _sanitize(stripped))
            pdf.ln(2)

    pdf.output(pdf_path)
