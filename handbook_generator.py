import asyncio
import re
import os
from cerebras.cloud.sdk import Cerebras
from dataclasses import dataclass
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from config import (
    CEREBRAS_API_KEY, CEREBRAS_MODEL_NAME,
    MIN_WORDS_PER_PARAGRAPH, MAX_WORDS_PER_PARAGRAPH,
    TARGET_HANDBOOK_WORDS, OUTPUT_DIR,
)
from rag_engine import RAGEngine


@dataclass
class ParagraphPlan:
    number: int
    main_point: str
    word_count: int


@dataclass
class HandbookProgress:
    total_paragraphs: int
    completed_paragraphs: int
    current_section: str
    total_words_written: int
    status: str  # "planning", "writing", "compiling", "complete", "error"


PLAN_PROMPT = """You are a professional handbook architect. Given the following instruction, \
break it down into a detailed writing plan for a comprehensive handbook.

Each paragraph should be between {min_words} and {max_words} words.
The total handbook should be approximately {target_words} words (not more, not less).
Create exactly {num_paragraphs} paragraphs to reach this target.
Aim for an average of 500 words per paragraph. Do NOT create more than {num_paragraphs} paragraphs.

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
- Target exactly {target_word_count} words for this section
- Maintain professional tone and logical flow from previous sections
- You may add a small subtitle at the start of this section
- Do not write a conclusion or summary unless this is the final section
- Output ONLY the paragraph text, nothing else"""


class HandbookGenerator:
    def __init__(self, rag_engine: RAGEngine):
        self.rag_engine = rag_engine
        self.client = Cerebras(api_key=CEREBRAS_API_KEY)

    @retry(
        wait=wait_exponential(min=30, max=120),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(Exception),
    )
    async def _generate(self, prompt: str, max_tokens: int = 4096) -> str:
        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=CEREBRAS_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content

    async def generate_plan(self, instruction: str, rag_context: str = "") -> list[ParagraphPlan]:
        num_paragraphs = TARGET_HANDBOOK_WORDS // ((MIN_WORDS_PER_PARAGRAPH + MAX_WORDS_PER_PARAGRAPH) // 2)

        prompt = PLAN_PROMPT.format(
            min_words=MIN_WORDS_PER_PARAGRAPH,
            max_words=MAX_WORDS_PER_PARAGRAPH,
            target_words=TARGET_HANDBOOK_WORDS,
            num_paragraphs=num_paragraphs,
            instruction=instruction,
            rag_context=rag_context or "No source documents available.",
        )

        raw_plan = await self._generate(prompt, max_tokens=4096)
        plan = self._parse_plan_response(raw_plan)

        if len(plan) < 10:
            prompt += "\n\nIMPORTANT: You MUST create at least 30 paragraphs to reach 20,000 words. Try again."
            raw_plan = await self._generate(prompt, max_tokens=4096)
            plan = self._parse_plan_response(raw_plan)

        return plan

    def _parse_plan_response(self, raw_plan: str) -> list[ParagraphPlan]:
        # Try original format: Paragraph N - Main Point: ... - Word Count: N
        pattern = r"Paragraph\s+(\d+)\s*[-\u2013\u2014]\s*Main Point:\s*(.+?)\s*[-\u2013\u2014]\s*Word Count:\s*(\d+)"
        matches = re.findall(pattern, raw_plan)
        if matches:
            return [
                ParagraphPlan(number=int(n), main_point=point.strip(), word_count=int(wc))
                for n, point, wc in matches
            ]

        # Fallback: multi-line format where main point is on one line, description + word count on next
        # e.g. "Paragraph 1 - Introduction to LongWriter:\nA description... - Word Count: 500"
        pattern2 = r"Paragraph\s+(\d+)\s*[-\u2013\u2014]\s*(.+?):\s*\n.*?[-\u2013\u2014]\s*Word Count:\s*(\d+)"
        matches = re.findall(pattern2, raw_plan)
        if matches:
            return [
                ParagraphPlan(number=int(n), main_point=point.strip(), word_count=int(wc))
                for n, point, wc in matches
            ]

        # Fallback 2: single-line without "Main Point:" prefix
        # e.g. "Paragraph 1 - Introduction to LongWriter - Word Count: 500"
        pattern3 = r"Paragraph\s+(\d+)\s*[-\u2013\u2014]\s*(.+?)\s*[-\u2013\u2014]\s*Word Count:\s*(\d+)"
        matches = re.findall(pattern3, raw_plan)
        if matches:
            return [
                ParagraphPlan(number=int(n), main_point=point.strip(), word_count=int(wc))
                for n, point, wc in matches
            ]

        return []

    async def write_paragraph(
        self,
        instruction: str,
        plan: list[ParagraphPlan],
        written_so_far: str,
        current_step: ParagraphPlan,
        rag_context: str,
    ) -> str:
        plan_text = "\n".join(
            f"Paragraph {p.number} - Main Point: {p.main_point} - Word Count: {p.word_count}"
            for p in plan
        )

        # Manage context window: truncate if too long
        display_text = written_so_far
        word_count = len(written_so_far.split())
        if word_count > 8000:
            words = written_so_far.split()
            truncated = " ".join(words[-3000:])
            covered = [p.main_point for p in plan if p.number < current_step.number]
            display_text = (
                f"[Earlier sections omitted. {word_count} words written so far covering: "
                f"{', '.join(covered[:10])}{'...' if len(covered) > 10 else ''}]\n\n"
                f"...{truncated}"
            )

        step_text = f"Paragraph {current_step.number} - Main Point: {current_step.main_point} - Word Count: {current_step.word_count}"

        prompt = WRITE_PROMPT.format(
            instruction=instruction,
            full_plan=plan_text,
            rag_context=rag_context or "No additional context.",
            written_text=display_text or "[This is the beginning of the handbook]",
            current_step=step_text,
            target_word_count=current_step.word_count,
        )

        return await self._generate(prompt, max_tokens=2048)

    async def generate_handbook(self, instruction: str, progress_callback=None) -> str:
        # Phase 0: Get broad context for planning
        try:
            broad_context = await self.rag_engine.query(instruction, mode="global")
        except Exception:
            broad_context = ""

        if progress_callback:
            progress_callback(HandbookProgress(0, 0, "Planning handbook structure...", 0, "planning"))

        # Phase 1: Generate plan
        plan = await self.generate_plan(instruction, broad_context)

        if not plan:
            raise ValueError("Failed to generate a valid handbook plan. Please try again.")

        if progress_callback:
            progress_callback(HandbookProgress(len(plan), 0, "Plan ready, starting to write...", 0, "writing"))

        # Phase 2: Write each paragraph
        written_paragraphs = []
        written_text = ""

        for i, step in enumerate(plan):
            if progress_callback:
                progress_callback(HandbookProgress(
                    len(plan), i, step.main_point,
                    len(written_text.split()), "writing",
                ))

            # Get section-specific RAG context
            try:
                section_context = await self.rag_engine.query_for_handbook_context(step.main_point)
            except Exception:
                section_context = ""

            paragraph = await self.write_paragraph(
                instruction=instruction,
                plan=plan,
                written_so_far=written_text,
                current_step=step,
                rag_context=section_context,
            )

            written_paragraphs.append(paragraph)
            written_text += "\n\n" + paragraph

            # Rate limit pause for Gemini free tier
            await asyncio.sleep(5)

        # Phase 3: Compile
        if progress_callback:
            progress_callback(HandbookProgress(
                len(plan), len(plan), "Compiling handbook...",
                len(written_text.split()), "compiling",
            ))

        final_handbook = self.compile_handbook(instruction, written_paragraphs, plan)

        if progress_callback:
            progress_callback(HandbookProgress(
                len(plan), len(plan), "Complete!",
                len(final_handbook.split()), "complete",
            ))

        return final_handbook

    def compile_handbook(
        self, instruction: str, paragraphs: list[str], plan: list[ParagraphPlan]
    ) -> str:
        # Build table of contents
        toc_lines = ["## Table of Contents\n"]
        for step in plan:
            toc_lines.append(f"{step.number}. {step.main_point}")

        toc = "\n".join(toc_lines)
        body = "\n\n".join(paragraphs)
        word_count = len(body.split())

        return (
            f"# Handbook: {instruction}\n\n"
            f"{toc}\n\n"
            f"---\n\n"
            f"{body}\n\n"
            f"---\n\n"
            f"*Total word count: approximately {word_count} words*"
        )

    def save_handbook(self, handbook_md: str, filename: str = "handbook") -> str:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Save Markdown
        md_path = os.path.join(OUTPUT_DIR, f"{filename}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(handbook_md)

        # Save PDF
        pdf_path = os.path.join(OUTPUT_DIR, f"{filename}.pdf")
        self._markdown_to_pdf(handbook_md, pdf_path)

        return pdf_path

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Replace unicode chars that fpdf can't handle with ASCII equivalents."""
        replacements = {
            "\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2022": "*",
            "\u00a0": " ", "\u200b": "", "\u2032": "'", "\u2033": '"',
            "\u2192": "->", "\u2190": "<-", "\u2265": ">=", "\u2264": "<=",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    @staticmethod
    def _markdown_to_pdf(md_text: str, pdf_path: str) -> None:
        from fpdf import FPDF

        sanitize = HandbookGenerator._sanitize_text
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()

        for line in md_text.split("\n"):
            stripped = line.strip()

            if stripped.startswith("# "):
                pdf.set_font("Helvetica", "B", 20)
                pdf.cell(0, 12, sanitize(stripped[2:]), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(4)
            elif stripped.startswith("## "):
                pdf.set_font("Helvetica", "B", 16)
                pdf.cell(0, 10, sanitize(stripped[3:]), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(3)
            elif stripped.startswith("### "):
                pdf.set_font("Helvetica", "B", 13)
                pdf.cell(0, 9, sanitize(stripped[4:]), new_x="LMARGIN", new_y="NEXT")
                pdf.ln(2)
            elif stripped == "---":
                pdf.ln(4)
                pdf.set_draw_color(200, 200, 200)
                pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
                pdf.ln(4)
            elif stripped.startswith("*") and stripped.endswith("*") and len(stripped) > 2:
                pdf.set_font("Helvetica", "I", 10)
                pdf.multi_cell(0, 6, sanitize(stripped.strip("*")))
                pdf.ln(2)
            elif stripped:
                pdf.set_font("Helvetica", "", 11)
                pdf.multi_cell(0, 6, sanitize(stripped))
                pdf.ln(2)

        pdf.output(pdf_path)
