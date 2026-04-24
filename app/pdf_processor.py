"""
PDF text extraction and cleaning utilities.
"""

import os
import re
import shutil
import logging
import pdfplumber
from app.config import UPLOAD_DIR

logger = logging.getLogger("handbook.pdf")


def save_uploaded_file(file_path: str) -> str:
    """Copy an uploaded file to the uploads directory and log it."""
    import json, time as _time
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = os.path.basename(file_path)
    dest = os.path.join(UPLOAD_DIR, filename)
    if file_path != dest:
        shutil.copy2(file_path, dest)
    logger.info(f"Saved uploaded file: {filename}")

    # Append to upload log
    log_path = os.path.join(UPLOAD_DIR, "upload_log.json")
    log_entries = []
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                log_entries = json.load(f)
        except Exception:
            log_entries = []
    log_entries.append({
        "filename": filename,
        "path": dest,
        "size_bytes": os.path.getsize(dest),
        "uploaded_at": _time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(log_path, "w") as f:
        json.dump(log_entries, f, indent=2)

    return dest


def list_uploaded_pdfs() -> list[dict]:
    """List all previously uploaded PDFs."""
    import json
    log_path = os.path.join(UPLOAD_DIR, "upload_log.json")
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path) as f:
            return json.load(f)
    except Exception:
        return []


def extract_text(pdf_path: str) -> str:
    """Extract and clean text from a PDF file."""
    logger.info(f"Extracting text from: {os.path.basename(pdf_path)}")
    text_parts = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            logger.info(f"Extracted {len(text_parts)} pages from {len(pdf.pages)} total")
    except Exception as e:
        logger.error(f"Failed to extract PDF: {e}")
        raise

    raw_text = "\n\n".join(text_parts)
    cleaned = _clean_text(raw_text)
    logger.info(f"Cleaned text: {len(cleaned):,} chars")
    return cleaned


def _clean_text(raw_text: str) -> str:
    """Normalize whitespace, keeping references intact."""
    text = raw_text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    lines = text.split("\n")
    text = "\n".join(line.strip() for line in lines)

    return text.strip()


def extract_title(pdf_path: str) -> str:
    """Extract the research paper title from the first page of a PDF.

    Heuristic: the title is typically the first substantial line(s) on page 1,
    usually the largest / most prominent text before the abstract or authors.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if not pdf.pages:
                return ""
            first_page = pdf.pages[0].extract_text() or ""
    except Exception:
        return ""

    lines = [l.strip() for l in first_page.split("\n") if l.strip()]
    if not lines:
        return ""

    # Skip lines that look like headers/footers (very short, page numbers, dates)
    skip = re.compile(
        r"^(page\s*\d|arxiv|preprint|submitted|accepted|doi:|http|©|\d{4}$)",
        re.IGNORECASE,
    )

    title_parts = []
    for line in lines:
        if skip.match(line):
            continue
        # Stop at "Abstract", author lines, etc.
        if re.match(r"^(Abstract|ABSTRACT|Keywords|Introduction|Author)", line):
            break
        # Author-like lines: contain @, university keywords, or camelCase names
        if "@" in line or re.search(r"\d{5,}", line):
            break
        # Detect author name patterns: "FirstLast, FirstLast" or "First Last,"
        # CamelCase concatenated names like "AlbertQ.Jiang,Alexandre..."
        if re.search(r"[a-z][A-Z][a-z].*[A-Z][a-z]", line) and "," in line:
            break
        # Names with asterisks (corresponding author markers)
        if "∗" in line or "†" in line:
            break
        title_parts.append(line)
        # Titles are usually 1-2 lines
        if len(title_parts) >= 2:
            break

    title = " ".join(title_parts).strip()
    # Clean up common artifacts
    title = re.sub(r"\s+", " ", title)
    # Remove trailing numbers/markers
    title = re.sub(r"\s*[\d∗†]+$", "", title)
    return title if len(title) > 5 else ""


def extract_references(raw_text: str) -> str:
    """Extract the references/bibliography section from raw PDF text."""
    text = raw_text.replace("\x00", "")
    for pattern in [
        r"\n\s*References\s*\n",
        r"\n\s*REFERENCES\s*\n",
        r"\n\s*Bibliography\s*\n",
        r"\n\s*BIBLIOGRAPHY\s*\n",
    ]:
        match = re.search(pattern, text)
        if match:
            return text[match.start():].strip()
    return ""
