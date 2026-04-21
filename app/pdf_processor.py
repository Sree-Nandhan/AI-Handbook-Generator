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
    """Copy an uploaded file to the uploads directory."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = os.path.basename(file_path)
    dest = os.path.join(UPLOAD_DIR, filename)
    if file_path != dest:
        shutil.copy2(file_path, dest)
    logger.info(f"Saved uploaded file: {filename}")
    return dest


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
    """Normalize whitespace and strip references/bibliography."""
    text = raw_text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    lines = text.split("\n")
    text = "\n".join(line.strip() for line in lines)

    # Strip references section at the end
    for pattern in [
        r"\n\s*References\s*\n",
        r"\n\s*REFERENCES\s*\n",
        r"\n\s*Bibliography\s*\n",
        r"\n\s*BIBLIOGRAPHY\s*\n",
    ]:
        match = re.search(pattern, text)
        if match:
            text = text[: match.start()]
            break

    return text.strip()
