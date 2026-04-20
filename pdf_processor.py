import os
import re
import pdfplumber
from config import UPLOAD_DIR


def save_uploaded_file(file_path: str) -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    filename = os.path.basename(file_path)
    dest = os.path.join(UPLOAD_DIR, filename)
    if file_path != dest:
        import shutil
        shutil.copy2(file_path, dest)
    return dest


def extract_text_from_pdf(pdf_path: str) -> str:
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    raw_text = "\n\n".join(text_parts)
    return clean_extracted_text(raw_text)


def clean_extracted_text(raw_text: str) -> str:
    text = raw_text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    lines = text.split("\n")
    cleaned_lines = [line.strip() for line in lines]
    text = "\n".join(cleaned_lines)
    text = text.strip()
    return text


def extract_text_from_multiple_pdfs(pdf_paths: list[str]) -> dict[str, str]:
    results = {}
    for path in pdf_paths:
        filename = os.path.basename(path)
        results[filename] = extract_text_from_pdf(path)
    return results
