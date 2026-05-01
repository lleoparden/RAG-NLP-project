# app/ingestion/parser.py
import fitz  # PyMuPDF
from pathlib import Path
from docx import Document


def parse_pdf(file_path: str) -> list[dict]:
    """Extract text page-by-page from a PDF, preserving metadata."""
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:  # skip empty pages
            pages.append({
                "text": text,
                "metadata": {
                    "source": Path(file_path).name,
                    "page": i + 1,
                    "type": "pdf"
                }
            })
    return pages


def parse_docx(file_path: str) -> list[dict]:
    """Extract paragraphs from a Word document."""
    doc = Document(file_path)
    paragraphs = []
    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text:
            paragraphs.append({
                "text": text,
                "metadata": {
                    "source": Path(file_path).name,
                    "paragraph": i + 1,
                    "type": "docx"
                }
            })
    return paragraphs


def parse_file(file_path: str) -> list[dict]:
    """Dispatcher — routes by extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext in (".docx", ".doc"):
        return parse_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")