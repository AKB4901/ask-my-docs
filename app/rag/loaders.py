"""
Loaders for uploaded documents.

Turns raw uploaded bytes into plain text the pipeline can index. Supports PDF
(resumes, papers, reports), plus txt and markdown. Kept deliberately small and
dependency-light — PDF text extraction via pypdf, everything else is a decode.
"""

from __future__ import annotations

from io import BytesIO

SUPPORTED_UPLOAD_TYPES = {"pdf", "txt", "md", "markdown"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


class UnsupportedFileType(ValueError):
    pass


class EmptyDocument(ValueError):
    pass


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(BytesIO(data))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(p for p in pages if p)


def load_upload(filename: str, data: bytes) -> str:
    """Extract text from an uploaded file. Raises on unsupported/empty input."""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("File is larger than the 10 MB limit.")

    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix not in SUPPORTED_UPLOAD_TYPES:
        raise UnsupportedFileType(
            f"Unsupported file type '.{suffix}'. Upload a PDF, .txt, or .md file."
        )

    if suffix == "pdf":
        text = _extract_pdf(data)
    else:
        text = data.decode("utf-8", errors="ignore")

    text = text.strip()
    if len(text) < 20:
        raise EmptyDocument(
            "Couldn't read any text from that file. If it's a scanned PDF, it "
            "has no selectable text to index."
        )
    return text
