import pytest

from app.rag.loaders import (
    EmptyDocument,
    UnsupportedFileType,
    load_upload,
)


def test_txt_upload():
    text = load_upload("notes.txt", b"This is a plain text file with enough content.")
    assert "plain text file" in text


def test_md_upload():
    text = load_upload("readme.md", b"# Heading\n\nSome markdown body content here.")
    assert "markdown body" in text


def test_unsupported_type_rejected():
    with pytest.raises(UnsupportedFileType):
        load_upload("malware.exe", b"MZ\x90\x00binary junk that is long enough")


def test_empty_document_rejected():
    with pytest.raises(EmptyDocument):
        load_upload("empty.txt", b"   ")


def test_oversized_file_rejected():
    big = b"x" * (11 * 1024 * 1024)
    with pytest.raises(ValueError):
        load_upload("huge.txt", big)
