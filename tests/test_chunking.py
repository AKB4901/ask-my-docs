from app.rag.chunking import chunk_document


def test_short_document_is_single_chunk():
    chunks = chunk_document("d", "one two three", chunk_size=900, chunk_overlap=150)
    assert len(chunks) == 1
    assert chunks[0].doc_id == "d"
    assert chunks[0].chunk_id == "d::0"


def test_chunks_respect_size():
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(20))
    chunks = chunk_document("d", text, chunk_size=400, chunk_overlap=80)
    assert len(chunks) > 1
    # Allow a little slack for overlap tails but nothing pathological.
    assert all(len(c.text) <= 400 + 200 for c in chunks)


def test_giant_paragraph_is_hard_split():
    text = "x" * 5000
    chunks = chunk_document("d", text, chunk_size=900, chunk_overlap=150)
    assert len(chunks) >= 5
    assert all(len(c.text) <= 900 for c in chunks)


def test_overlap_must_be_smaller_than_size():
    try:
        chunk_document("d", "text", chunk_size=100, chunk_overlap=100)
    except ValueError:
        return
    raise AssertionError("expected ValueError for overlap >= size")


def test_chunk_ids_are_sequential():
    text = "\n\n".join(f"Para {i} " + "w " * 60 for i in range(6))
    chunks = chunk_document("doc", text, chunk_size=300, chunk_overlap=50)
    ids = [c.chunk_id for c in chunks]
    assert ids == [f"doc::{i}" for i in range(len(chunks))]
