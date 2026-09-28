from rag.chunking import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_chunks_respect_size_roughly():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 30 for i in range(20))
    chunks = chunk_text(text, chunk_size=400, overlap=80)
    assert len(chunks) > 1
    assert all(len(c) <= 400 + 200 for c in chunks)


def test_long_paragraph_is_split():
    chunks = chunk_text("x" * 2000, chunk_size=500, overlap=100)
    assert len(chunks) >= 4
