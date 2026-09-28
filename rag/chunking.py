from pathlib import Path


def load_documents(data_dir):
    """Read every .md and .txt file in data_dir."""
    docs = []
    for path in sorted(Path(data_dir).iterdir()):
        if path.suffix in {".md", ".txt"}:
            docs.append({"source": path.name, "text": path.read_text(encoding="utf-8")})
    return docs


def chunk_text(text, chunk_size=800, overlap=150):
    """Split text into chunks of roughly chunk_size characters.

    Splits on paragraph boundaries where possible and carries the last
    paragraph(s) forward so neighbouring chunks overlap.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Break up any single paragraph that is longer than chunk_size.
    pieces = []
    for p in paragraphs:
        if len(p) <= chunk_size:
            pieces.append(p)
        else:
            step = chunk_size - overlap
            for i in range(0, len(p), step):
                pieces.append(p[i:i + chunk_size])

    chunks, current = [], []
    for piece in pieces:
        if current and len("\n\n".join(current + [piece])) > chunk_size:
            chunks.append("\n\n".join(current))
            # Carry trailing pieces forward as overlap.
            carry = []
            for prev in reversed(current):
                if len("\n\n".join([prev] + carry)) > overlap:
                    break
                carry.insert(0, prev)
            current = carry
        current.append(piece)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


if __name__ == "__main__":
    for doc in load_documents("data"):
        chunks = chunk_text(doc["text"], chunk_size=300, overlap=60)
        print(f"\n=== {doc['source']}: {len(chunks)} chunks ===")
        for i, c in enumerate(chunks):
            print(f"--- chunk {i} ({len(c)} chars) ---\n{c}\n")
