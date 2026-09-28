from rag.chunking import chunk_text, load_documents
from rag.config import DATA_DIR
from rag.embeddings import embed
from rag.vectorstore import get_collection


def ingest():
    collection = get_collection(reset=True)
    ids, texts, metadatas = [], [], []

    for doc in load_documents(DATA_DIR):
        for i, chunk in enumerate(chunk_text(doc["text"])):
            ids.append(f"{doc['source']}::{i}")
            texts.append(chunk)
            metadatas.append({"source": doc["source"], "chunk": i})

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embed(texts),
        metadatas=metadatas,
    )
    print(f"Indexed {len(ids)} chunks into '{collection.name}'.")


if __name__ == "__main__":
    ingest()
