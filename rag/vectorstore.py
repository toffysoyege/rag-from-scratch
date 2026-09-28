import chromadb

from rag.config import COLLECTION_NAME, DB_DIR
from rag.embeddings import embed


def get_collection(reset=False):
    db = chromadb.PersistentClient(path=DB_DIR)
    if reset:
        try:
            db.delete_collection(COLLECTION_NAME)
        except ValueError:
            pass  # collection didn't exist yet
    # Chroma defaults to L2 distance. We ask for cosine instead, because
    # nomic-embed-text vectors are not normalized, so L2 and cosine can
    # rank results differently.
    return db.get_or_create_collection(
        COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine"}},
    )


def search(query, k=4):
    """Return the k chunks most similar to the query."""
    collection = get_collection()
    results = collection.query(query_embeddings=embed(query), n_results=k)
    return [
        {"text": doc, "source": meta["source"], "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]
