import math

from rag.config import EMBED_MODEL, client


def embed(texts, batch_size=100):
    """Return one embedding vector per input text."""
    if isinstance(texts, str):
        texts = [texts]
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

if __name__ == "__main__":
    query = "Do you use genuine Apple parts when fixing my iPhone?"
    candidates = [
        "OEM parts are original components manufactured by or for the device manufacturer.",
        "A protective case designed to help protect smartphones from everyday drops. Price: $18.99.",
        "Technicians receive scheduled breaks during their shifts.",
    ]
    q_vec, *c_vecs = embed([query] + candidates)
    print(f"Vector length: {len(q_vec)}\n")
    for text, vec in zip(candidates, c_vecs):
        print(f"{cosine_similarity(q_vec, vec):.3f}  {text}")
