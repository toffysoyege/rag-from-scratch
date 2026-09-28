from rag.llm import chat
from rag.vectorstore import search

SYSTEM_PROMPT = """You are a helpful assistant for CellFix.
Answer the user's question using ONLY the context provided below.
If the answer is not in the context, say: "I don't know based on the available documents."
Cite the source file for each fact in square brackets, e.g. [products.md]."""


def build_context(hits):
    return "\n\n---\n\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)


def answer(question, k=4):
    hits = search(question, k=k)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{build_context(hits)}\n\nQuestion: {question}"},
    ]
    return chat(messages), hits
