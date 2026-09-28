from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Reuse our own settings so this works with both OpenAI and Ollama.
from rag.config import CHAT_MODEL, DATA_DIR, EMBED_MODEL, client

# 1. Load
docs = [
    Document(page_content=p.read_text(encoding="utf-8"), metadata={"source": p.name})
    for p in sorted(Path(DATA_DIR).glob("*.md"))
]

# 2. Chunk
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(docs)

# 3. Embed + store
# check_embedding_ctx_length=False makes LangChain send plain text instead of
# OpenAI token IDs, which Ollama can't read.
embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    base_url=str(client.base_url),
    check_embedding_ctx_length=False,
)
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    collection_name="cellfix_langchain",
    collection_metadata={"hnsw:space": "cosine"},
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 4. Prompt + LLM
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the context. If unsure, say you don't know. "
               "Cite sources in [brackets].\n\nContext:\n{context}"),
    ("human", "{question}"),
])
llm = ChatOpenAI(model=CHAT_MODEL, base_url=str(client.base_url), temperature=0)


def ask(question):
    hits = retriever.invoke(question)
    context = "\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in hits)
    return llm.invoke(prompt.format_messages(context=context, question=question)).content


if __name__ == "__main__":
    print(ask("How long is CellFix's repair warranty?"))
