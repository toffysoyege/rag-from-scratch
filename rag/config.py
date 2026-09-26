import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

DATA_DIR = "data"
DB_DIR = "chroma_db"
COLLECTION_NAME = "cellfix_docs"

# The OpenAI client automatically reads OPENAI_API_KEY and (optionally)
# OPENAI_BASE_URL from the environment, so the same code works for Ollama.
client = OpenAI()
