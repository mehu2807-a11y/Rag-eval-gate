"""
Central configuration. Everything here is overridable via environment
variables so you can sweep configurations (chunk size, embedding model,
top_k, etc.) without touching code -- and log each combination as a
separate MLflow run.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Data / index locations ---
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_docs")

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 400))       # tokens (approx, word-based)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# --- Embedding ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", 4))

# --- Generation ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")  # "anthropic" or "openai"
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Eval thresholds (used as the CI quality gate) ---
FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", 0.7))
ANSWER_RELEVANCY_THRESHOLD = float(os.getenv("ANSWER_RELEVANCY_THRESHOLD", 0.6))

TESTSET_PATH = Path(os.getenv("TESTSET_PATH", BASE_DIR / "eval" / "testset.json"))
EVAL_REPORT_PATH = Path(os.getenv("EVAL_REPORT_PATH", BASE_DIR / "eval" / "last_report.json"))
