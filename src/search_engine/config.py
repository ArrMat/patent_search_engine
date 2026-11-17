"""Project-wide settings and defaults. Env vars override where noted."""

import os
from pathlib import Path

# --- Services (env-overridable) ---
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/patents"
)
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
# Test-only instances (docker compose: postgres-test, qdrant-test). Fixtures wipe them.
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/patents_test?connect_timeout=2",
)
TEST_QDRANT_URL = os.environ.get("TEST_QDRANT_URL", "http://localhost:6335")

# --- API server ---
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
RELOAD = os.environ.get("RELOAD", "1") == "1"

# --- Models ---
DENSE_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dims, truncates at 512 tokens
BM25_MODEL = "Qdrant/bm25"
RERANK_MODEL = os.environ.get("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")
# Named after the models: switching one creates a fresh collection, so everything is re-embedded.
COLLECTION_PREFIX = "patents"
COLLECTION = f"{COLLECTION_PREFIX}_{DENSE_MODEL}_{BM25_MODEL}".replace("/", "_")

# --- Data paths ---
DATA = Path(__file__).parents[2] / "data"
RAW_DIR = DATA / "raw"
CLEAN_DIR = DATA / "clean"  # written by `uv run clean-patents`
REMOVALS = DATA / "removals.txt"  # doc numbers to take out of the index

# --- Ingest ---
INGEST_BATCH_SIZE = 64

# --- Search ---
DEFAULT_LIMIT = 10
MAX_LIMIT = 100
# Candidates taken from stage 1 and rescored by the cross-encoder (default and maximum).
# Cost grows linearly: ~0.1 s per candidate on CPU with the default model.
MAX_RERANK_CANDIDATES = 100
RERANK_WORDS = 256  # per candidate; fits the first claim for most patents (~1.5 s on CPU)
RRF_K = 60  # the usual reciprocal-rank-fusion constant
# Hybrid: each ranker (dense, BM25) hands max(limit * factor, min) candidates to the fusion.
PREFETCH_FACTOR = 5
PREFETCH_MIN = 50

# --- Evaluation ---
EVAL_K = 10
QUERY_WORDS = 40  # long enough to identify a patent, short enough to resemble a query
MIN_WORDS = 8  # shorter texts (truncated claims) cannot identify one patent: skipped
