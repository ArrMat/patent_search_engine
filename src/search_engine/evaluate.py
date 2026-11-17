"""Retrieval evaluation: a measurement, not a test.

Run:  uv run eval-search FILE.jsonl   labeled queries (the main metric)
      uv run eval-search [N]          known-item sanity check, N queries per source
(needs: docker compose up -d, and the index built by starting the app)

Labeled queries: one JSON object per line, relevance graded 1 (related) or 2 (on point),
unlisted patents count as 0. Filters are optional, same shape as POST /search:
  {"query": "bicycle spoke with threaded ends", "relevant": {"20240051333": 2},
   "filters": {"classification_prefixes": ["B60B"]}}
Reported: nDCG@K, recall@K (share of the relevant patents found), MRR@K (first relevant).

Known-item (no file): each patent becomes one query made from its own text; a search is a hit when that
patent comes back in the top K. No hand labels needed. Two query sources:

- claim:       start of the first claim. Verbatim in the BM25 text and absent from the
               dense text, so it flatters the lexical side.
- description: start of the longest description paragraph. Indexed nowhere, so it is
               the fairer one: unseen text about the same invention.

ponytail: queries are the patent's own wording, so scores are an upper bound on what
real user queries would get. Add hand-written queries with expected patents when the
ranking itself (not just "can it find a known patent") needs judging.
"""

import json
import math
import sys
from collections.abc import Callable
from pathlib import Path

from search_engine.config import CLEAN_DIR, MIN_WORDS, QUERY_WORDS, RERANK_MODEL
from search_engine.config import EVAL_K as K
from search_engine.models import SearchRequest
from search_engine.repositories.patents import PatentRepository
from search_engine.repositories.vectors import VectorRepository
from search_engine.rerank import Reranker
from search_engine.services.search import SearchService

SOURCES: dict[str, Callable[[dict], str]] = {
    "claim": lambda r: r["claims"][0],
    "description": lambda r: max(r["detailed_description"], key=len, default=""),
}


def to_query(text: str) -> str | None:
    words = text.split()[:QUERY_WORDS]
    return " ".join(words) if len(words) >= MIN_WORDS else None


def metrics(ranks: list[int | None]) -> dict[str, float]:
    """ranks: 1-based position of the expected patent per query, None when not in the top K."""
    n = len(ranks)
    out = {f"recall@{k}": sum(r is not None and r <= k for r in ranks) / n for k in (1, 5, K)}
    out[f"mrr@{K}"] = sum(1 / r for r in ranks if r) / n
    return out


def graded_metrics(found: list[str], relevant: dict[str, int]) -> dict[str, float]:
    """One labeled query: found = returned doc_numbers in order, relevant = doc -> grade."""
    gains = [relevant.get(d, 0) for d in found[:K]]
    dcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(relevant.values(), reverse=True)[:K]
    idcg = sum((2**g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    first = next((i + 1 for i, g in enumerate(gains) if g > 0), None)
    return {
        f"ndcg@{K}": dcg / idcg if idcg else 0.0,
        f"recall@{K}": sum(g > 0 for g in gains) / sum(g > 0 for g in relevant.values()),
        f"mrr@{K}": 1 / first if first else 0.0,
    }


MODES = {  # name -> (hybrid, rerank)
    "dense": (False, "none"),
    "hybrid": (True, "none"),
    "rerank": (True, "cross_encoder"),  # hybrid top candidates -> cross-encoder
    "blend": (True, "blend"),  # same model scores as "rerank": served from the cache
}


class CachedReranker(Reranker):
    """rerank and blend score the same (query, candidates): run the model once."""

    def __init__(self) -> None:
        super().__init__()
        self._seen: dict[tuple, list[float]] = {}

    def score(self, query: str, texts: list[str]) -> list[float]:
        key = (query, tuple(texts))
        if key not in self._seen:
            self._seen[key] = super().score(query, texts)
        return self._seen[key]


def sample(items: list, n: int | None) -> list:
    """n items spread evenly over the list (files are sorted by week); None: all."""
    return items if not n or n >= len(items) else items[:: len(items) // n][:n]


def load_labeled(path: Path) -> list[dict]:
    queries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for i, q in enumerate(queries, 1):
        if not q.get("query") or not any(g > 0 for g in q.get("relevant", {}).values()):
            raise SystemExit(f"{path}:{i}: needs a query and at least one relevant patent")
    return queries


def eval_labeled(service: SearchService, queries: list[dict]) -> None:
    print(f"{'mode':<8}{'queries':>8}" + "".join(f"{k:>11}" for k in graded_metrics([], {"x": 1})))
    for mode, (hybrid, rerank) in MODES.items():
        rows = []
        for q in queries:
            req = SearchRequest(
                query=q["query"], hybrid=hybrid, rerank=rerank, limit=K, filters=q.get("filters", {})
            )
            rows.append(graded_metrics([r.doc_number for r in service.search(req).results], q["relevant"]))
        print(f"{mode:<8}{len(rows):>8}" + "".join(f"{sum(r[k] for r in rows) / len(rows):>11.3f}" for k in rows[0]))


def main() -> None:
    # `uv run eval-search queries.jsonl`: labeled queries.
    # `uv run eval-search 50`: known-item, 50 queries per source. Rerank costs ~2 s per query on CPU.
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    service = SearchService(PatentRepository(), VectorRepository(), CachedReranker())
    print(f"rerank model: {RERANK_MODEL}")
    if arg and not arg.isdigit():
        eval_labeled(service, load_labeled(Path(arg)))
        return
    n = int(arg) if arg else None
    records = [
        json.loads(line)
        for f in sorted(CLEAN_DIR.glob("*.jsonl"))
        for line in f.read_text(encoding="utf-8").splitlines()
    ]
    header = None
    for source, pick in SOURCES.items():
        queries = sample([(r["doc_number"], q) for r in records if (q := to_query(pick(r)))], n)
        for mode, (hybrid, rerank) in MODES.items():
            ranks = []
            for doc_number, query in queries:
                req = SearchRequest(query=query, hybrid=hybrid, rerank=rerank, limit=K)
                found = [r.doc_number for r in service.search(req).results]
                ranks.append(found.index(doc_number) + 1 if doc_number in found else None)
            m = metrics(ranks)
            if header is None:
                header = f"{'query source':<13}{'mode':<8}{'queries':>8}" + "".join(f"{k:>11}" for k in m)
                print(header)
            print(f"{source:<13}{mode:<8}{len(ranks):>8}" + "".join(f"{v:>11.3f}" for v in m.values()))
