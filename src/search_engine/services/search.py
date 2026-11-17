"""Business logic. Knows nothing about HTTP, SQL or Qdrant's API."""

import json
from pathlib import Path
from time import perf_counter

from search_engine.config import CLEAN_DIR, INGEST_BATCH_SIZE, REMOVALS, RERANK_WORDS, RRF_K
from search_engine.models import (
    Patent,
    SearchFilters,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from search_engine.repositories.patents import PatentRepository
from search_engine.repositories.vectors import VectorRepository
from search_engine.rerank import Reranker


def read_removals(path: Path) -> list[str]:
    """One doc number per line; blank lines and `#` comments ignored. No file: none."""
    if not path.exists():
        return []
    lines = (line.split("#", 1)[0].strip() for line in path.read_text("utf-8").splitlines())
    return sorted({line for line in lines if line})


class SearchService:
    def __init__(
        self, patents: PatentRepository, vectors: VectorRepository, reranker: Reranker
    ) -> None:
        self._patents = patents
        self._vectors = vectors
        self._reranker = reranker

    def warm_up(self) -> None:
        """Load (and on first run download) every model now, so no request pays for it.
        Ingest alone doesn't: it only embeds when patents are pending."""
        self._vectors.search("warm up", SearchFilters(), hybrid=True, limit=1)
        self._reranker.score("warm up", ["warm up"])

    def ingest(
        self,
        data_dir: Path = CLEAN_DIR,
        removals: Path = REMOVALS,
        batch_size: int = INGEST_BATCH_SIZE,
    ) -> None:
        """JSON -> Postgres -> Qdrant. Safe to re-run: resumes where it stopped.

        Patents in the removal list are deleted and never re-added, even though their
        clean (and raw) records are kept.
        """
        if self._vectors.ensure_collection():
            self._patents.reset_indexed()  # empty Qdrant: everything must be re-embedded
        removed = read_removals(removals)
        if removed:
            self._vectors.delete(removed)  # Qdrant first: if Postgres then fails, a re-run retries both
            self._patents.remove(removed)
        skip = set(removed)
        self._patents.add(
            p
            for f in sorted(data_dir.glob("*.jsonl"))
            for line in f.read_text(encoding="utf-8").splitlines()
            if (p := Patent(**json.loads(line))).doc_number not in skip
        )
        while batch := self._patents.pending(batch_size):
            self._vectors.upsert(batch)
            self._patents.mark_indexed([p.doc_number for p in batch])

    def get(self, doc_number: str) -> Patent | None:
        return self._patents.get(doc_number)

    def search(self, req: SearchRequest) -> SearchResponse:
        start = perf_counter()
        rerank = req.rerank != "none"
        n = req.rerank_candidates if rerank else req.limit  # limit <= candidates: validated
        points = self._vectors.search(req.query, req.filters, req.hybrid, n)
        results = [SearchResult(**p.payload, score=p.score) for p in points]
        if rerank and results:
            results = self._rerank(req.query, results, req.rerank == "blend")[: req.limit]
        took_ms = (perf_counter() - start) * 1000
        return SearchResponse(results=results, took_ms=round(took_ms, 2))

    def _rerank(self, query: str, results: list[SearchResult], blend: bool) -> list[SearchResult]:
        """`results` in stage-1 order. blend: RRF of the stage-1 rank and the
        cross-encoder rank, so a confident stage 1 isn't overruled by a confused model."""
        patents = self._patents.get_many([r.doc_number for r in results])
        texts = [_rerank_text(patents.get(r.doc_number), r) for r in results]
        scores = self._reranker.score(query, texts)
        by_model = sorted(range(len(results)), key=lambda i: scores[i], reverse=True)
        model_rank = {i: rank for rank, i in enumerate(by_model)}
        for i, r in enumerate(results):
            r.score = 1 / (RRF_K + i + 1) + 1 / (RRF_K + model_rank[i] + 1) if blend else scores[i]
        return sorted(results, key=lambda r: r.score, reverse=True)


def _rerank_text(patent: Patent | None, result: SearchResult) -> str:
    """Title + abstract + first claim (text the meaning vector never saw), cut to
    RERANK_WORDS: cross-encoder cost grows faster than linearly with length."""
    if patent is None:  # in Qdrant but not Postgres (the stores drifted): title only
        return result.title or ""
    text = "\n".join(filter(None, [patent.title, patent.abstract, *patent.claims[:1]]))
    return " ".join(text.split()[:RERANK_WORDS])
