from typing import Literal

from pydantic import BaseModel, Field, model_validator

from search_engine.config import DEFAULT_LIMIT, MAX_LIMIT, MAX_RERANK_CANDIDATES


class Patent(BaseModel):
    """One patent application, as stored in data/patent_data_small/*.json."""

    # Only doc_number and claims are guaranteed by the cleaner; the rest may be null.
    doc_number: str
    title: str | None = None
    abstract: str | None = None
    classification: str | None = None
    claims: list[str]
    detailed_description: list[str]
    filename: str | None = None
    bibtex: str | None = None


class SearchFilters(BaseModel):
    classification_prefixes: list[str] = []  # any of, e.g. ["B60B"]
    keywords: list[str] = []  # all must appear in the title or abstract
    title: str | None = None  # exact title, case-insensitive


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    hybrid: bool = True  # True: dense + BM25 fused with RRF. False: dense only.
    limit: int = Field(DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)
    # Stage 2 on the top `rerank_candidates` from the search above. "cross_encoder": order
    # by the cross-encoder alone. "blend": RRF of the stage-1 order and the cross-encoder order.
    rerank: Literal["none", "cross_encoder", "blend"] = "none"
    rerank_candidates: int = Field(
        MAX_RERANK_CANDIDATES, ge=1, le=MAX_RERANK_CANDIDATES
    )
    filters: SearchFilters = SearchFilters()

    @model_validator(mode="after")
    def _limit_fits_candidates(self) -> "SearchRequest":
        if self.rerank != "none" and self.limit > self.rerank_candidates:
            raise ValueError("with rerank, limit must be <= rerank_candidates")
        return self


class SearchResult(BaseModel):
    doc_number: str
    title: str | None = None
    classification: str | None = None
    # cross_encoder: model score. blend: RRF. Otherwise cosine (hybrid=False) or RRF.
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    took_ms: float
