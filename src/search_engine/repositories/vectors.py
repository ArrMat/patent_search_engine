"""Qdrant: vectors plus the minimal payload needed to filter and list results."""

from qdrant_client import QdrantClient, models

from search_engine.config import (
    BM25_MODEL,
    COLLECTION,
    COLLECTION_PREFIX,
    DENSE_MODEL,
    PREFETCH_FACTOR,
    PREFETCH_MIN,
    QDRANT_URL,
)
from search_engine.models import Patent, SearchFilters

RESULT_FIELDS = ["doc_number", "title", "classification"]


class VectorRepository:
    def __init__(self, url: str = QDRANT_URL) -> None:
        self._client = QdrantClient(url=url)

    def ensure_collection(self) -> bool:
        """Create the collection and its payload indexes, dropping collections built with
        other models. True if it was just created."""
        c = self._client
        if c.collection_exists(COLLECTION):
            return False
        for old in c.get_collections().collections:
            if old.name.startswith(COLLECTION_PREFIX):
                c.delete_collection(old.name)
        c.create_collection(
            COLLECTION,
            vectors_config={
                "dense": models.VectorParams(
                    size=c.get_embedding_size(DENSE_MODEL), distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        # Payload indexes are what let Qdrant filter *before* scoring vectors.
        for field in ("classification_prefixes", "title_norm"):
            c.create_payload_index(COLLECTION, field, models.PayloadSchemaType.KEYWORD)
        for field in ("title", "abstract"):
            c.create_payload_index(
                COLLECTION,
                field,
                models.TextIndexParams(
                    type=models.TextIndexType.TEXT,
                    tokenizer=models.TokenizerType.WORD,
                    lowercase=True,
                ),
            )
        return True

    def upsert(self, patents: list[Patent]) -> None:
        self._client.upload_points(
            COLLECTION,
            [
                models.PointStruct(
                    id=int(p.doc_number),
                    vector={
                        # Dense: title + abstract (fits the model's 512-token window).
                        # With neither, the claims (truncated by the model) stand in.
                        "dense": models.Document(
                            text="\n".join(filter(None, [p.title, p.abstract]))
                            or "\n".join(p.claims),
                            model=DENSE_MODEL,
                        ),
                        # BM25 has no length limit, so it also covers the claims.
                        "bm25": models.Document(
                            text="\n".join(filter(None, [p.title, p.abstract, *p.claims])),
                            model=BM25_MODEL,
                        ),
                    },
                    payload={
                        "doc_number": p.doc_number,
                        "title": p.title,
                        "abstract": p.abstract,
                        "classification": p.classification,
                        "title_norm": _norm(p.title) if p.title else None,
                        # "B60B104FI" -> ["B", "B6", "B60", "B60B", ...]: turns a prefix
                        # constraint into an exact keyword match, which is indexable.
                        "classification_prefixes": [
                            p.classification[:i]
                            for i in range(1, len(p.classification or "") + 1)
                        ],
                    },
                )
                for p in patents
            ],
            wait=True,
        )

    def delete(self, doc_numbers: list[str]) -> None:
        self._client.delete(
            COLLECTION, models.PointIdsList(points=[int(d) for d in doc_numbers]), wait=True
        )

    def search(
        self, query: str, filters: SearchFilters, hybrid: bool, limit: int
    ) -> list[models.ScoredPoint]:
        flt = _to_filter(filters)
        dense = models.Document(text=query, model=DENSE_MODEL)
        if not hybrid:
            return self._client.query_points(
                COLLECTION,
                query=dense,
                using="dense",
                query_filter=flt,
                limit=limit,
                with_payload=RESULT_FIELDS,
            ).points
        # Filters go inside each prefetch so both rankers only score matching patents,
        # then reciprocal rank fusion merges the two candidate lists.
        candidates = max(limit * PREFETCH_FACTOR, PREFETCH_MIN)
        return self._client.query_points(
            COLLECTION,
            prefetch=[
                models.Prefetch(query=dense, using="dense", filter=flt, limit=candidates),
                models.Prefetch(
                    query=models.Document(text=query, model=BM25_MODEL),
                    using="bm25",
                    filter=flt,
                    limit=candidates,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=RESULT_FIELDS,
        ).points


def _norm(title: str) -> str:
    return " ".join(title.casefold().split())


def _to_filter(f: SearchFilters) -> models.Filter | None:
    must: list[models.Condition] = []
    if f.classification_prefixes:  # any of the prefixes
        must.append(
            models.FieldCondition(
                key="classification_prefixes",
                match=models.MatchAny(any=[p.strip().upper() for p in f.classification_prefixes]),
            )
        )
    for kw in f.keywords:  # every keyword, each in title OR abstract (whole words)
        must.append(
            models.Filter(
                should=[
                    models.FieldCondition(key=k, match=models.MatchText(text=kw))
                    for k in ("title", "abstract")
                ]
            )
        )
    if f.title:  # exact title, ignoring case and extra whitespace
        must.append(
            models.FieldCondition(key="title_norm", match=models.MatchValue(value=_norm(f.title)))
        )
    return models.Filter(must=must) if must else None
