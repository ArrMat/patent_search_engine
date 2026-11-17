"""Stage 2 of search: a cross-encoder rescoring the candidates stage 1 found.

Unlike the bi-encoder in Qdrant, it reads the query and a patent's text together, so it
is more precise but nothing can be precomputed: it only runs on a short list of candidates.
"""

from functools import cached_property

from fastembed.rerank.cross_encoder import TextCrossEncoder

from search_engine.config import RERANK_MODEL


class Reranker:
    def __init__(self, model: str = RERANK_MODEL) -> None:
        self.model_name = model

    @cached_property
    def _model(self) -> TextCrossEncoder:
        return TextCrossEncoder(self.model_name)  # loaded (and downloaded) on first use

    def score(self, query: str, texts: list[str]) -> list[float]:
        """One relevance score per text; higher is more relevant. Not bounded."""
        return list(self._model.rerank(query, texts))
