"""HTTP layer. Parses requests, calls the service, shapes responses. No logic."""

from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from search_engine.models import Patent, SearchRequest, SearchResponse
from search_engine.repositories.patents import PatentRepository
from search_engine.repositories.vectors import VectorRepository
from search_engine.rerank import Reranker
from search_engine.services.search import SearchService

router = APIRouter()


@lru_cache
def get_service() -> SearchService:
    return SearchService(PatentRepository(), VectorRepository(), Reranker())


Service = Annotated[SearchService, Depends(get_service)]


@router.post("/search")
def search(req: SearchRequest, service: Service) -> SearchResponse:
    return service.search(req)


@router.get("/patents/{doc_number}")
def get_patent(doc_number: str, service: Service) -> Patent:
    patent = service.get(doc_number)
    if patent is None:
        raise HTTPException(404, "patent not found")
    return patent
