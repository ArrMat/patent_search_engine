"""API, service and Qdrant query-building tests. Postgres and Qdrant are mocked: no docker needed."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest
from fastapi.testclient import TestClient

from search_engine import app
from search_engine.clean import clean_record
from search_engine.config import MAX_RERANK_CANDIDATES
from search_engine.controllers.search import get_service
from search_engine.models import Patent, SearchFilters
from search_engine.repositories.patents import PatentRepository
from search_engine.repositories.vectors import VectorRepository, _to_filter
from search_engine.rerank import Reranker
from search_engine.services.search import SearchService, _rerank_text

PATENT = Patent(doc_number="1", title="  Spoke ", abstract="A rim.", classification="B60B104FI",
                claims=["c1", "c2"], detailed_description=["d"], filename="f", bibtex="b")
POINT = SimpleNamespace(
    payload={"doc_number": "1", "title": "SPOKE", "classification": "B60B104FI"}, score=0.9
)


@pytest.fixture
def patent_repo():
    return create_autospec(PatentRepository, instance=True)


@pytest.fixture
def vector_repo():
    return create_autospec(VectorRepository, instance=True)


@pytest.fixture
def reranker():
    return create_autospec(Reranker, instance=True)


@pytest.fixture
def service(patent_repo, vector_repo, reranker):
    return SearchService(patent_repo, vector_repo, reranker)


@pytest.fixture
def client(service):
    app.dependency_overrides[get_service] = lambda: service
    yield TestClient(app)  # not entered as a context manager: the lifespan (real ingest) never runs
    app.dependency_overrides.clear()


@pytest.fixture
def qdrant(monkeypatch):
    """The fake QdrantClient instance a VectorRepository() built in the test will hold."""
    cls = Mock()
    monkeypatch.setattr("search_engine.repositories.vectors.QdrantClient", cls)
    cls.return_value.query_points.return_value.points = [POINT]
    return cls.return_value


# --- HTTP layer + service -------------------------------------------------------------


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_search_returns_scored_results(client, vector_repo):
    vector_repo.search.return_value = [POINT]
    r = client.post("/search", json={"query": "wheel"})
    assert r.status_code == 200
    body = r.json()
    assert body["results"] == [{**POINT.payload, "score": 0.9}]
    assert body["took_ms"] >= 0
    vector_repo.search.assert_called_once_with("wheel", SearchFilters(), True, 10)


def test_search_returns_results_with_null_title_and_classification(client, vector_repo):
    payload = {"doc_number": "2", "title": None, "classification": None}
    vector_repo.search.return_value = [SimpleNamespace(payload=payload, score=0.5)]
    r = client.post("/search", json={"query": "wheel"})
    assert r.status_code == 200 and r.json()["results"] == [{**payload, "score": 0.5}]


def test_search_forwards_options_and_filters(client, vector_repo):
    vector_repo.search.return_value = []
    flt = {"classification_prefixes": ["b60b"], "keywords": ["spoke"], "title": "x"}
    r = client.post("/search", json={"query": "q", "hybrid": False, "limit": 5, "filters": flt})
    assert r.json()["results"] == []
    vector_repo.search.assert_called_once_with("q", SearchFilters(**flt), False, 5)


def test_rerank_rescores_candidates_and_cuts_to_limit(client, patent_repo, vector_repo, reranker):
    points = [SimpleNamespace(payload={"doc_number": d, "title": f"T{d}", "classification": None},
                              score=0.1) for d in "123"]
    vector_repo.search.return_value = points
    patent_repo.get_many.return_value = {  # "3" is missing from Postgres: falls back to its title
        "1": PATENT, "2": PATENT.model_copy(update={"doc_number": "2", "abstract": None}),
    }
    reranker.score.return_value = [0.5, 2.0, 1.0]

    r = client.post("/search", json={"query": "q", "limit": 2, "rerank": "cross_encoder"})

    assert [(x["doc_number"], x["score"]) for x in r.json()["results"]] == [("2", 2.0), ("3", 1.0)]
    vector_repo.search.assert_called_once_with("q", SearchFilters(), True, MAX_RERANK_CANDIDATES)  # default candidates
    reranker.score.assert_called_once_with("q", ["Spoke A rim. c1", "Spoke c1", "T3"])


def test_rerank_candidates_is_customizable(client, vector_repo):
    vector_repo.search.return_value = []
    body = {"query": "q", "limit": 5, "rerank": "blend", "rerank_candidates": 8}
    assert client.post("/search", json=body).status_code == 200
    vector_repo.search.assert_called_once_with("q", SearchFilters(), True, 8)


def test_rerank_text_is_cut_to_word_budget(monkeypatch):
    monkeypatch.setattr("search_engine.services.search.RERANK_WORDS", 3)
    long = PATENT.model_copy(update={"abstract": "one two three four"})
    assert _rerank_text(long, None) == "Spoke one two"


def test_blend_fuses_stage_one_rank_with_model_rank(client, patent_repo, vector_repo, reranker):
    vector_repo.search.return_value = [
        SimpleNamespace(payload={"doc_number": d, "title": None, "classification": None}, score=0.1)
        for d in "123"
    ]
    patent_repo.get_many.return_value = {}
    reranker.score.return_value = [0.0, 0.5, 9.0]  # model ranks: 3, 2, 1

    r = client.post("/search", json={"query": "q", "rerank": "blend"}).json()["results"]

    # stage-1 ranks 1,2,3 + model ranks 3,2,1: "2" is 2nd in both, the others tie
    rrf = {"1": 1 / 61 + 1 / 63, "2": 2 / 62, "3": 1 / 63 + 1 / 61}
    assert [x["doc_number"] for x in r][0] in "13" and r[-1]["doc_number"] == "2"
    assert {x["doc_number"]: x["score"] for x in r} == pytest.approx(rrf)


def test_rerank_with_no_candidates_skips_the_model(client, vector_repo, reranker):
    vector_repo.search.return_value = []
    r = client.post("/search", json={"query": "q", "rerank": "cross_encoder"})
    assert r.json()["results"] == []
    reranker.score.assert_not_called()


def test_search_without_rerank_never_calls_the_model(client, vector_repo, reranker):
    vector_repo.search.return_value = [POINT]
    client.post("/search", json={"query": "q"})
    reranker.score.assert_not_called()


@pytest.mark.parametrize("body", [{"query": ""}, {}, {"query": "q", "limit": 0}, {"query": "q", "limit": 101},
                                  {"query": "q", "rerank": "llm"},
                                  {"query": "q", "rerank": "blend", "rerank_candidates": MAX_RERANK_CANDIDATES + 1},
                                  {"query": "q", "rerank": "blend", "rerank_candidates": 0},
                                  {"query": "q", "rerank": "blend", "rerank_candidates": 5, "limit": 30}])
def test_search_rejects_invalid_request(client, vector_repo, body):
    assert client.post("/search", json=body).status_code == 422
    vector_repo.search.assert_not_called()


def test_get_patent(client, patent_repo):
    patent_repo.get.return_value = PATENT
    r = client.get("/patents/1")
    assert r.status_code == 200 and r.json() == PATENT.model_dump()
    patent_repo.get.assert_called_once_with("1")


def test_get_patent_not_found(client, patent_repo):
    patent_repo.get.return_value = None
    assert client.get("/patents/nope").status_code == 404


# --- ingest ---------------------------------------------------------------------------


def test_ingest_loads_files_then_indexes_pending(service, patent_repo, vector_repo, tmp_path):
    (tmp_path / "a.jsonl").write_text(PATENT.model_dump_json() + "\n", "utf-8")
    vector_repo.ensure_collection.return_value = True
    patent_repo.pending.side_effect = [[PATENT], []]

    service.ingest(tmp_path, tmp_path / "removals.txt")

    patent_repo.reset_indexed.assert_called_once_with()  # new collection: re-embed everything
    assert list(patent_repo.add.call_args.args[0]) == [PATENT]
    vector_repo.upsert.assert_called_once_with([PATENT])
    patent_repo.mark_indexed.assert_called_once_with(["1"])


def test_ingest_accepts_every_record_the_cleaner_keeps(service, patent_repo, vector_repo, tmp_path):
    rec, _ = clean_record({"doc_number": "1", "claims": ["c1"]})  # everything else null
    (tmp_path / "a.jsonl").write_text(json.dumps(rec) + "\n", "utf-8")
    vector_repo.ensure_collection.return_value = False
    patent_repo.pending.return_value = []

    service.ingest(tmp_path, tmp_path / "removals.txt")

    (stored,) = patent_repo.add.call_args.args[0]
    assert stored.doc_number == "1" and stored.title is None and stored.bibtex is None


def test_ingest_keeps_index_state_when_collection_exists(service, patent_repo, vector_repo, tmp_path):
    vector_repo.ensure_collection.return_value = False
    patent_repo.pending.return_value = []

    service.ingest(tmp_path, tmp_path / "removals.txt")

    patent_repo.reset_indexed.assert_not_called()
    vector_repo.upsert.assert_not_called()


def test_ingest_removes_listed_patents_and_never_re_adds_them(service, patent_repo, vector_repo, tmp_path):
    other = PATENT.model_copy(update={"doc_number": "2"})
    (tmp_path / "a.jsonl").write_text(PATENT.model_dump_json() + "\n" + other.model_dump_json() + "\n", "utf-8")
    (tmp_path / "removals.txt").write_text("# withdrawn\n1\n\n3  # not stored, harmless\n1\n", "utf-8")
    vector_repo.ensure_collection.return_value = False
    patent_repo.pending.return_value = []

    service.ingest(tmp_path, tmp_path / "removals.txt")

    vector_repo.delete.assert_called_once_with(["1", "3"])
    patent_repo.remove.assert_called_once_with(["1", "3"])
    assert list(patent_repo.add.call_args.args[0]) == [other]


def test_ingest_without_removal_list_removes_nothing(service, patent_repo, vector_repo, tmp_path):
    vector_repo.ensure_collection.return_value = False
    patent_repo.pending.return_value = []

    service.ingest(tmp_path, tmp_path / "removals.txt")

    vector_repo.delete.assert_not_called()
    patent_repo.remove.assert_not_called()


# --- Qdrant query building ------------------------------------------------------------


def test_no_filters_means_no_qdrant_filter():
    assert _to_filter(SearchFilters()) is None


def test_classification_prefixes_are_upper_cased():
    (cond,) = _to_filter(SearchFilters(classification_prefixes=[" b60b ", "A"])).must
    assert cond.key == "classification_prefixes" and cond.match.any == ["B60B", "A"]


def test_every_keyword_must_match_title_or_abstract():
    must = _to_filter(SearchFilters(keywords=["spoke", "rim"])).must
    assert [[(c.key, c.match.text) for c in f.should] for f in must] == [
        [("title", "spoke"), ("abstract", "spoke")],
        [("title", "rim"), ("abstract", "rim")],
    ]


def test_title_filter_ignores_case_and_whitespace():
    (cond,) = _to_filter(SearchFilters(title="  SpOke   Wheel ")).must
    assert cond.key == "title_norm" and cond.match.value == "spoke wheel"


def test_dense_search_queries_only_the_dense_vector(qdrant):
    points = VectorRepository().search("q", SearchFilters(title="spoke"), hybrid=False, limit=7)
    assert points == [POINT]
    kw = qdrant.query_points.call_args.kwargs
    assert kw["using"] == "dense" and kw["limit"] == 7 and "prefetch" not in kw
    assert kw["query_filter"] == _to_filter(SearchFilters(title="spoke"))


def test_hybrid_search_filters_inside_both_prefetches(qdrant):
    flt = SearchFilters(keywords=["rim"])
    points = VectorRepository().search("q", flt, hybrid=True, limit=10)
    assert points == [POINT]
    kw = qdrant.query_points.call_args.kwargs
    assert [p.using for p in kw["prefetch"]] == ["dense", "bm25"]
    assert all(p.filter == _to_filter(flt) and p.limit == 50 for p in kw["prefetch"])
    assert kw["limit"] == 10 and "query_filter" not in kw


def test_upsert_handles_patent_with_only_number_and_claims(qdrant):
    bare = Patent(doc_number="2", claims=["c1", "c2"], detailed_description=[])
    VectorRepository().upsert([bare])
    (point,) = qdrant.upload_points.call_args.args[1]
    assert point.vector["dense"].text == point.vector["bm25"].text == "c1\nc2"
    assert point.payload["title_norm"] is None
    assert point.payload["classification_prefixes"] == []


def test_delete_removes_points_by_doc_number(qdrant):
    VectorRepository().delete(["2", "10"])
    assert qdrant.delete.call_args.args[1].points == [2, 10]


def test_ensure_collection_leaves_existing_collection_alone(qdrant):
    qdrant.collection_exists.return_value = True
    assert VectorRepository().ensure_collection() is False
    qdrant.create_collection.assert_not_called()
    qdrant.create_payload_index.assert_not_called()


def test_ensure_collection_creates_vectors_and_payload_indexes(qdrant):
    qdrant.collection_exists.return_value = False
    qdrant.get_embedding_size.return_value = 384
    qdrant.get_collections.return_value.collections = [
        SimpleNamespace(name="patents"), SimpleNamespace(name="other_app"),
    ]

    assert VectorRepository().ensure_collection() is True

    qdrant.delete_collection.assert_called_once_with("patents")  # old model's, not other apps'

    kw = qdrant.create_collection.call_args.kwargs
    assert kw["vectors_config"]["dense"].size == 384
    assert list(kw["sparse_vectors_config"]) == ["bm25"]
    assert [c.args[1] for c in qdrant.create_payload_index.call_args_list] == [
        "classification_prefixes", "title_norm", "title", "abstract",
    ]


def test_upsert_builds_filterable_payload(qdrant):
    VectorRepository().upsert([PATENT])
    (point,) = qdrant.upload_points.call_args.args[1]
    assert point.id == 1
    assert point.payload["title_norm"] == "spoke"
    assert point.payload["classification_prefixes"][:4] == ["B", "B6", "B60", "B60B"]
    assert point.payload["classification_prefixes"][-1] == "B60B104FI"
