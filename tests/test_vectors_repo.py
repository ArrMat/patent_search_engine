"""VectorRepository against the test-only Qdrant. Needs: docker compose up -d qdrant-test

Never uses QDRANT_URL: the fixture drops the collection, so it must not reach the dev index.
Embeddings are computed locally by the real models (downloaded on first use, then cached).
"""

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

from search_engine.config import COLLECTION, TEST_QDRANT_URL
from search_engine.models import Patent, SearchFilters
from search_engine.repositories.vectors import VectorRepository

PATENTS = [
    Patent(doc_number="1", title="Wheel Spoke", abstract="A reinforced rim for a bicycle wheel.",
           classification="B60B104FI", claims=["A spoke."], detailed_description=[]),
    Patent(doc_number="2", title="Brake Pad", abstract="Spokes are not involved; friction material.",
           classification="F16D6500", claims=["A pad."], detailed_description=[]),
    Patent(doc_number="3", title="Engine Mount", abstract="A vibration damper for an engine.",
           classification="B60K500", claims=["A mount."], detailed_description=[]),
    # Only what the cleaner guarantees: no title, abstract or classification.
    Patent(doc_number="4", claims=["A wheel with a spoke and a rim."], detailed_description=[]),
]


@pytest.fixture(scope="module")
def repo():
    """A freshly created collection holding PATENTS. Seeded once: the tests only read it."""
    try:
        client = QdrantClient(url=TEST_QDRANT_URL)
        if client.collection_exists(COLLECTION):
            client.delete_collection(COLLECTION)
    except ResponseHandlingException:
        pytest.skip("test Qdrant is down: docker compose up -d qdrant-test")
    repo = VectorRepository(TEST_QDRANT_URL)
    assert repo.ensure_collection() is True
    repo.upsert(PATENTS)
    return repo


@pytest.fixture(params=[False, True], ids=["dense", "hybrid"])
def hybrid(request):
    return request.param


def found(repo, hybrid, query="wheel", **filters) -> set[str]:
    points = repo.search(query, SearchFilters(**filters), hybrid, 10)
    return {p.payload["doc_number"] for p in points}


def test_ensure_collection_reports_existing_collection(repo):
    assert repo.ensure_collection() is False


def test_no_filters_returns_everything_best_first(repo, hybrid):
    points = repo.search("wheel", SearchFilters(), hybrid, 10)
    assert {p.payload["doc_number"] for p in points} == {"1", "2", "3", "4"}
    scores = [p.score for p in points]
    assert scores == sorted(scores, reverse=True)


def test_limit_caps_the_results(repo, hybrid):
    assert len(repo.search("wheel", SearchFilters(), hybrid, 2)) == 2


def test_obvious_query_ranks_its_patent_first(repo, hybrid):
    (top,) = repo.search("brake pad friction material", SearchFilters(), hybrid, 1)
    assert top.payload == {"doc_number": "2", "title": "Brake Pad", "classification": "F16D6500"}


@pytest.mark.parametrize(
    ("prefixes", "expected"),
    [
        (["b60b"], {"1"}),  # case-insensitive
        (["B60"], {"1", "3"}),  # shorter prefix matches more
        (["B60B", "F16D"], {"1", "2"}),  # any of
        (["X"], set()),
    ],
)
def test_classification_prefix_filter(repo, hybrid, prefixes, expected):
    assert found(repo, hybrid, classification_prefixes=prefixes) == expected


@pytest.mark.parametrize(
    ("keywords", "expected"),
    [
        (["spoke"], {"1"}),  # whole word: not "Spokes" (2); title/abstract only: not claims (4)
        (["SPOKE", "rim"], {"1"}),  # one in the title, one in the abstract
        (["spoke", "friction"], set()),  # every keyword is required
    ],
)
def test_keyword_filter(repo, hybrid, keywords, expected):
    assert found(repo, hybrid, keywords=keywords) == expected


def test_title_filter_is_exact_ignoring_case_and_whitespace(repo, hybrid):
    assert found(repo, hybrid, title="  wheel   SPOKE ") == {"1"}
    assert found(repo, hybrid, title="wheel") == set()


def test_filters_combine_with_and(repo, hybrid):
    assert found(repo, hybrid, classification_prefixes=["B60"], keywords=["engine"]) == {"3"}


def test_patent_without_title_or_classification_is_still_searchable(repo, hybrid):
    # Not asserted as top hit: in hybrid mode it can tie with patent 1 after rank fusion.
    points = repo.search("a wheel with a spoke and a rim", SearchFilters(), hybrid, 2)
    assert {"doc_number": "4", "title": None, "classification": None} in [p.payload for p in points]
