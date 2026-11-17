"""PatentRepository against the test-only Postgres. Needs: docker compose up -d postgres-test

Never uses DATABASE_URL: the fixture truncates tables, so it must not reach the dev database.
"""

import psycopg
import pytest

from search_engine.config import TEST_DATABASE_URL
from search_engine.models import Patent
from search_engine.repositories.patents import PatentRepository


def patent(n: int, claims: tuple[str, ...] = ("c1", "c2")) -> Patent:
    return Patent(doc_number=str(n), title=f"T{n}", abstract="A", classification="B60B",
                  claims=list(claims), detailed_description=["d1", "d2"], filename="f", bibtex="b")


@pytest.fixture
def repo():
    """A repository on an empty schema: every test starts from the same state."""
    try:
        repo = PatentRepository(TEST_DATABASE_URL)
    except psycopg.OperationalError:
        pytest.skip("test Postgres is down: docker compose up -d postgres-test")
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        conn.execute("TRUNCATE patents CASCADE")
    return repo


def test_add_then_get_round_trips_with_claims_in_order(repo):
    p = patent(1, claims=("z", "a", "m"))
    repo.add([p])
    assert repo.get("1") == p


def test_round_trips_patent_with_only_number_and_claims(repo):
    bare = Patent(doc_number="9", claims=["c1"], detailed_description=[])
    repo.add([bare])
    assert repo.get("9") == bare


def test_get_unknown_patent_returns_none(repo):
    assert repo.get("nope") is None


def test_get_many_returns_known_patents_by_number(repo):
    repo.add([patent(1), patent(2), patent(3)])
    assert repo.get_many(["3", "1", "404"]) == {"3": patent(3), "1": patent(1)}


def test_add_leaves_already_stored_patents_untouched(repo):
    repo.add([patent(1)])
    repo.add([patent(1, claims=("changed",)), patent(2)])
    assert repo.get("1") == patent(1)
    assert repo.get("2") == patent(2)


def test_pending_is_ordered_limited_and_excludes_indexed(repo):
    repo.add([patent(3), patent(1), patent(2)])
    assert [p.doc_number for p in repo.pending(2)] == ["1", "2"]
    repo.mark_indexed(["1", "2"])
    assert [p.doc_number for p in repo.pending(10)] == ["3"]


def test_remove_deletes_patents_and_their_claims(repo):
    repo.add([patent(1), patent(2), patent(3)])
    repo.remove(["1", "3", "404"])  # unknown numbers are ignored
    assert (repo.get("1"), repo.get("2"), repo.get("3")) == (None, patent(2), None)
    with psycopg.connect(TEST_DATABASE_URL) as conn:
        assert conn.execute("SELECT DISTINCT doc_number FROM claims").fetchall() == [("2",)]


def test_reset_indexed_makes_everything_pending_again(repo):
    repo.add([patent(1), patent(2)])
    repo.mark_indexed(["1", "2"])
    assert repo.pending(10) == []
    repo.reset_indexed()
    assert [p.doc_number for p in repo.pending(10)] == ["1", "2"]
