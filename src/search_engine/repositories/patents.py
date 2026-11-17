"""Postgres: source of truth for patents, their claims and pipeline state."""

from collections.abc import Iterable

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from search_engine.config import DATABASE_URL
from search_engine.models import Patent

SCHEMA = """
CREATE TABLE IF NOT EXISTS patents (
    doc_number           text PRIMARY KEY,
    title                text,
    abstract             text,
    classification       text,
    filename             text,
    bibtex               text,
    detailed_description jsonb NOT NULL,
    indexed_at           timestamptz  -- pipeline state: NULL = not yet in Qdrant
);
-- Tables created before these columns became nullable (no-op otherwise).
ALTER TABLE patents
    ALTER COLUMN title DROP NOT NULL, ALTER COLUMN abstract DROP NOT NULL,
    ALTER COLUMN classification DROP NOT NULL, ALTER COLUMN filename DROP NOT NULL,
    ALTER COLUMN bibtex DROP NOT NULL;
CREATE TABLE IF NOT EXISTS claims (
    doc_number text REFERENCES patents ON DELETE CASCADE,
    position   int,
    text       text NOT NULL,
    PRIMARY KEY (doc_number, position)
);
CREATE INDEX IF NOT EXISTS patents_pending ON patents (doc_number) WHERE indexed_at IS NULL;
"""

SELECT = """
SELECT p.doc_number, title, abstract, classification, filename, bibtex, detailed_description,
       COALESCE((SELECT array_agg(c.text ORDER BY c.position)
                 FROM claims c WHERE c.doc_number = p.doc_number), '{}') AS claims
FROM patents p
"""


class PatentRepository:
    def __init__(self, url: str = DATABASE_URL) -> None:
        self._url = url
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def _connect(self) -> psycopg.Connection:
        # ponytail: one connection per call (commits and closes on exit).
        # Use psycopg_pool.ConnectionPool when request volume makes connects show up.
        return psycopg.connect(self._url, row_factory=dict_row)

    def add(self, patents: Iterable[Patent]) -> None:
        """Insert patents and their claims; ones already stored are left untouched."""
        patents = list(patents)
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO patents (doc_number, title, abstract, classification, filename,"
                " bibtex, detailed_description) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                " ON CONFLICT DO NOTHING",
                [
                    (p.doc_number, p.title, p.abstract, p.classification, p.filename,
                     p.bibtex, Jsonb(p.detailed_description))
                    for p in patents
                ],
            )
            cur.executemany(
                "INSERT INTO claims (doc_number, position, text) VALUES (%s, %s, %s)"
                " ON CONFLICT DO NOTHING",
                [(p.doc_number, i, c) for p in patents for i, c in enumerate(p.claims)],
            )

    def get(self, doc_number: str) -> Patent | None:
        with self._connect() as conn:
            row = conn.execute(SELECT + "WHERE p.doc_number = %s", (doc_number,)).fetchone()
        return Patent(**row) if row else None

    def get_many(self, doc_numbers: list[str]) -> dict[str, Patent]:
        """Patents by doc number; unknown numbers are left out."""
        with self._connect() as conn:
            rows = conn.execute(SELECT + "WHERE p.doc_number = ANY(%s)", (doc_numbers,)).fetchall()
        return {r["doc_number"]: Patent(**r) for r in rows}

    def pending(self, limit: int) -> list[Patent]:
        """Patents not yet embedded into Qdrant."""
        with self._connect() as conn:
            rows = conn.execute(
                SELECT + "WHERE indexed_at IS NULL ORDER BY p.doc_number LIMIT %s", (limit,)
            ).fetchall()
        return [Patent(**r) for r in rows]

    def mark_indexed(self, doc_numbers: list[str]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE patents SET indexed_at = now() WHERE doc_number = ANY(%s)", (doc_numbers,)
            )

    def remove(self, doc_numbers: list[str]) -> None:
        """Delete patents and (by cascade) their claims."""
        with self._connect() as conn:
            conn.execute("DELETE FROM patents WHERE doc_number = ANY(%s)", (doc_numbers,))

    def reset_indexed(self) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE patents SET indexed_at = NULL")
