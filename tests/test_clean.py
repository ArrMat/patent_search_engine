"""Cleaner tests on synthetic records. Files live in pytest's tmp_path, never in data/."""

import json

import pytest

from search_engine.clean import clean_record, main, run

GOOD = {"doc_number": " 1 ", "title": "T", "abstract": "N/A", "classification": "B60B",
        "filename": "f", "bibtex": "b", "claims": ["c1", "", "  "], "detailed_description": [""]}


@pytest.fixture
def dirs(tmp_path):
    """A raw dir with one valid file (1 good + 1 excluded record) and one broken file."""
    raw, clean = tmp_path / "raw", tmp_path / "clean"
    raw.mkdir()
    (raw / "a.json").write_text(json.dumps([GOOD, {**GOOD, "doc_number": None}]), "utf-8")
    (raw / "b.json").write_text("{not json", "utf-8")
    return raw, clean


def test_clean_record_normalises_empty_values():
    rec, stats = clean_record(GOOD)
    assert rec["doc_number"] == "1"
    assert rec["abstract"] is None
    assert rec["claims"] == ["c1"]
    assert rec["detailed_description"] == []
    assert stats["empty:abstract"] == stats["empty:detailed_description"] == 1
    assert stats["blank_items_dropped:claims"] == 2


def test_clean_record_drops_canceled_claim_placeholders():
    claims = ["1 .- 15 . (canceled)", "16 . (canceled)", "17 - 20 : (canceled)",
              "21. A tire with noise cancelling foam.", "22 (Cancelled)"]
    rec, stats = clean_record({**GOOD, "claims": claims})
    assert rec["claims"] == ["21. A tire with noise cancelling foam."]
    assert stats["canceled_claims_dropped"] == 4


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        ({**GOOD, "claims": [""]}, "excluded:no_claims"),
        ({**GOOD, "claims": ["1 - 5 . (canceled)"]}, "excluded:no_claims"),
        ({"title": "no number", "claims": ["c"]}, "excluded:no_doc_number"),
        ("junk", "excluded:not_an_object"),
    ],
)
def test_clean_record_excludes_unusable_records(raw, reason):
    rec, stats = clean_record(raw)
    assert rec is None
    assert stats[reason] == 1


def test_run_writes_clean_file_and_manifest(dirs):
    raw, clean = dirs
    m = run(raw, clean)
    assert m["errors"] == 1 and "b.json" not in m["files"]
    assert m["totals"]["records"] == 2 and m["totals"]["kept"] == 1
    assert m["totals"]["excluded:no_doc_number"] == 1
    lines = (clean / "a.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["doc_number"] == "1"
    assert json.loads((clean / "manifest.json").read_text("utf-8")) == m


def test_run_is_idempotent_and_skips_unchanged_files(dirs):
    raw, clean = dirs
    first = run(raw, clean)
    stamp = (clean / "a.jsonl").stat().st_mtime_ns
    assert run(raw, clean) == first
    assert (clean / "a.jsonl").stat().st_mtime_ns == stamp  # not rewritten


def test_run_keeps_previous_clean_copy_when_raw_file_becomes_unreadable(dirs):
    raw, clean = dirs
    first = run(raw, clean)
    before = (clean / "a.jsonl").read_text("utf-8")
    (raw / "a.json").write_text("{broken", "utf-8")

    m = run(raw, clean)

    assert m["errors"] == 2  # a.json now, b.json as before
    assert m["files"]["a.json"] == first["files"]["a.json"]
    assert (clean / "a.jsonl").read_text("utf-8") == before


def test_run_reprocesses_modified_raw_file(dirs):
    raw, clean = dirs
    first = run(raw, clean)
    (raw / "a.json").write_text(json.dumps([GOOD, {**GOOD, "doc_number": "2"}]), "utf-8")

    entry = run(raw, clean)["files"]["a.json"]

    assert entry["kept"] == 2 and entry["sha256"] != first["files"]["a.json"]["sha256"]
    assert len((clean / "a.jsonl").read_text("utf-8").splitlines()) == 2


def test_run_reprocesses_everything_when_rules_version_changes(dirs, monkeypatch):
    raw, clean = dirs
    old = run(raw, clean)["files"]["a.json"]["rules_version"]
    monkeypatch.setattr("search_engine.clean.RULES_VERSION", old + 1)
    assert run(raw, clean)["files"]["a.json"]["rules_version"] == old + 1


def test_run_accepts_file_holding_a_single_object(dirs):
    raw, clean = dirs
    (raw / "c.json").write_text(json.dumps(GOOD), "utf-8")
    entry = run(raw, clean)["files"]["c.json"]
    assert entry["records"] == entry["kept"] == 1


@pytest.mark.parametrize(("keep_broken_file", "code"), [(True, 1), (False, 0)])
def test_main_exit_code_reports_errors(dirs, monkeypatch, keep_broken_file, code):
    raw, clean = dirs
    if not keep_broken_file:
        (raw / "b.json").unlink()
    monkeypatch.setattr("sys.argv", ["clean-patents", str(raw), str(clean)])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == code


def test_run_keeps_clean_copy_of_archived_raw_file(dirs):
    raw, clean = dirs
    first = run(raw, clean)
    (raw / "a.json").unlink()  # archived elsewhere: not a reason to drop its patents
    assert run(raw, clean)["files"] == first["files"]
    assert (clean / "a.jsonl").exists()
