"""Clean the raw patent files, one partition (weekly file) at a time.

    data/raw/patents_ipaYYMMDD.json  ->  data/clean/patents_ipaYYMMDD.jsonl
    data/clean/manifest.json             state + cleaning report per file, and totals

Run:  uv run clean-patents [raw_dir clean_dir]

Incremental and idempotent: a raw file is skipped when its sha256 and RULES_VERSION
match the manifest, so re-running costs one hash per file. Stdlib only.

Rules (bump RULES_VERSION when you change them, so every file is reprocessed):
- Absent key, null, "", whitespace and "N/A" are the same thing: empty.
  Text fields become null; list fields drop their empty items (so [""] becomes []).
- A published application always has a number and at least one claim, so a record
  without either is an extraction error we cannot use: it is excluded and counted.
- Any other empty field (including detailed_description) is also an extraction
  error, but the record is still searchable: it is kept and the gap is counted.
- A canceled-claim placeholder ("1 .- 15 . (canceled)") holds no claim text, so it is
  dropped like an empty item and counted. A record left with only those has no claims.
"""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from search_engine.config import CLEAN_DIR, RAW_DIR

RULES_VERSION = 2
# Only numbers, punctuation and "(canceled)": "16 . (canceled)", "1 - 15 : (canceled)".
CANCELED = re.compile(r"[\d\s.,:;\-–]*\(cancell?ed\)[\s.]*", re.IGNORECASE)

TEXT_FIELDS = ("doc_number", "title", "abstract", "classification", "filename", "bibtex")
LIST_FIELDS = ("claims", "detailed_description")
REQUIRED = ("doc_number", "claims")


def _text(value: object) -> str | None:
    s = "" if value is None else str(value).strip()
    return None if s.upper() in ("", "N/A") else s


def clean_record(raw: object) -> tuple[dict | None, Counter]:
    """Return (clean record, or None if excluded) and what was found in it."""
    stats: Counter = Counter()
    if not isinstance(raw, dict):
        stats["excluded:not_an_object"] += 1
        return None, stats
    rec: dict = {k: _text(raw.get(k)) for k in TEXT_FIELDS}
    for k in LIST_FIELDS:
        value = raw.get(k)
        items = value if isinstance(value, list) else [] if value is None else [value]
        rec[k] = [s for s in map(_text, items) if s]
        if dropped := len(items) - len(rec[k]):
            stats[f"blank_items_dropped:{k}"] += dropped
    claims = [c for c in rec["claims"] if not CANCELED.fullmatch(c)]
    if canceled := len(rec["claims"]) - len(claims):
        stats["canceled_claims_dropped"] += canceled
        rec["claims"] = claims
    for k, v in rec.items():
        if not v:
            stats[f"empty:{k}"] += 1
    for k in REQUIRED:
        if not rec[k]:
            stats[f"excluded:no_{k}"] += 1
            return None, stats
    return rec, stats


def run(raw_dir: Path = RAW_DIR, clean_dir: Path = CLEAN_DIR) -> dict:
    """Clean new or modified raw files. Returns the manifest it wrote."""
    clean_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = clean_dir / "manifest.json"
    # Entries of raw files no longer in raw_dir are kept: raw files get archived, and
    # archiving is not a reason to drop data. Patents are removed via data/removals.txt.
    files: dict[str, dict] = (
        json.loads(manifest_path.read_text("utf-8"))["files"] if manifest_path.exists() else {}
    )
    errors = 0

    for src in sorted(raw_dir.glob("*.json")):
        out = clean_dir / f"{src.stem}.jsonl"
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        prev = files.get(src.name)
        if prev and (prev["sha256"], prev["rules_version"]) == (digest, RULES_VERSION) and out.exists():
            continue
        try:
            records = json.loads(src.read_text("utf-8"))
        except ValueError as e:  # bad JSON or bad encoding: keep the previous clean copy
            print(f"ERROR   {src.name}: {e}", file=sys.stderr)
            errors += 1
            continue
        stats: Counter = Counter()
        kept = []
        for raw in records if isinstance(records, list) else [records]:
            rec, found = clean_record(raw)
            stats += found
            if rec:
                kept.append(rec)
        with out.open("w", encoding="utf-8", newline="\n") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in kept)
        stats["records"] = len(kept) + sum(v for k, v in stats.items() if k.startswith("excluded:"))
        stats["kept"] = len(kept)
        files[src.name] = {"sha256": digest, "rules_version": RULES_VERSION, **dict(sorted(stats.items()))}
        print(f"cleaned {src.name}: {len(kept)}/{stats['records']} kept")

    totals: Counter = Counter()
    for entry in files.values():
        totals += Counter({k: v for k, v in entry.items() if k not in ("sha256", "rules_version")})
    manifest = {"files": files, "totals": dict(sorted(totals.items())), "errors": errors}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{len(files)} files in manifest. Totals:")
    for k, v in manifest["totals"].items():
        print(f"  {k:42} {v}")
    return manifest


def main() -> None:
    args = [Path(a) for a in sys.argv[1:3]]
    sys.exit(1 if run(*args)["errors"] else 0)
