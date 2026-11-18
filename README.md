# Video link
https://drive.google.com/file/d/1nEO3VXFdZib7fVdR-Kr3RdF0RZ-rtoyi/view?usp=drive_link

# Patent search engine

Thinkstruct coding task. This README covers **Part 1** (a basic search engine with
hybrid search: semantic search + constraints) and **Part 3** ([two-phase
search](#part-3-two-phase-search-with-a-cross-encoder) with a cross-encoder). The Part 2
design doc (running it at 10M patents) is in [DESIGN.md](DESIGN.md).

## The problem

A patent examiner has an idea in plain words, for example *"wheel spoke with
reinforced rim"*, and wants to find the most similar patent applications.
Often they also want to narrow the results:

- only patents in a classification, e.g. codes starting with `B60B` (vehicle wheels),
- only patents whose title or abstract contains certain keywords,
- a patent with a specific title.

So the search engine takes:

- **Input:** a sentence in natural language, plus optional filters.
- **Output:** a ranked list of patents (document number, title, classification, score)
  and how long the search took. Any patent can then be opened in full (abstract,
  claims, description, citation) by its document number.

## How it works

```
raw JSON files ──> clean files ──> Postgres ──> Qdrant (vector search) <── POST /search
```

1. **Clean** (`uv run clean-patents`): reads the raw files in `data/raw/`, fixes or
   drops bad records (see [Missing fields](#missing-fields)) and writes clean copies to
   `data/clean/`.
2. **Store** (on app startup): every clean patent is saved in Postgres, the source of
   truth that `GET /patents/{doc_number}` reads from.
3. **Index** (on app startup): each patent is turned into two vectors in Qdrant:
   - a **meaning** vector (`BAAI/bge-small-en-v1.5`) of the title + abstract, which
     finds patents that say the same thing in other words;
   - a **keyword** vector (BM25) of the title + abstract + claims, which finds
     patents that use the same words.
4. **Search** (`POST /search`): the query is matched against both vectors and the two
   result lists are merged (reciprocal rank fusion). Set `"hybrid": false` to use only
   the meaning vector.
5. **Rerank** (optional, Part 3): `"rerank": "cross_encoder"` or `"blend"` rescores the
   top `rerank_candidates` with a cross-encoder ([details](#part-3-two-phase-search-with-a-cross-encoder)).

Filters are optional and all of them must hold (AND):

| Filter                    | Meaning                                                        |
|---------------------------|----------------------------------------------------------------|
| `classification_prefixes` | the classification starts with any of these, e.g. `["B60B"]`   |
| `keywords`                | every keyword appears (as a whole word) in the title or abstract |
| `title`                   | exact title, ignoring upper/lower case and extra spaces         |

> **About the word "hybrid":** the task uses it for *search + filters*. In this API,
> filters are the `filters` field, and the `hybrid` flag means *meaning + keyword
> vectors*. Both can be turned on and off independently.

Code layout: `controllers/` (HTTP) → `services/` (logic) → `repositories/` (Postgres, Qdrant).

## Key decisions

| Decision | Instead of | Why |
|---|---|---|
| Input: free text + optional filters. Output: ranked patents, full patent by number ([details](#the-problem)) | Patent ID as input; a specific claim or paragraph as output | Free text covers both an idea and a pasted claim, with no scraper needed. Vectors are per patent, so results are whole patents |
| Qdrant for search | Postgres + pgvector, FAISS | Meaning vectors, keyword vectors and indexed filters in one query, with filters applied before scoring ([details](#why-filtering-is-cheap-here)) |
| Postgres holds the data, Qdrant is only an index | Qdrant only | Full patent text and indexing progress live in a real database, so the index can be rebuilt from it at any time |
| Meaning vector from title + abstract only | Embedding all the text | The model reads at most 512 tokens; the keyword vector covers the claims, since it has no length limit |
| Both vectors merged (RRF), on by default | Meaning vector only | Finds the right patent in the top 10 about 21 points more often ([results](#how-good-are-the-results)) |
| Reranking is opt-in (`rerank`, default `"none"`); use `"blend"` | Always on | ~1.9 s per query on CPU vs ~15 ms without ([details](#part-3-two-phase-search-with-a-cross-encoder)) |
| Classification prefixes stored as exact keywords | Matching prefixes at query time | "Starts with `B60B`" becomes an indexed lookup instead of a string scan |
| Drop only patents with no ID or no claims | Dropping any incomplete patent | All 640 kept; the 119 without a description are still searchable ([details](#missing-fields)) |
| Cleaning is its own step; raw files are never edited | Cleaning while loading | Re-runs are safe, and cleaning rules can change without losing the original data ([details](#data-pipeline-details)) |
| Patents are removed through an explicit list | Deleting files | Removing a patent is deliberate and can be undone ([details](#data-pipeline-details)) |

## Missing fields

**Choice:** a patent is **excluded** only if it has no document number (it cannot be
identified) or no claims (the legal definition of the invention is missing). Every
other patent is **kept**, even with gaps.

Why: in a real published application every field exists (except the bibtex citation),
so an empty field is an error in the dataset, not a property of the patent. The only
question is whether the record is still useful, and it is as long as it has an ID and
claims.

How gaps are handled:

- A value counts as empty if it is missing, `null`, `""`, only spaces or `"N/A"`.
  Empty text becomes `null`; empty items are dropped from lists.
- Canceled-claim placeholders such as `1 .- 15 . (canceled)` hold no claim text, so
  they are dropped too. A patent left with only those would count as having no claims.
- No description: still searchable by title, abstract and claims.
- No classification: still searchable, but never passes a classification filter.
- No title and no abstract: the meaning vector is built from the claims instead.

**Result on the given data (640 patents):** 0 excluded. 119 patents (18.6%) have no
detailed description. 167 canceled-claim placeholders were dropped, and every patent
still has real claims. No other field is empty anywhere. The cleaner prints these
counts and saves them in `data/clean/manifest.json`.

## How to run

Requirements: Docker running, and the dataset unzipped into `data/raw/` (files named
`patents_ipa*.json`). Then:

```bash
./run.sh
```

This cleans the data, starts Postgres and Qdrant, and serves the API at
http://127.0.0.1:8000, with interactive docs at http://127.0.0.1:8000/docs. Ctrl+C stops it.

Open http://127.0.0.1:8000 for a small search page (query, filters, rerank mode, click a
result to read it). It is a demo client for trying the API, not the Part 3 "Interface at
Scale" enhancement.

The first run downloads the models and indexes everything, which takes a few minutes.
Models are cached in a Docker volume, so later starts are fast. This is the Part 2 proof
of concept: step 1 of the build order in [DESIGN.md](DESIGN.md).

Search with filters:

```bash
curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" -d '{"query": "wheel spoke with reinforced rim", "filters": {"classification_prefixes": ["B60B"], "keywords": ["wheel"]}}'
```

Open one patent:

```bash
curl http://127.0.0.1:8000/patents/20240051333
```

## Timings: with and without filters

To reproduce, with the app's index built:

```bash
uv run bench-search
```

It times the same thing as `took_ms` in a search response (embedding the query +
searching, no HTTP overhead). Query *"wheel spoke with reinforced rim"*, 640 patents,
median of 20 runs on a local machine. "With filters" = classification `B60B` + keyword
`wheel`.

| Ranking                           | No filters | With filters |
|-----------------------------------|-----------:|-------------:|
| Meaning only (`hybrid=false`)     |   19.43 ms |     11.77 ms |
| Meaning + keyword (`hybrid=true`) |   14.88 ms |     14.18 ms |

**What this shows:** at 640 patents the filters cost nothing measurable. Beyond that,
the numbers are mostly noise: they move by several milliseconds between runs, more than
the differences between rows. The dataset is too small to show filters *speeding
things up*: most of the time is the fixed cost of embedding the query and calling
Qdrant.

### Why filtering is cheap here

The naive way is to search everything and then throw away results that fail the
filters. That wastes work and can return fewer results than asked for. Here the filters
run **before** scoring, so only matching patents are scored:

- **Indexes on every filtered field**, so Qdrant finds matching patents from an index
  instead of checking every patent.
- **Prefix search as an exact lookup.** Each patent stores every prefix of its
  classification (`B`, `B6`, `B60`, `B60B`, ...), so "starts with `B60B`" is a fast
  exact match instead of a string scan.
- **Text indexes** on title and abstract for keywords, and a normalized title for the
  exact-title filter.
- **Filters applied to both vectors** before their results are merged.

So on a large dataset a selective filter should make search *faster*, because there
are fewer patents to score.

## How good are the results?

`eval-search` has two modes. Both need the app's index built, and both run every
ranking mode (meaning only, meaning + keyword, and the Part 3 rerank modes).

**Labeled queries (the main metric).** Pass a JSONL file with one query per line.
Each line lists its relevant patents, graded `2` (on point) or `1` (related);
any patent not listed counts as not relevant. `filters` is optional and has the same
shape as in `POST /search`:

```json
{"query": "bicycle spoke with threaded ends", "relevant": {"20240051333": 2}, "filters": {"classification_prefixes": ["B60B"]}}
```

```bash
uv run eval-search data/eval/queries.jsonl
```

It reports nDCG@10 (ranking quality, with on-point patents counting more), recall@10
(the share of relevant patents found) and MRR@10 (how high the first relevant patent
ranks). With 50–100 queries a run takes ~3–6 min, almost all of it in the rerank modes.
A line with no query or no relevant patent stops the run with its line number.

**Known-item sanity check (no labels needed).** Each patent is used as its own test:
a piece of its text is the query, and the search is a hit if that patent comes back
in the top k. This only shows that a known patent can be found, not whether the
ranking is good, and the queries use the patent's own wording.

```bash
uv run eval-search
```

On all patents this takes ~35 min. Pass a number (e.g. `uv run eval-search 50`) to use
that many queries per source, spread evenly over the dataset. The table below is from
this mode.

| Query taken from | Ranking           | Queries | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|------------------|-------------------|--------:|---------:|---------:|----------:|-------:|
| First claim      | Meaning only      |     546 |    0.615 |    0.766 |     0.817 |  0.682 |
| First claim      | Meaning + keyword |     546 |    0.784 |    0.980 |     0.991 |  0.874 |
| Description      | Meaning only      |     521 |    0.340 |    0.474 |     0.547 |  0.397 |
| Description      | Meaning + keyword |     521 |    0.493 |    0.702 |     0.762 |  0.586 |

- The **description** row is the fairer one: that text is not indexed at all, so it is
  like a new description of the same invention. (The first claim is indexed by the
  keyword vector, which makes that row look better.)
- Merging both vectors finds the right patent in the top 10 about 21 points more often.
  That is why `hybrid` is on by default.
- Texts under 8 words are skipped, so there are fewer than 640 queries.
- Limit: the queries use the patent's own wording, so real examiner queries would score
  lower.

## Part 3: two-phase search with a cross-encoder

**Why this enhancement:** stage 1 compares two vectors computed separately: one for the
query, and one for the patent's title + abstract, computed ahead of time. That is fast
enough for millions of patents, but the model never sees the query and the patent
together, and it never sees the claims at all. A **cross-encoder** reads the query and a
patent's text as one input and scores the pair. That is more precise, but nothing can be
computed ahead of time, so it only runs on a short list.

**How it works:**

```
query ──> stage 1: hybrid search (Qdrant), top `rerank_candidates` ──> stage 2: cross-encoder ──> top `limit` (10)
```

- Stage 1 is the Part 1 search, filters included, returning `rerank_candidates` patents
  (default and maximum `MAX_RERANK_CANDIDATES` in `config.py`). `limit` must be at
  most `rerank_candidates`, otherwise the request is rejected (422). A patent stage 1
  ranks below the candidates is never recovered.
- Stage 2 loads each candidate's title + abstract + first claim from Postgres, cut to
  256 words, and scores it against the query with `Xenova/ms-marco-MiniLM-L-6-v2`
  (via `fastembed`, which was already installed, so no new dependency). Set the
  `RERANK_MODEL` environment variable to try another model.
- Two ways to use the scores (`rerank` field of `POST /search`):
  - `"blend"` (**use this one**): reciprocal rank fusion of the stage-1 order and the
    model's order, so a model that misreads patent language can't throw out a good
    stage-1 match on its own.
  - `"cross_encoder"`: order by the model's score alone. This is the standard two-stage
    setup, kept as the baseline: it shows the gain comes from blending, not from
    reranking alone (on description queries it does worse than no rerank, see
    [Results](#results)). It is also useful for trying a stronger `RERANK_MODEL`, whose
    score may be worth trusting on its own. Same model call and cost as `"blend"`.

```bash
curl -X POST http://127.0.0.1:8000/search -H "Content-Type: application/json" -d '{"query": "wheel spoke with reinforced rim", "rerank": "blend", "rerank_candidates": 20, "limit": 10}'
```

### Results

Same eval as [above](#how-good-are-the-results), on **50 queries per source** (spread
evenly over the dataset), so the full set isn't needed. To reproduce (~6 min):

```bash
uv run eval-search 50
```

| Query taken from | Ranking               | Recall@1 | Recall@5 | Recall@10 | MRR@10 |
|------------------|-----------------------|---------:|---------:|----------:|-------:|
| First claim      | Meaning + keyword     |    0.780 |    0.960 |     1.000 |  0.875 |
| First claim      | + cross-encoder       |    0.880 |    0.940 |     0.980 |  0.910 |
| First claim      | + blend               |    0.840 |    0.960 |     1.000 |  0.898 |
| Description      | Meaning + keyword     |    0.520 |    0.740 |     0.820 |  0.608 |
| Description      | + cross-encoder       |    0.480 |    0.680 |     0.760 |  0.560 |
| Description      | + blend               |    0.560 |    0.780 |     0.800 |  0.634 |

Latency (`uv run bench-search`, same query and machine as the Part 1 timings, median of
20):

| Ranking                   | No filters | With filters |
|---------------------------|-----------:|-------------:|
| Meaning + keyword         |   15.21 ms |     16.14 ms |
| + cross-encoder (20 → 10) | 1869.92 ms |   1847.86 ms |

**What this shows:**

- **Blend is the mode to use.** It improves recall@1 and MRR on both query sources and
  keeps recall@10 (within one query out of 50).
- **The cross-encoder alone helps only on claim queries.** On the fair (description)
  test, it does worse than plain hybrid search: a model trained on web search (MS MARCO)
  misreads some patent wording. Blending it with the stage-1 order is what makes it
  safe.
- **A small sample:** one query is 2 points, so read differences of 2–4 points as noise.
- **It is slow on CPU:** ~100 ms per (query, patent) pair at full length, and the cost
  grows faster than the text. Batch size and thread count barely changed this. In
  production it would run on a GPU (tens of ms per query).

**Next steps:**

- A larger reranker (`BAAI/bge-reranker-base` is in fastembed) on a GPU.
- Fine-tune the cross-encoder on (description paragraph, own patent) pairs from this
  data, which should fix the domain mismatch.
- Run the full eval once the reranker is fast enough.

## Data pipeline details

- **Re-running is safe.** The cleaner records a hash of each raw file in
  `data/clean/manifest.json` and only reprocesses new or changed files. The app marks
  each patent once it is indexed, so an interrupted startup picks up where it stopped.
- **Raw files are never edited or deleted.** They are the only thing that cannot be
  rebuilt.
- **Removing a patent:** add its document number to `data/removals.txt` (one per line,
  `#` for comments) and restart. It is deleted from Qdrant and Postgres and never
  re-added. Remove the line and restart to bring it back.

## Known limitations and next steps

What in the current code stops it from growing, and what fixes it. The full plan for
10M patents is in [DESIGN.md](DESIGN.md).

**Limitations at scale:**

- **Indexing runs inside the API's startup.** Fine for 640 patents. At 10M, the API
  would not serve anything for days. Fix: separate ingest and index workers that claim
  batches from Postgres (DESIGN.md §3).
- **Corrected patents are never re-loaded.** Inserts skip existing IDs, so a fixed raw
  record keeps its old text and vectors. Weekly updates need a `content_hash` per patent
  and an upsert when it changes.
- **Qdrant point IDs are `int(doc_number)`.** This fails for non-US numbers such as
  `EP1234567A1`. Fix: `uuid5(doc_number)`.
- **One Qdrant node with full-precision vectors.** RAM is the main cost at 10M. Next:
  int8 quantization, then sharding with replicas.
- **One Postgres connection per call.** Fine for one user. Concurrent traffic needs a
  connection pool.
- **The reranker takes ~1.9 s per query on CPU.** It needs a GPU before it can be on by
  default for many users.

**Next steps (measure before building):**

- Run on a 100k sample of the large dataset and measure ingest and index throughput,
  storage size, and search latency with and without filters. At 640 patents, filters
  can't show a speed-up. Medium-selectivity filters (e.g. `B60`) are the case to watch.
- Time query embedding and the Qdrant search separately. Today they are one number
  (12-19 ms), so it's unclear which part grows with the data.
