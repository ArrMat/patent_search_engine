# Part 2: the search engine at 10 million patents

How to run the Part 1 search engine on ~10M patents with weekly updates. It keeps the
Part 1 design (raw files → Postgres → Qdrant) and only changes what breaks at scale.

**In short:** raw files in object storage, cleaning and indexing done in batches by
workers, Postgres as source of truth *and* job queue, Qdrant split over 3 machines, and
one GPU for the optional reranking step. **~$2,550 / month** (~$2,150 without the GPU),
or ~$930 for a single-node start.

## 1. Sizing

Measured on the Part 1 data: 640 patents = 15.9 MB, so **~25 KB per patent**. The rest
is calculated or estimated; measure on a 100k sample before buying hardware.

| Item | × 10M patents |
|---|---:|
| Raw JSON (gzipped) | ~250 GB (~50 GB) |
| Postgres (text, claims, description) | ~300 GB |
| Qdrant on disk (dense + BM25 vectors, payload, indexes) | ~70 GB |
| Qdrant in RAM (int8 dense vectors, graph, payload indexes) | ~15-20 GB |

Not "big data": no Spark or Kafka needed. **RAM for Qdrant is the real limit**, and
the main cost.

## 2. Components

```
USPTO weekly file ─> Fetcher (cron) ─> S3 raw/ (never edited)
                                            │
                                     Clean workers (clean.py)
                                            │ upsert
                    Postgres: patents, claims, job + status tables
                                            │ pending patents
                                     Index workers (embed + upsert)
                                            │
          users ─> Search API (×2) ─> Qdrant (3 nodes, 6 shards, 2 copies each)
                         ├─> Postgres read replica (GET /patents/{id}, rerank texts)
                         └─> Reranker (1 GPU, only when `rerank` is set)
```

| Component | What | Why |
|---|---|---|
| S3 | One raw file per weekly batch | Cheap; the only thing that can't be rebuilt |
| Clean workers | Part 1's `clean.py`, one file per job | Already written; files are independent |
| Postgres (RDS) | Part 1 tables + job/status tables | Source of truth and job queue in one (`SELECT … FOR UPDATE SKIP LOCKED`); no extra queue to run |
| Index workers | Part 1's indexing loop, N copies | Embedding is the slow step, so it gets the parallelism |
| Qdrant cluster | 3 nodes, int8 vectors in RAM, full vectors on disk | Same queries as Part 1; one node can die without downtime |
| Search API | Part 1's FastAPI app, stateless | Embeds the query on CPU in a few ms |
| Reranker | Part 3's cross-encoder behind a small HTTP service, 1 GPU | ~1.9 s per query on CPU (measured); a GPU should bring it to tens of ms (estimate, measure). Separate so the API stays cheap CPU tasks |

## 3. Pipelines

Every step is **idempotent**, so the fix for most failures is "run it again".

**Ingest (weekly; the backfill is the same with all files at once)**
1. Fetcher uploads the file to S3 and inserts it in `ingest_files` with its SHA-256.
   Same hash already there → stop.
2. A clean worker claims the file, runs the Part 1 cleaning rules, and upserts patents
   in batches of ~1,000. Each patent has a `content_hash`: new or changed → `indexed_at
   = NULL` (needs indexing); unchanged → skip.
3. File marked `loaded` with the cleaner's counts (kept, dropped, why).

**Index**
1. A worker claims ~256 patents with `indexed_at IS NULL`.
2. Builds the same dense + BM25 vectors as Part 1 and upserts them to Qdrant.
3. Only after Qdrant confirms, sets `indexed_at`. A crash in between just means the
   batch is indexed twice, which is harmless (upsert by ID).

For the 10M backfill, rent a few GPUs (`bge-small` should take hours, not days; measure
it) and turn off Qdrant's graph building until the load is done. Weekly updates need
one small CPU worker.

**Search:** unchanged from Part 1 (dense + BM25, filters before scoring, RRF). Qdrant
queries all shards in parallel; very selective filters automatically fall back to an
exact scan of the matching points.

**Rerank (opt-in, Part 3):** with `rerank` set, the API takes the top `rerank_candidates`
from search, reads their title + abstract + first claim from the Postgres replica (one
lookup by primary key each), and sends the pairs to the reranker. Cost per query doesn't
grow with the database: always `rerank_candidates` pairs. Use `"blend"`, which was the best mode in the Part 3 eval.

**Rebuild (new model):** fill a new collection from Postgres, run `eval-search` on it,
then switch the `patents` alias to it. No downtime, and the old collection is the
rollback.

## 4. Errors and status

**Claiming work:** a worker sets `status = running` and a `heartbeat_at` time on the
job, and refreshes it while working. A job whose heartbeat is older than 10 minutes
goes back to `pending`, so a dead worker costs one batch. Don't hold a database
transaction open while processing.

| Failure | Handling |
|---|---|
| Malformed file | File → `failed` with the error; a person looks at it |
| One bad patent | Dropped and counted by the cleaner (Part 1 behaviour) |
| Embedding/Qdrant write fails | Retry with backoff; after 5 attempts the patent is skipped and shown on the dashboard |
| Qdrant node dies | Second copy serves; replace the node |
| Qdrant lost | Rebuild from Postgres (~a day); weekly snapshots to S3 make it faster |
| Postgres dies | Managed failover; search keeps working (only needs Qdrant) |
| Reranker down or over ~500 ms | Return the stage-1 order and say so in the response; search never fails because of reranking |

**Status tables:** `ingest_files` (file, hash, status, counts, error) and, on
`patents`: `content_hash`, `indexed_at`, `index_attempts`, `index_error`. The dashboard
is SQL over these: pending to index, failed, files by status, Postgres count vs Qdrant
count, plus API latency. A nightly job checks 1,000 random IDs exist in both stores.

## 5. Cost

AWS us-east-1, on-demand, looked up on 2026-11-17 from price trackers. Confirm in the
[AWS Pricing Calculator](https://calculator.aws/) before buying.

| Component | Calculation | Monthly |
|---|---|---:|
| Qdrant: 3 × `r6i.2xlarge` (64 GB) + 600 GB disk | 3 × $0.504/h × 730 + 600 × $0.08 | ~$1,150 |
| Postgres: RDS `db.r6g.xlarge` Multi-AZ, 500 GB | $0.899/h × 730 + storage | ~$715-770 |
| Search API: 2 Fargate tasks + load balancer | 2 × $72 + $22 | ~$165 |
| Workers, S3, monitoring | mostly idle; monitoring estimated | ~$70 |
| Reranker: 1 × `g4dn.xlarge` (T4 GPU), on-demand | $0.526/h × 730 | ~$385 |
| **Total** | | **~$2,550** |

- **One-time backfill:** a few GPU spot hours (`g4dn.xlarge`, ~$0.25/h) plus CPU for
  loading: **under $25**.
- **Cheaper start (~$930):** one Qdrant node, single-AZ Postgres, one API task, no GPU
  (reranking on the API's CPU at ~2 s per query, fine for a few users). Fine for 10M
  patents; any failure is downtime.
- **Reranker GPU:** one is enough until reranked queries reach several per second.
  It's always on, and unused without reranking, so drop it if almost nobody asks for it.
- **What drives cost:** RAM. A bigger embedding model (1024 dims) or chunking
  descriptions multiplies the vectors; int8/binary quantization claws most of it back.

## 6. Challenges

1. **Filters of medium selectivity** (e.g. `B60`) are the slow case for the vector
   index. Measure on real data.
2. **Patents without a classification code** silently fail every classification
   filter. Not seen in our data, but likely in the full database: tell the user how
   many were excluded, or predict the code from similar patents.
3. **Postgres and Qdrant can drift**, since they are written separately. The "mark
   indexed after Qdrant confirms" rule and the nightly check keep them close.
4. **Document IDs:** Part 1 uses `int(doc_number)`, which fails for IDs like
   `EP1234567A1`. Use `uuid5(doc_number)`.
5. **Only title + abstract get a meaning vector**; claims rely on BM25 and descriptions
   aren't searched. Chunking or a long-context model fixes it but multiplies cost.
6. **Rebuilds need double the Qdrant space** while two collections exist.
7. **Evaluation:** `eval-search FILE.jsonl` scores a fixed set of labeled queries
   (nDCG/recall/MRR@10, relevance graded 1–2) and is the number to compare model changes
   on. No labeled set exists yet: build one with 50–100 queries (natural language, claims,
   filtered), with relevant patents judged from the pooled top results of every mode, and
   keep a held-out part if anything gets tuned on it. Until then, `eval-search [N]`
   (each patent's own text as the query) is only a sanity check: it flatters BM25 and
   has one relevant patent per query.
8. **The reranker model wasn't trained on patents** (MS MARCO, web search). Alone, it
   did worse than plain search on description queries in Part 3; blending fixes that.
   Fine-tuning it on this data, or a larger reranker (`bge-reranker-base`) once it runs
   on a GPU, should help. Compare with `eval-search` before switching.

## 7. Build order

1. Dockerfiles for the API and workers; run everything with `docker compose`.
2. Status tables, `content_hash`, `uuid5` IDs.
3. Move startup indexing into `ingest-worker` and `index-worker` commands with
   heartbeat claiming.
4. Dashboard queries.
5. Qdrant quantization, alias, then 3 nodes with sharding.
6. Backfill on a 100k sample, measure sizes and speed, then the full 10M.
7. Weekly fetcher cron and alerts.
