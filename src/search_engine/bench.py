"""Search latency with and without filters, per ranking mode: a measurement, not a test.

Run:  uv run bench-search   (needs: docker compose up -d, and the index built by starting the app)

Times the same call `took_ms` measures (query embedding + Qdrant search, plus the
cross-encoder for rerank), median of RUNS.
"""

from statistics import median
from time import perf_counter

from search_engine.controllers.search import get_service
from search_engine.models import SearchFilters, SearchRequest
from search_engine.services.search import SearchService

QUERY = "wheel spoke with reinforced rim"
FILTERS = SearchFilters(classification_prefixes=["B60B"], keywords=["wheel"])
RUNS = 20
LIMIT = 10
# blend costs the same as rerank (same model call, plus a sort), so it isn't timed.
MODES = {"dense": (False, "none"), "hybrid": (True, "none"), "rerank": (True, "cross_encoder")}


def time_ms(service: SearchService, req: SearchRequest) -> float:
    service.search(req)  # warm-up: model load, connections
    times = []
    for _ in range(RUNS):
        start = perf_counter()
        service.search(req)
        times.append((perf_counter() - start) * 1000)
    return median(times)


def main() -> None:
    service = get_service()
    print(f"query={QUERY!r}  filters={FILTERS.model_dump(exclude_defaults=True)}  median of {RUNS}")
    print(f"{'ranking':<10}{'no filters':>12}{'with filters':>14}")
    for mode, (hybrid, rerank) in MODES.items():
        req = SearchRequest(query=QUERY, hybrid=hybrid, rerank=rerank, limit=LIMIT)
        plain = time_ms(service, req)
        filtered = time_ms(service, req.model_copy(update={"filters": FILTERS}))
        print(f"{mode:<10}{plain:>9.2f} ms{filtered:>11.2f} ms")
