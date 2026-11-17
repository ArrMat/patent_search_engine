from contextlib import asynccontextmanager

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from search_engine.controllers.search import get_service, router


@asynccontextmanager
async def lifespan(_: FastAPI):
    service = get_service()
    service.ingest()  # load and index pending patents before serving
    service.warm_up()  # first request shouldn't pay for model downloads
    yield


app = FastAPI(title="search-engine", lifespan=lifespan)
app.include_router(router)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def main() -> None:
    import uvicorn

    from search_engine.config import HOST, PORT, RELOAD

    uvicorn.run("search_engine:app", host=HOST, port=PORT, reload=RELOAD)
