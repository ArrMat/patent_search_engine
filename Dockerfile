FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependencies first so code edits don't reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH" \
    HOST=0.0.0.0 RELOAD=0 \
    FASTEMBED_CACHE_PATH=/models

# Cleaning skips unchanged files (manifest hashes), so running it on every start is cheap.
CMD ["sh", "-c", "clean-patents && search-engine"]
