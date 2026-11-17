#!/usr/bin/env bash
# Builds and starts the whole project (Postgres, Qdrant, app) in Docker.
# API: http://127.0.0.1:8000 (docs at /docs). Ctrl+C stops it.
set -euo pipefail
cd "$(dirname "$0")"
exec docker compose --profile app up --build app
