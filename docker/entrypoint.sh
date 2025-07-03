#!/bin/bash
set -e

export PYTHONPATH=/app

echo "Starting ingestion check..."

python -m shakers_case_study.rag.pipelines.ingestion.run_ingestion


echo "Starting FastAPI server..."

exec uvicorn shakers_case_study.app.backend.main:app --host 0.0.0.0 --port 8000
