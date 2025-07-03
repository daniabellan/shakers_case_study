from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from shakers_case_study.app.backend.routers import documents, rag

app = FastAPI()

# Define the base directory by resolving four levels up from the current file location
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# Mount the directory containing uploaded markdown documents as a static files route
app.mount("/uploaded_docs", StaticFiles(directory=BASE_DIR / "uploaded_docs"), name="uploaded_docs")

# Include the API routers for document handling and RAG (Retrieval-Augmented Generation) features
app.include_router(documents.router)
app.include_router(rag.router)


def main():
    uvicorn.run("shakers_case_study.app.backend.main:app", host="127.0.0.1", port=8000, reload=True)
