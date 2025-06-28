import os

from fastapi import APIRouter
from fastapi import Query, HTTPException
from fastapi.responses import FileResponse
from shakers_case_study.rag.ingestion.loaders.markdown_loader import MarkdownLoader
from shakers_case_study.app.backend.utils.response_formatter import ResponseFormatter

router = APIRouter()
formatter = ResponseFormatter()

DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../uploaded_docs"))


@router.get("/documents")
async def list_documents():
    try:
        files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".md")]
        return formatter.format(payload=files, status="success", count=len(files))
    except Exception as e:
        return formatter.format(status="error", message=str(e))


@router.get("/documents/{filename}")
async def get_document(filename: str):
    file_path = os.path.join(DOCS_DIR, filename)
    if not os.path.exists(file_path) or not filename.endswith(".md"):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=file_path, media_type="text/markdown", filename=filename)
