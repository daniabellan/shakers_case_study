# backend/routers/documents.py
from fastapi import APIRouter
import os
from shakers_case_study.backend.utils.response_formatter import ResponseFormatter

router = APIRouter()
formatter = ResponseFormatter()

DOCS_DIR = "uploaded_docs"

@router.get("/documents")
async def list_documents():
    try:
        files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".md")]
        return formatter.format(payload=files, status="success", count=len(files))
    except Exception as e:
        return formatter.format(status="error", message=str(e))
