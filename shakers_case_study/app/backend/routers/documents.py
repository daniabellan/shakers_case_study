import os

import psycopg
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from shakers_case_study.app.backend.utils.response_formatter import ResponseFormatter

router = APIRouter()
formatter = ResponseFormatter()

# Absolute path to the directory containing uploaded markdown documents
DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../uploaded_docs"))


@router.get("/documents")
async def list_documents():
    """
    Retrieve a list of all markdown (.md) documents available in the DOCS_DIR.

    Returns:
        A formatted response containing the list of markdown filenames,
        the status of the request, and the count of documents found.

    Raises:
        Returns an error status with a message if an exception occurs while accessing the directory.
    """
    try:
        # List all files ending with .md in the documents directory
        files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".md")]
        return formatter.format(payload=files, status="success", count=len(files))
    except Exception as e:
        return formatter.format(status="error", message=str(e))


@router.get("/documents/{filename}")
async def get_document(filename: str):
    """
    Retrieve the contents of a specific markdown document by filename.

    Args:
        filename (str): The name of the markdown file to retrieve.

    Returns:
        FileResponse: The requested markdown file served with appropriate media type.

    Raises:
        HTTPException 404: If the file does not exist or does not have a '.md' extension.
    """
    file_path = os.path.join(DOCS_DIR, filename)

    # Validate file existence and extension
    if not os.path.exists(file_path) or not filename.endswith(".md"):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(path=file_path, media_type="text/markdown", filename=filename)


@router.get("/rag/history")
async def get_rag_history(user_id: str, thread_id: str = "thread-1"):
    """
    Retrieve RAG (Retrieval-Augmented Generation) history records from the database
    filtered by user ID and thread ID, ordered by timestamp.

    Args:
        user_id (str): The identifier of the user.
        thread_id (str, optional): The thread identifier. Defaults to "thread-1".

    Returns:
        dict: A dictionary with 'status' and 'history' keys. 'history' contains
              a list of JSON values ordered by timestamp.

    Raises:
        HTTPException 500: If a database connection or query execution error occurs.
    """
    # Database connection URI - consider moving to config or environment variables for security
    db_uri = "postgresql://shakers_case_study:shakers_case_study@localhost:5432/shakers_case_study?sslmode=disable"  # noqa: E501

    prefix = f"{user_id}.{thread_id}"

    query = """
        SELECT value
        FROM store
        WHERE prefix = %s
        ORDER BY value->>'timestamp' ASC;
    """

    try:
        with psycopg.connect(db_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(query, (prefix,))
                rows = cur.fetchall()
        history = [row[0] for row in rows]
        return {"status": "success", "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
