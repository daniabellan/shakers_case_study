from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from shakers_case_study.backend.routers import documents

app = FastAPI()

app.mount("/uploaded_docs", StaticFiles(directory="uploaded_docs"), name="uploaded_docs")

app.include_router(documents.router)
