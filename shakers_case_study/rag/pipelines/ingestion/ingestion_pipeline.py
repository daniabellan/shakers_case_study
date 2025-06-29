from typing import List, Optional

from langchain.embeddings.base import Embeddings
from langchain.schema import Document

from shakers_case_study.rag.ingestion.loaders.base_loader import BaseLoader
from shakers_case_study.rag.ingestion.splitters.base_splitter import BaseSplitter
from shakers_case_study.rag.ingestion.vectorstore.qdrant_index import QdrantIndex
from shakers_case_study.utils.logging import get_logger

logger = get_logger("run_ingestion")


class IngestionPipeline:
    """
    Pipeline class to orchestrate document ingestion:
    loading, splitting, embedding, and storing in a vector database.
    """

    def __init__(
        self,
        loader: BaseLoader,
        splitter: BaseSplitter,
        embedder: Embeddings,
        vectorstore: QdrantIndex,
    ):
        """
        Initialize the ingestion pipeline with its components.

        Args:
            loader (BaseLoader): Component to load documents from a source.
            splitter (BaseSplitter): Component to split documents into chunks.
            embedder (Embeddings): Embedding model for text vectorization.
            vectorstore (QdrantIndex): Vector database interface to store embeddings.
        """
        self.loader = loader
        self.splitter = splitter
        self.embedder = embedder
        self.vectorstore = vectorstore

    def run(self, source_path: Optional[str] = None) -> None:
        """
        Execute the ingestion pipeline: load, split, embed, and store documents.

        Args:
            source_path (Optional[str]): Optional path or URL to load documents from.
        """
        logger.info("Running pipeline...")

        # 1. Load documents
        documents: List[Document] = self.loader.load_documents(source_path)

        # 2. Split documents into chunks
        split_documents: List[Document] = self.splitter.split_documents(documents)

        # 3. (Embedding is done internally in vectorstore.add_documents)

        # 4. Add split documents to the vectorstore (which performs embeddings internally)
        self.vectorstore.add_documents(split_documents)
