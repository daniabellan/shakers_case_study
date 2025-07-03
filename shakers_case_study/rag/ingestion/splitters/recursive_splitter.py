from typing import List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from shakers_case_study.rag.ingestion.splitters.base_splitter import BaseSplitter
from shakers_case_study.utils.logging import get_logger

logger = get_logger("run_ingestion")


class RecursiveTextSplitter(BaseSplitter):
    """
    Splits documents into chunks using a recursive character-based text splitter.

    This class wraps LangChain's RecursiveCharacterTextSplitter and
    provides a method to split documents into smaller chunks with specified
    chunk size and overlap.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initializes the RecursiveTextSplitter with chunk size and overlap.

        Args:
            chunk_size (int): The maximum size of each chunk (default 1000 characters).
            chunk_overlap (int): The number of overlapping characters between chunks (default 200).
        """
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into smaller chunks.

        Args:
            documents (List[Document]): The list of documents to split.

        Returns:
            List[Document]: The resulting list of chunked documents.
        """
        logger.info(f"Splitting documents using {type(self.splitter).__name__} strategy")

        splitted_docs = self.splitter.split_documents(documents)

        logger.info(f"Splitted documents in {len(splitted_docs)} chunks")

        return splitted_docs
