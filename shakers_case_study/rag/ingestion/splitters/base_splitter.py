from abc import ABC, abstractmethod
from langchain.schema import Document
from typing import List

class BaseSplitter(ABC):
    """
    Abstract base class for document splitters.

    Defines the interface for splitting a list of documents into smaller chunks.
    """

    @abstractmethod
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of documents into smaller document chunks.

        Args:
            documents (List[Document]): The input list of documents to be split.

        Returns:
            List[Document]: The resulting list of split document chunks.
        """
        pass
