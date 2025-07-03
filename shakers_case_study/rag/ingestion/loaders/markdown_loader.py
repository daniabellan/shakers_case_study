from typing import List, Optional

from langchain.schema import Document

from shakers_case_study.rag.ingestion.loaders.base_loader import BaseLoader
from shakers_case_study.utils.logging import get_logger

logger = get_logger("run_ingestion")


class MarkdownLoader(BaseLoader):
    """
    Loader class to fetch and parse Markdown documents from a remote HTTP service.

    Inherits from BaseLoader and implements the `load_documents` method to
    download Markdown files, convert them into Document objects with metadata.
    """

    def load_documents(self, source_path: Optional[str] = None) -> List[Document]:
        """
        Loads and returns a list of markdown documents from the configured base URL.

        Args:
            source_path (Optional[str]): Optional path to specify a subset of documents.
            (Not used currently.)

        Returns:
            List[Document]: A list of Document objects containing the content and metadata.
        """
        if source_path:
            # TODO: Implement logic to load from disk
            logger.info(f"Loading documents from {source_path}...")

        else:
            # Load documents from URL
            logger.info(f"Loading documents from {self.base_url}...")
            filenames = self.fetch_filenames()
            documents = []

            for filename in sorted(filenames):
                url = f"{self.base_url}/{self.document_prefix}/{filename}"
                resp = self.fetch_document_content(url)
                doc = Document(
                    page_content=resp.text, metadata={"source_file": filename, "source": url}
                )
                documents.append(doc)

        logger.info(f"Loaded {len(documents)} documents")

        return documents
