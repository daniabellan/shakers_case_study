from langchain.schema import Document
from typing import List, Optional
import requests

class BaseLoader:
    """
    Base class for document loaders that fetch documents from a remote HTTP service.
    
    Attributes:
        base_url (str): Base URL for the remote document service.
        document_prefix (str): URL path segment for document-related endpoints.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.document_prefix = "documents"

    def load_documents(self, source_path: Optional[str] = None) -> List[Document]:
        """
        Abstract method to load documents, optionally from a specific source path.
        
        Must be implemented by subclasses.
        
        Args:
            source_path (Optional[str]): Optional path or identifier to specify a subset of documents.
            
        Returns:
            List[Document]: A list of loaded Document objects.
        """
        raise NotImplementedError

    def fetch_filenames(self, base_url: Optional[str] = None) -> List[str]:
        """
        Fetches the list of available document filenames from the server.
        
        Args:
            base_url (Optional[str]): Optional base URL to override the default base_url.
        
        Returns:
            List[str]: List of filenames retrieved from the server.
        
        Raises:
            RuntimeError: If the server responds with an error status.
            requests.HTTPError: If the HTTP request fails.
        """
        url_to_use = base_url or self.base_url
        url = f"{url_to_use}/documents"
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "success":
            return data.get("payload", [])
        else:
            raise RuntimeError(f"Error fetching documents list: {data.get('message')}")

    def fetch_document_content(self, url: str) -> requests.Response:
        """
        Fetches the raw content of a single document from a given URL.
        
        Args:
            url (str): Full URL to the document.
        
        Returns:
            requests.Response: HTTP response object containing the document content.
        
        Raises:
            requests.HTTPError: If the HTTP request fails.
        """
        resp = requests.get(url)
        resp.raise_for_status()
        return resp
