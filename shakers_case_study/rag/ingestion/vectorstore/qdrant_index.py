from typing import List, Optional
from langchain_qdrant import QdrantVectorStore
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PointIdsList


class QdrantIndex:
    """
    Wrapper class for interacting with a Qdrant vector database collection.

    Handles creating collections, adding documents as points with embeddings,
    and performing similarity search.
    """

    def __init__(
        self,
        embedder: Embeddings,
        collection_name: str,
        host: str,
        port: int,
    ):
        """
        Initialize QdrantIndex with an embedder, collection details, and connection info.

        Args:
            embedder (Embeddings): Embedding model to generate vectors.
            collection_name (str): Name of the Qdrant collection.
            host (str): Qdrant host address.
            port (int): Qdrant port number.
        """
        self.embedder = embedder
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.vectorstore: Optional[QdrantVectorStore] = None

    def _get_client(self) -> QdrantClient:
        """
        Create and return a Qdrant client connected to the configured host and port.

        Returns:
            QdrantClient: Client instance connected to Qdrant server.
        """
        return QdrantClient(host=self.host, port=self.port)

    def _get_document_id(self, doc: Document) -> str:
        """
        Generate a stable UUID5 identifier for a document based on its source and content.

        Args:
            doc (Document): Document for which to generate an ID.

        Returns:
            str: UUID5 string based on document metadata and content.
        """
        source = doc.metadata.get("source", "")
        content = doc.page_content
        return str(uuid.uuid5(uuid.NAMESPACE_URL, source + content))

    def _ensure_collection_exists(self, vector_size: int) -> None:
        """
        Check if the collection exists in Qdrant, create it if missing.

        Args:
            vector_size (int): Dimensionality of the vectors stored.
        """
        client = self._get_client()
        existing_collections = [col.name for col in client.get_collections().collections]
        if self.collection_name not in existing_collections:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def add_documents(self, documents: List[Document]) -> None:
        """
        Embed and add documents as points to the Qdrant collection.

        If documents with same IDs exist, they are deleted before upsert.

        Args:
            documents (List[Document]): List of documents to add.
        """
        client = self._get_client()
        embeddings = self.embedder.embed_documents([doc.page_content for doc in documents])
        ids = [self._get_document_id(doc) for doc in documents]

        self._ensure_collection_exists(len(embeddings[0]))

        try:
            client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=ids),
            )
        except Exception as e:
            print(f"[Warning] Failed to delete points: {e}")

        points = [
            PointStruct(
                id=id_,
                vector=vec,
                payload={
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                },
            )
            for id_, vec, doc in zip(ids, embeddings, documents)
        ]

        client.upsert(collection_name=self.collection_name, points=points)

        if self.vectorstore is None:
            self.vectorstore = QdrantVectorStore(
                embedding=self.embedder,
                collection_name=self.collection_name,
                client=client,
                content_payload_key="page_content",
                metadata_payload_key="metadata",
            )

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform a similarity search in the collection using a query string.

        Args:
            query (str): Query text to embed and search.
            k (int): Number of nearest neighbors to return.

        Returns:
            List[Document]: List of matching documents.
        """
        if self.vectorstore is None:
            raise RuntimeError("Qdrant index not initialized.")
        return self.vectorstore.similarity_search(query, k)

    def load(self) -> None:
        """
        Load the vectorstore from Qdrant client for future operations.
        """
        client = self._get_client()
        self.vectorstore = QdrantVectorStore(
            embedding_function=self.embedder.embed_query,
            collection_name=self.collection_name,
            client=client,
        )
