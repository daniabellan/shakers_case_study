from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from shakers_case_study.rag.config import (EmbedderConfig, LoaderConfig,
                                           Secrets, SplitterConfig,
                                           VectorstoreConfig)
from shakers_case_study.rag.ingestion.loaders.markdown_loader import \
    MarkdownLoader
from shakers_case_study.rag.ingestion.splitters.recursive_splitter import \
    RecursiveTextSplitter
from shakers_case_study.rag.ingestion.vectorstore.qdrant_index import \
    QdrantIndex

# ----- FACTORY: Loader -----


def get_loader(cfg: LoaderConfig):
    """
    Factory function to get the loader instance based on the configuration.

    Args:
        cfg (LoaderConfig): Configuration object for the loader.

    Returns:
        Loader instance corresponding to the specified type.

    Raises:
        ValueError: If the loader type is unsupported.
    """
    if cfg.type == "markdown":
        return MarkdownLoader(base_url=cfg.document_base_url)
    raise ValueError(f"Unsupported loader type: {cfg.type}")


# ----- FACTORY: Splitter -----


def get_splitter(cfg: SplitterConfig):
    """
    Factory function to get the splitter instance based on the configuration.

    Args:
        cfg (SplitterConfig): Configuration object for the splitter.

    Returns:
        Splitter instance corresponding to the specified type.

    Raises:
        ValueError: If the splitter type is unsupported.
    """
    if cfg.type == "recursive":
        return RecursiveTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
    raise ValueError(f"Unsupported splitter type: {cfg.type}")


# ----- FACTORY: Embedder -----


def get_embedder(cfg: EmbedderConfig, secrets: Secrets) -> Embeddings:
    """
    Factory function to get the embedder instance based on the configuration and secrets.

    Args:
        cfg (EmbedderConfig): Configuration object for the embedder.
        secrets (Secrets): Secrets object containing API keys, etc.

    Returns:
        An Embeddings instance from LangChain.

    Raises:
        ValueError: If the embedder type is unsupported or required secrets are missing.
    """
    if cfg.type == "google":
        return _google_embedder(cfg, secrets)
    elif cfg.type == "huggingface":
        return _huggingface_embedder(cfg)
    raise ValueError(f"Unsupported embedding provider: {cfg.type}")


def _google_embedder(cfg: EmbedderConfig, secrets: Secrets) -> Embeddings:
    """
    Helper to create a Google Generative AI embedder.

    Args:
        cfg (EmbedderConfig): Embedder configuration.
        secrets (Secrets): Secrets containing the Gemini API key.

    Returns:
        GoogleGenerativeAIEmbeddings instance.

    Raises:
        ValueError: If the Google API key is missing.
    """
    if not secrets.gemini_api_key:
        raise ValueError("Missing Google API key (gemini_api_key).")
    return GoogleGenerativeAIEmbeddings(
        model=cfg.model_name,
        google_api_key=secrets.gemini_api_key.get_secret_value(),
    )


def _huggingface_embedder(cfg: EmbedderConfig) -> Embeddings:
    """
    Helper to create a HuggingFace embedder.

    Args:
        cfg (EmbedderConfig): Embedder configuration.

    Returns:
        HuggingFaceEmbeddings instance.
    """
    return HuggingFaceEmbeddings(model_name=cfg.model_name)


# ----- FACTORY: VectorStore -----


def get_vectorstore(cfg: VectorstoreConfig, embedder: Embeddings):
    """
    Factory function to get the vectorstore instance based on the configuration.

    Args:
        cfg (VectorstoreConfig): Configuration for the vectorstore.
        embedder (Embeddings): Embedder instance to use.

    Returns:
        Vectorstore instance corresponding to the specified type.

    Raises:
        ValueError: If the vectorstore type is unsupported.
    """
    if cfg.type == "qdrant":
        return QdrantIndex(
            embedder=embedder,
            collection_name=cfg.collection_name,
            host=cfg.host,
            port=cfg.port,
        )
    raise ValueError(f"Unsupported vectorstore type: {cfg.type}")
