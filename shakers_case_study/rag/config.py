from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, SecretStr, field_validator

# ---- Configuration for each pipeline subcomponent ----


class LoaderConfig(BaseModel):
    type: Literal["markdown"]
    document_base_url: str = Field(default="http://localhost:8000")


class SplitterConfig(BaseModel):
    type: Literal["recursive"]
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_must_be_less_than_chunk(cls, v, info):
        """
        Validate that chunk_overlap is less than chunk_size.
        """
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and v >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return v


class EmbedderConfig(BaseModel):
    type: Literal["google"]
    model_name: str


class VectorstoreConfig(BaseModel):
    type: Literal["qdrant"]
    collection_name: str
    host: str = "localhost"
    port: int = 6333


class LLMConfig(BaseModel):
    type: Literal["google"]
    model_name: Literal["gemini-2.0-flash"]  # Avoid using other models (API costs)


# ---- General ingestion pipeline configuration ----


class IngestPipelineConfig(BaseModel):
    loader: LoaderConfig
    splitter: SplitterConfig
    embedder: EmbedderConfig
    vectorstore: VectorstoreConfig


# ---- Placeholder for future RAG stage ----


class RagPipelineConfig(BaseModel):
    vectorstore: VectorstoreConfig
    embedder: EmbedderConfig
    llm: LLMConfig


# ---- Overall pipeline configuration loading from YAML ----


class PipelineConfig(BaseModel):
    ingest: IngestPipelineConfig = None
    rag: Optional[RagPipelineConfig] = None


# ---- Secrets handling ----


class Secrets(BaseModel):
    gemini_api_key: Optional[SecretStr] = None

    @classmethod
    def from_env(cls):
        """
        Load secrets from environment variables using pydantic_settings.
        """
        from pydantic_settings import BaseSettings

        class EnvSettings(BaseSettings):
            gemini_api_key: Optional[SecretStr] = None

            class Config:
                env_file = ".env"
                extra = "ignore"

        s = EnvSettings()
        return cls(gemini_api_key=s.gemini_api_key)


# ---- Function to load pipeline config from YAML ----


def load_pipeline_config(path: str) -> PipelineConfig:
    """
    Load the pipeline configuration from a YAML file.

    Args:
        path (str): Path to the YAML config file.

    Returns:
        PipelineConfig: Parsed pipeline configuration.
    """
    with open(path, "r") as f:
        raw_cfg = yaml.safe_load(f)

    if "ingest" not in raw_cfg:
        raise ValueError(f"Missing 'ingest' section in {path}")
    if "rag" not in raw_cfg:
        raw_cfg["rag"] = {}

    return PipelineConfig(**raw_cfg)
