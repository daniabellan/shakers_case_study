import os

from shakers_case_study.rag.config import Secrets, load_pipeline_config
from shakers_case_study.rag.ingestion.component_factory import (
    get_embedder,
    get_llm,
    get_vectorstore,
)
from shakers_case_study.rag.pipelines.rag.rag_pipeline import RAGPipeline
from shakers_case_study.utils.logging import get_logger, setup_logging

setup_logging(env=os.getenv("ENV", "development"), config_filename="configs/logging_config.yaml")

logger = get_logger()


def build_rag_pipeline(config_path: str) -> RAGPipeline:
    config = load_pipeline_config(config_path)
    secrets = Secrets.from_env()

    embedder = get_embedder(config.rag.embedder, secrets)
    vectorstore = get_vectorstore(config.rag.vectorstore, embedder).load()
    llm = get_llm(config.rag.llm, secrets)

    db_uri = "postgresql://shakers_case_study:shakers_case_study@localhost:5432/shakers_case_study?sslmode=disable"  # noqa: E501
    pipeline = RAGPipeline(vectorstore, llm, db_uri, embedder)

    logger.info("Loaded RAG pipeline")

    return pipeline


if __name__ == "__main__":
    config_path = "pipeline_configs/standard_pipeline.yaml"
    pipeline = build_rag_pipeline(config_path)

    pipeline.profile_population()
