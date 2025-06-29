import os

from shakers_case_study.rag.config import Secrets, load_pipeline_config
from shakers_case_study.rag.ingestion.component_factory import (
    get_embedder,
    get_loader,
    get_splitter,
    get_vectorstore,
)
from shakers_case_study.rag.pipelines.ingestion.ingestion_pipeline import IngestionPipeline
from shakers_case_study.utils.logging import get_logger, setup_logging

setup_logging(env=os.getenv("ENV", "development"), config_filename="configs/logging_config.yaml")

logger = get_logger("run_ingestion")


def build_ingestion_pipeline(config_path: str) -> IngestionPipeline:
    """
    Build the ingestion pipeline from a configuration file.

    Args:
        config_path (str): Path to the pipeline configuration YAML file.

    Returns:
        IngestionPipeline: Configured ingestion pipeline instance.
    """
    config = load_pipeline_config(config_path)
    secrets = Secrets.from_env()

    loader = get_loader(config.ingest.loader)
    splitter = get_splitter(config.ingest.splitter)
    embedder = get_embedder(config.ingest.embedder, secrets)
    vectorstore = get_vectorstore(config.ingest.vectorstore, embedder)

    pipeline = IngestionPipeline(loader, splitter, embedder, vectorstore)
    logger.info(f"Loaded pipeline: {type(pipeline).__name__}")

    return pipeline


if __name__ == "__main__":
    config_path = "pipeline_configs/standard_pipeline.yaml"
    pipeline = build_ingestion_pipeline(config_path)
    pipeline.run()
