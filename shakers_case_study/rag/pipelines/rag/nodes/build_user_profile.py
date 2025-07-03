import time

import numpy as np
from langchain_core.runnables.config import RunnableConfig

from shakers_case_study.rag.pipelines.rag.enums.rag_schema import MyStateSchema
from shakers_case_study.rag.pipelines.rag.nodes.get_user_history import get_recent_user_messages
from shakers_case_study.rag.pipelines.rag.nodes.log_llm_metrics import log_llm_metrics
from shakers_case_study.utils.logging import get_logger

logger = get_logger()


def build_user_profile(user_history, state: MyStateSchema, config: RunnableConfig, max_messages=10):
    embedder = config["configurable"]["embedder"]
    start_time = time.time()

    recent_messages = get_recent_user_messages(user_history, max_messages)

    if not recent_messages:
        empty_embedding = embedder.embed_query("")
        return {"state": state, "profile_vector": np.zeros_like(empty_embedding)}

    # Decay weights, mensajes más recientes pesan más
    recency_weights = np.arange(len(recent_messages), 0, -1)

    try:
        embeddings = embedder.embed_documents(recent_messages)
        embeddings = np.array(embeddings)
    except Exception as e:
        logger.error(f"Error during batch embedding: {e}")
        empty_embedding = embedder.embed_query("")
        return {"state": state, "profile_vector": np.zeros_like(empty_embedding)}

    if embeddings.shape[0] == 0:
        empty_embedding = embedder.embed_query("")
        return {"state": state, "profile_vector": np.zeros_like(empty_embedding)}

    # Embedding ponderado por recencia
    weighted_embeddings = embeddings * recency_weights[:, np.newaxis]
    profile_vector = weighted_embeddings.sum(axis=0) / recency_weights.sum()

    # Loggear métrica genérica (usar nombre de operación adecuado)
    state = log_llm_metrics(state, "build_user_profile", start_time)

    return {"state": state, "profile_vector": profile_vector}
