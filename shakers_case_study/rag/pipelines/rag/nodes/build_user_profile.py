import time

import numpy as np
from langchain_core.runnables.config import RunnableConfig

from shakers_case_study.rag.pipelines.rag.enums.rag_schema import MyStateSchema
from shakers_case_study.rag.pipelines.rag.nodes.get_user_history import get_recent_user_messages
from shakers_case_study.rag.pipelines.rag.nodes.log_llm_metrics import log_llm_metrics
from shakers_case_study.utils.logging import get_logger

logger = get_logger()


def build_user_profile(
    user_history: list,
    state: MyStateSchema,
    config: RunnableConfig,
    max_messages: int = 10,
) -> dict:
    """
    Builds a weighted user profile embedding vector based on recent user messages.

    The function retrieves the most recent user messages, computes their embeddings,
    applies a recency-based weighting (more recent messages weigh more),
    and returns a profile vector representing the user.

    Args:
        user_history (list): Full history of user messages (strings).
        state (MyStateSchema): Current pipeline state object.
        config (RunnableConfig): Configuration containing the embedder and other components.
        max_messages (int, optional): Maximum number of recent messages to consider. Defaults to 10.

    Returns:
        dict: Contains updated 'state' and computed 'profile_vector' (numpy.ndarray).
              If no recent messages exist or embedding fails, returns a zero vector of
              embedding size.
    """
    embedder = config["configurable"]["embedder"]
    start_time = time.time()

    # Retrieve recent user messages limited by max_messages
    recent_messages = get_recent_user_messages(user_history, max_messages)

    # Handle case with no recent messages by returning zero vector
    if not recent_messages:
        empty_embedding = embedder.embed_query("")
        return {"state": state, "profile_vector": np.zeros_like(empty_embedding)}

    # Create recency weights: more recent messages get higher weight
    recency_weights = np.arange(len(recent_messages), 0, -1)

    try:
        # Compute embeddings for recent messages as a batch
        embeddings = embedder.embed_documents(recent_messages)
        embeddings = np.array(embeddings)
    except Exception as e:
        logger.error(f"Error during batch embedding: {e}")
        empty_embedding = embedder.embed_query("")
        return {"state": state, "profile_vector": np.zeros_like(empty_embedding)}

    # Handle case of empty embeddings
    if embeddings.shape[0] == 0:
        empty_embedding = embedder.embed_query("")
        return {"state": state, "profile_vector": np.zeros_like(empty_embedding)}

    # Compute weighted average embedding by recency
    weighted_embeddings = embeddings * recency_weights[:, np.newaxis]
    profile_vector = weighted_embeddings.sum(axis=0) / recency_weights.sum()

    # Log the operation latency metric with an appropriate operation name
    state = log_llm_metrics(state, operation_name="build_user_profile", start_time=start_time)

    return {"state": state, "profile_vector": profile_vector}
