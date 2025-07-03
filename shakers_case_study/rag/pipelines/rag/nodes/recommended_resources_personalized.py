import time
from typing import Any, Dict, List

import numpy as np
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.runnables.config import RunnableConfig

from shakers_case_study.rag.pipelines.rag.enums.rag_schema import MyStateSchema
from shakers_case_study.rag.pipelines.rag.nodes.build_user_profile import build_user_profile
from shakers_case_study.rag.pipelines.rag.nodes.log_llm_metrics import log_llm_metrics
from shakers_case_study.rag.pipelines.rag.prompts.prompts import RESOURCE_RECOMMENDATION_PROMPT


def cosine_similarity(
    a: np.ndarray, b: np.ndarray, a_norm: float = None, b_norm: float = None
) -> float:
    """
    Compute the cosine similarity between two vectors a and b.

    Args:
        a (np.ndarray): Vector a.
        b (np.ndarray): Vector b.
        a_norm (float, optional): Precomputed norm of vector a. If None, it will be computed.
        b_norm (float, optional): Precomputed norm of vector b. If None, it will be computed.

    Returns:
        float: Cosine similarity value between -1 and 1, or 0 if any vector has zero norm.
    """
    if a_norm is None:
        a_norm = np.linalg.norm(a)
    if b_norm is None:
        b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return np.dot(a, b) / (a_norm * b_norm)


def recommend_resources_personalized(
    user_history: List[Dict[str, Any]],
    state: MyStateSchema,
    config: RunnableConfig,
    user_question: str,
    alpha: float = 0.7,
    top_k: int = 20,
    n_recommendations: int = 3,
    diversity_threshold: float = 0.85,
) -> Dict[str, Any]:
    """
    Generate personalized resource recommendations based on user history, question embedding,
    and content diversity, then provide explanation for each recommended resource using an LLM.

    Args:
        user_history (List[Dict]): List of historical user messages with metadata.
        state (MyStateSchema): Current pipeline state.
        config (RunnableConfig): Configuration holding embedder, vectorstore, and LLM.
        user_question (str): Current user question string.
        alpha (float, optional): Weighting factor between question embedding and user
        profile vector.
        top_k (int, optional): Number of top documents to retrieve from vectorstore.
        n_recommendations (int, optional): Number of final diverse recommendations to select.
        diversity_threshold (float, optional): Cosine similarity threshold to enforce diversity.

    Returns:
        Dict[str, Any]: Dictionary with updated state and list of recommendations with explanations.
    """
    start_time = time.time()

    embedder = config["configurable"]["embedder"]
    vectorstore = config["configurable"]["vectorstore"]
    llm = config["configurable"]["llm"]

    # Build user profile vector based on recent history
    profile_payload = build_user_profile(user_history, state, config)
    state = profile_payload["state"]
    profile_vector = profile_payload["profile_vector"]

    # Embed the current question
    question_emb = np.array(embedder.embed_query(user_question))

    # Combine question embedding with profile vector (weighted average)
    combined_vector = alpha * question_emb + (1 - alpha) * profile_vector
    norm = np.linalg.norm(combined_vector)
    if norm > 0:
        combined_vector /= norm

    # Track previously seen sources to avoid duplicate recommendations
    seen_sources = {msg.get("source_file") for msg in user_history if msg.get("source_file")}

    # Query vectorstore for top_k similar documents
    client = vectorstore._get_client()
    search_results = client.search(
        collection_name=vectorstore.collection_name,
        query_vector=combined_vector.tolist(),
        limit=top_k,
        with_payload=True,
        with_vectors=True,
    )

    recommendations = []
    selected_embeddings = []
    combined_norm = np.linalg.norm(combined_vector)

    # Select diverse and unseen recommendations based on cosine similarity
    for hit in search_results:
        payload = hit.payload or {}
        metadata = payload.get("metadata", {})
        source_file = metadata.get("source_file", "unknown")

        # Skip already seen documents
        if source_file in seen_sources:
            continue

        doc_vec = np.array(hit.vector)
        doc_norm = np.linalg.norm(doc_vec)
        if doc_norm == 0:
            continue

        # Skip if document too similar to already selected ones (diversity filter)
        if any(
            cosine_similarity(doc_vec, emb, doc_norm, np.linalg.norm(emb)) > diversity_threshold
            for emb in selected_embeddings
        ):
            continue

        # Compute similarity score with combined user vector
        score = cosine_similarity(combined_vector, doc_vec, combined_norm, doc_norm)

        recommendations.append(
            {
                "source_file": source_file,
                "page_content": payload.get("page_content", ""),
                "score": score,
            }
        )
        selected_embeddings.append(doc_vec)

        if len(recommendations) >= n_recommendations:
            break

    # Use LLM to generate explanations for each recommended resource
    explanations = []
    for rec in recommendations:
        prompt_template = PromptTemplate(
            input_variables=["resource_content", "user_question"],
            template=RESOURCE_RECOMMENDATION_PROMPT,
        )
        formatted_prompt = prompt_template.format(
            resource_content=rec["page_content"],
            user_question=user_question,
        )
        messages = [HumanMessage(content=formatted_prompt)]
        response = llm.invoke(messages)

        explanations.append(
            {
                "source_file": rec["source_file"],
                "score": rec["score"],
                "explanation": response.content.strip(),
            }
        )

    # Log metrics related to this recommendation operation
    state = log_llm_metrics(state, "recommend_resources_personalized", start_time)

    return {
        "state": state,
        "recommendations": explanations,
    }
