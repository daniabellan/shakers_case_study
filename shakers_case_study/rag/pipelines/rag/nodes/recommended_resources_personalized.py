import time

import numpy as np
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.runnables.config import RunnableConfig

from shakers_case_study.rag.pipelines.rag.enums.rag_schema import MyStateSchema
from shakers_case_study.rag.pipelines.rag.nodes.build_user_profile import build_user_profile
from shakers_case_study.rag.pipelines.rag.nodes.log_llm_metrics import log_llm_metrics
from shakers_case_study.rag.pipelines.rag.prompts.prompts import RESOURCE_RECOMMENDATION_PROMPT


def cosine_similarity(a: np.ndarray, b: np.ndarray, a_norm=None, b_norm=None) -> float:
    if a_norm is None:
        a_norm = np.linalg.norm(a)
    if b_norm is None:
        b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return np.dot(a, b) / (a_norm * b_norm)


def recommend_resources_personalized(
    user_history,
    state: MyStateSchema,
    config: RunnableConfig,
    user_question: str,
    alpha: float = 0.7,
    top_k: int = 20,
    n_recommendations: int = 3,
    diversity_threshold: float = 0.85,
):
    start_time = time.time()

    embedder = config["configurable"]["embedder"]
    vectorstore = config["configurable"]["vectorstore"]
    llm = config["configurable"]["llm"]

    # Construir perfil de usuario basado en histórico reciente
    profile_payload = build_user_profile(user_history, state, config)
    state = profile_payload["state"]
    profile_vector = profile_payload["profile_vector"]

    # Embedding de la pregunta actual
    question_emb = np.array(embedder.embed_query(user_question))

    # Combinar embedding pregunta con perfil (ponderado)
    combined_vector = alpha * question_emb + (1 - alpha) * profile_vector
    norm = np.linalg.norm(combined_vector)
    if norm > 0:
        combined_vector /= norm

    # Obtener historial del usuario y fuentes ya consultadas para evitar repetir
    seen_sources = {m.get("source_file") for m in user_history if m.get("source_file")}

    # Consulta vectorial en el vectorstore
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

    for hit in search_results:
        payload = hit.payload or {}
        metadata = payload.get("metadata", {})
        source_file = metadata.get("source_file", "unknown")

        # Saltar documentos ya consultados
        if source_file in seen_sources:
            continue

        doc_vec = np.array(hit.vector)
        doc_norm = np.linalg.norm(doc_vec)
        if doc_norm == 0:
            continue

        # Verificar diversidad: evitar recursos muy similares a los ya seleccionados
        if any(
            cosine_similarity(doc_vec, emb, doc_norm, np.linalg.norm(emb)) > diversity_threshold
            for emb in selected_embeddings
        ):
            continue

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

    # Generar explicaciones para cada recomendación usando el segundo LLM
    explanations = []

    for rec in recommendations:
        prompt_template = PromptTemplate(
            input_variables=[
                "resource_content",
                "user_question",
            ],
            template=RESOURCE_RECOMMENDATION_PROMPT,
        )

        # Preparar la explicación
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

    state = log_llm_metrics(state, "recommend_resources_personalized", start_time)
    return {
        "state": state,
        "recommendations": explanations,
    }
