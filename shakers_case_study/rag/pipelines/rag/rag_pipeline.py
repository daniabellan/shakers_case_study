import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import psycopg
from backoff import expo, on_exception
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.ai import AIMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.store.postgres import PostgresStore
from pydantic import BaseModel
from ratelimit import RateLimitException, limits

from shakers_case_study.rag.pipelines.rag.prompts.prompts import (
    AMBIGUOUS_QUESTION_PROMPT,
    COMPANY_QA_PROMPT,
    INTENT_PROMPT,
    MALICIOUS_DETECTOR_PROMPT,
    NO_RESOURCES_FOUND_PROMPT,
    OUT_OF_SCOPE_PROMPT,
    RESOURCE_RECOMMENDATION_PROMPT,
    UNSAFE_FALLBACK_PROMPT,
)
from shakers_case_study.rag.vectorstore.qdrant_index import QdrantIndex
from shakers_case_study.utils.logging import get_logger

logger = get_logger()


class MyStateSchema(BaseModel):
    messages: List[Dict]
    intent: Optional[str] = None
    sentiment: Optional[str] = None
    moderation: Optional[str] = None
    current_node: Optional[str] = None
    metrics: Dict[str, float] = {}


class RAGPipeline:
    def __init__(
        self, vectorstore: QdrantIndex, llm: BaseChatModel, db_uri: str, embedder: Embeddings
    ):
        self.vectorstore = vectorstore
        self.llm = llm
        self.db_uri = db_uri
        self.embedder = embedder

        # Set up checkpoint and user history in BBDD
        self._initialize_checkpoint_memory()

        self.sentiment_tones = {
            "positive": "Be cheerful, warm, and encouraging in your response.",
            "neutral": "Be clear, professional, and concise in your response.",
            "negative": "Be empathetic, patient, and understanding in your response.",
        }

    def _initialize_checkpoint_memory(self):
        with (
            PostgresStore.from_conn_string(self.db_uri) as store,
            PostgresSaver.from_conn_string(self.db_uri) as checkpointer,
        ):
            store.setup()
            checkpointer.setup()

    def log_final_metrics(self, state: MyStateSchema, config: RunnableConfig, *, store):
        return state

    def log_llm_metrics(
        self,
        state: MyStateSchema,
        operation_name: str,
        start_time: float,
        response: Optional[Union[dict, AIMessage]] = None,
    ):
        latency = time.time() - start_time

        if response is None:
            usage = {}
        elif isinstance(response, dict):
            usage = response.get("usage_metadata", {})
        else:
            # Assume response is AIMessage and has usage_metadata attribute
            usage = getattr(response, "usage_metadata", {})

        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

        logger.info(f"[LLM METRICS] Operation: {operation_name}")
        logger.info(f"  Latency: {latency:.2f}s")
        logger.info(f"  Input Tokens: {input_tokens}")
        logger.info(f"  Output Tokens: {output_tokens}")
        logger.info(f"  Total Tokens: {total_tokens}")

        state.metrics["total_input_tokens"] = (
            state.metrics.get("total_input_tokens", 0) + input_tokens
        )
        state.metrics["total_output_tokens"] = (
            state.metrics.get("total_output_tokens", 0) + output_tokens
        )
        state.metrics["total_tokens"] = state.metrics.get("total_tokens", 0) + total_tokens
        state.metrics["llm_latency"] = state.metrics.get("llm_latency", 0.0) + latency

        return state

    def save_message(self, state: MyStateSchema, config: RunnableConfig, *, store):
        start_time = time.time()
        user_id = config["configurable"]["user_id"]
        thread_id = config["configurable"]["thread_id"]

        # Si el último mensaje es assistant y el penúltimo user, guardar ambos
        if (
            len(state.messages) >= 2
            and state.messages[-1]["role"] == "assistant"
            and state.messages[-2]["role"] == "user"
        ):
            messages_to_save = [state.messages[-2], state.messages[-1]]
        else:
            messages_to_save = [state.messages[-1]]

        for msg in messages_to_save:
            if msg["role"] == "assistant":
                current_node = state.current_node
            else:
                current_node = ""

            message_data = {
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "thread_id": thread_id,
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "node_name": current_node,
                "is_flagged": False,
                "metadata": {},
            }
            store.put((user_id, thread_id), message_data["id"], message_data)

        print(f"save_message time: {time.time()-start_time:.2f} secs")
        return state

    def sentiment_detection(self, state: MyStateSchema):
        start_time = time.time()

        user_message = state.messages[-1]["content"]

        # Simple prompt-based sentiment detection
        sentiment_prompt = f'Detect the sentiment of this message: "{user_message}". Return one word: Positive, Negative, or Neutral.'  # noqa: E501

        messages = [HumanMessage(content=sentiment_prompt)]
        response = self.llm.invoke(messages)
        state = self.log_llm_metrics(state, "sentiment_detection", start_time, response)

        sentiment = response.content.strip().lower()
        state.sentiment = sentiment

        return state

    def get_user_history(self, user_id, thread_id):
        start = time.time()
        with psycopg.connect(self.db_uri) as conn:
            prefix = f"{user_id}.{thread_id}"
            query = """
                SELECT value
                FROM store
                WHERE prefix = %s
                ORDER BY value->>'timestamp' ASC;
            """
            with conn.cursor() as cur:
                cur.execute(query, (prefix,))
                rows = cur.fetchall()

            user_history = [row[0] for row in rows]

        print(f"get_user_history time: {time.time()-start:.2f} secs")
        return user_history

    def build_user_profile(self, state: MyStateSchema, max_messages=10):
        start_time = time.time()
        user_messages = [
            m.get("content")
            for m in self.user_history
            if m.get("role") == "user" and m.get("content")
        ]
        recent_messages = user_messages[-max_messages:]

        if not recent_messages:
            dummy_embedding = self.embedder.embed_query("")
            return np.zeros_like(dummy_embedding)

        weights = np.arange(len(recent_messages), 0, -1)

        try:
            embeddings = self.embedder.embed_documents(recent_messages)
            embeddings = np.array(embeddings)
        except Exception as e:
            print(f"Error batch embedding: {e}")
            return np.zeros_like(self.cached_embed_query(""))

        if embeddings.shape[0] == 0:
            return np.zeros_like(self.cached_embed_query(""))

        weighted_embeddings = embeddings * weights[:, np.newaxis]
        profile_vector = weighted_embeddings.sum(axis=0) / weights.sum()

        state = self.log_llm_metrics(state, "sentiment_detection", start_time)

        return {"state": state, "profile_vector": profile_vector}

    def intent_detector(self, state: MyStateSchema, config: RunnableConfig, *, store):
        start_time = time.time()
        user_question = state.messages[-1]["content"]

        prompt = PromptTemplate(
            input_variables=["user_question"],
            template=INTENT_PROMPT,
        )

        formatted_prompt = prompt.format(
            user_question=user_question,
        )

        messages = [HumanMessage(content=formatted_prompt)]
        response = self.llm.invoke(messages)
        state = self.log_llm_metrics(state, "intent_detector", start_time, response)

        state.intent = response.content

        print(f"intent_detector time: {time.time()-start_time:.2f} secs")
        return state

    def malicious_query_detector(self, state: MyStateSchema, config: RunnableConfig, *, store):
        start_time = time.time()
        user_question = state.messages[-1]["content"]

        prompt = PromptTemplate(
            input_variables=["user_question"],
            template=MALICIOUS_DETECTOR_PROMPT,
        )

        formatted_prompt = prompt.format(
            user_question=user_question,
        )

        messages = [HumanMessage(content=formatted_prompt)]
        response = self.llm.invoke(messages)
        state = self.log_llm_metrics(state, "malicious_query_detector", start_time, response)

        state.moderation = response.content

        print(f"malicious_query_detector time: {time.time()-start_time:.2f} secs")
        return state

    def unsafe_fallback(self, state: MyStateSchema, config: RunnableConfig, *, store):
        start_time = time.time()
        user_question = state.messages[-1]["content"]

        prompt = PromptTemplate(
            input_variables=["user_question"],
            template=UNSAFE_FALLBACK_PROMPT,
        )

        formatted_prompt = prompt.format(
            user_question=user_question,
        )

        messages = [HumanMessage(content=formatted_prompt)]
        response = self.llm.invoke(messages)
        state = self.log_llm_metrics(state, "unsafe_fallback", start_time, response)

        state.current_node = "unsafe_fallback"
        state.messages.append({"role": "assistant", "content": response.content})

        return state

    def out_of_scope_answer(self, state: MyStateSchema, config: RunnableConfig, *, store):
        start_time = time.time()
        user_question = state.messages[-1]["content"]

        prompt = PromptTemplate(
            input_variables=["user_question"],
            template=OUT_OF_SCOPE_PROMPT,
        )

        formatted_prompt = prompt.format(
            user_question=user_question,
        )

        messages = [HumanMessage(content=formatted_prompt)]
        response = self.llm.invoke(messages)
        state = self.log_llm_metrics(state, "out_of_scope_answer", start_time, response)

        state.current_node = "out_of_scope"
        state.messages.append({"role": "assistant", "content": response.content})

        return state

    def ambiguous_question_answer(self, state: MyStateSchema, config: RunnableConfig, *, store):
        start_time = time.time()
        user_question = state.messages[-1]["content"]

        prompt = PromptTemplate(
            input_variables=["user_question"],
            template=AMBIGUOUS_QUESTION_PROMPT,
        )

        formatted_prompt = prompt.format(
            user_question=user_question,
        )

        messages = [HumanMessage(content=formatted_prompt)]
        response = self.llm.invoke(messages)
        state = self.log_llm_metrics(state, "out_of_scope_answer", start_time, response)

        state.current_node = "ambiguous_question"
        state.messages.append({"role": "assistant", "content": response.content})

        return state

    def explain_recommendation(
        self,
        state: MyStateSchema,
        sentiment_tone: str,
        combined_resource_str: str,
        user_query: str,
        user_profile_summary: str,
    ) -> str:
        """
        Usa el LLM para generar una explicación personalizada de por qué se recomienda este recurso.
        """
        start_time = time.time()
        prompt_template = PromptTemplate(
            input_variables=[
                "combined_resource_str",
                "user_query",
                "user_profile_summary",
                "sentiment_tone",
            ],
            template=RESOURCE_RECOMMENDATION_PROMPT,
        )

        # Preparar la explicación
        formatted_prompt = prompt_template.format(
            combined_resource_str=combined_resource_str,
            user_query=user_query,
            user_profile_summary=user_profile_summary,
            sentiment_tone=sentiment_tone,
        )

        messages = [HumanMessage(content=formatted_prompt)]
        response = self.llm.invoke(messages)
        state = self.log_llm_metrics(state, "explain_recommendation", start_time, response)

        return {"state": state, "recommendation_explanation": response.content}

    def build_user_profile_summary(self, user_id, thread_id, max_messages=10) -> str:
        """
        Opcional: Genera un resumen en texto natural del perfil del usuario
        basado en sus mensajes para dar contexto al LLM.
        """
        user_messages = [
            m.get("content")
            for m in self.user_history
            if m.get("role") == "user" and m.get("content")
        ]

        recent_messages = user_messages[-max_messages:]
        if not recent_messages:
            return "No previous user information"

        # Podrías concatenar y/o hacer un resumen simple
        summary = "Recent user questions: " + ", ".join(set(recent_messages))
        return summary

    @lru_cache(maxsize=1000)
    def cached_embed_query(self, text):
        return self.embedder.embed_query(text)

    def recommend_resources(
        self,
        state: MyStateSchema,
        user_question: str,
        alpha: float = 0.7,
        top_k: int = 10,
        n_recommendations: int = 3,
        diversity_threshold: float = 0.85,
    ) -> List[Tuple[float, dict]]:
        """
        Devuelve lista de (score, recurso_dict) donde recurso_dict tiene keys 'id' y 'text' mínimo.
        """
        start_time = time.time()

        profile_vector_payload = self.build_user_profile(state)
        state = profile_vector_payload["state"]
        profile_vector = profile_vector_payload["profile_vector"]

        question_embedding = np.array(self.cached_embed_query(user_question))

        combined_vector = alpha * question_embedding + (1 - alpha) * profile_vector
        norm = np.linalg.norm(combined_vector)
        combined_vector = combined_vector / norm if norm != 0 else combined_vector

        client = self.vectorstore._get_client()
        search_results = client.search(
            collection_name=self.vectorstore.collection_name,
            query_vector=combined_vector.tolist(),
            limit=top_k,
            with_payload=True,
            with_vectors=True,
        )

        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        recommendations = []
        embeddings_selected = []

        start_diversity = time.time()
        for hit in search_results:
            doc_vec = np.array(hit.vector)
            if any(cosine_sim(doc_vec, v) > diversity_threshold for v in embeddings_selected):
                continue

            score = cosine_sim(combined_vector, doc_vec)

            payload = hit.payload or {}
            metadata = payload.get("metadata", {})
            resource_dict = {
                "source_file": metadata.get("source_file", "desconocido"),
                "page_content": payload.get("page_content", ""),
            }

            recommendations.append((score, resource_dict))
            embeddings_selected.append(doc_vec)

            if len(recommendations) >= n_recommendations:
                break

            print(f"start_diversity time: {time.time()-start_diversity:.2f} secs")

        state = self.log_llm_metrics(state, "out_of_scope_answer", start_time)
        return {"state": state, "recommendations": recommendations}

    # Nodo que procesa input usuario y genera respuesta con LLM
    def llm_reply(
        self,
        state: MyStateSchema,
        config: RunnableConfig,
        max_hist_messages: int = 10,
    ):
        start_time = time.time()
        user_question = state.messages[-1]["content"]
        user_id = config["configurable"]["user_id"]
        thread_id = config["configurable"]["thread_id"]

        sentiment_tone = self.sentiment_tones.get(state.sentiment, self.sentiment_tones["neutral"])

        # Retrieval
        resources = self.vectorstore.similarity_search_with_score(query=user_question)

        self.user_history = self.get_user_history(user_id, thread_id)

        # Fallback: verificar si hay documentos
        if resources:
            company_info = "\n".join(
                f"""<document>  # noqa: E501
        <source>{doc[0].metadata.get('source_file', 'Unknown Resource').replace('_', ' ').title()}</source>
        <content>{doc[0].page_content.strip()}</content>
        </document>"""
                for doc in resources
            )
            full_prompt_str = COMPANY_QA_PROMPT
            previous_context = "\n".join(
                f'{msg["role"].capitalize()}: {msg["content"]}'
                for msg in self.user_history[-max_hist_messages:]
            )
        else:
            # Prompt alternativo si no se encuentran recursos relevantes
            company_info = ""
            previous_context = ""
            full_prompt_str = NO_RESOURCES_FOUND_PROMPT

        full_prompt_str = full_prompt_str.format(
            company_info=company_info,
            sentiment_tone=sentiment_tone,
        )

        prompt = PromptTemplate(
            input_variables=["user_question", "previous_context"],
            template=full_prompt_str,
        )
        formatted_prompt = prompt.format(
            user_question=user_question,
            previous_context=previous_context,
        )

        messages = [HumanMessage(content=formatted_prompt)]
        llm_response = self.llm.invoke(messages)

        # Recommendation system
        recommended_payload = self.recommend_resources(state, user_question)
        recommended = recommended_payload["recommendations"]
        state = recommended_payload["state"]

        high_sim_recommendations = [(score, r) for score, r in recommended if score > 0.25]

        user_profile_summary = self.build_user_profile_summary(user_id, thread_id)

        # Armar una lista de recursos con índice
        resource_blocks = []
        for i, (score, resource) in enumerate(high_sim_recommendations, 1):
            block = f"""Resource {i}:
        Title: {resource.get("source_file", "Unknown")}
        Summary: {resource.get("page_content", "No content")}
        Similarity Score: {score:.2f}
        """
            resource_blocks.append(block)

        combined_resource_str = "\n".join(resource_blocks)
        explanations = self.explain_recommendation(
            state, sentiment_tone, combined_resource_str, user_question, user_profile_summary
        )
        explanations = []
        for score, resource in high_sim_recommendations:
            explanation_payload = self.explain_recommendation(
                state, sentiment_tone, resource, user_question, user_profile_summary
            )

            state = explanation_payload["state"]
            explanation = explanation_payload["recommendation_explanation"]

            explanations.append((resource, score, explanation))

        # Construir respuesta para el usuario
        response_text = "Recommendations:\n"
        for res, score, explanation in explanations:  # noqa: E501
            response_text += f"\n- {res['source_file'].replace('_', ' ').title()} (Cosine similarity: {score:.2f}):\n{explanation}\n"  # noqa: E501

        final_response = llm_response.content + "\n\n" + response_text

        state.current_node = "question_answer"
        state.messages.append({"role": "assistant", "content": final_response})

        state = self.log_llm_metrics(state, "llm_reply", start_time)
        return state

    # 10 requests per minute per user
    @on_exception(expo, RateLimitException, max_tries=3)
    @limits(calls=10, period=60)
    def run(self) -> str:
        with (
            PostgresStore.from_conn_string(self.db_uri) as store,
            PostgresSaver.from_conn_string(self.db_uri) as checkpointer,
        ):
            graph = StateGraph(state_schema=MyStateSchema)

            # Añadir todos los nodos
            graph.add_node("malicious_query_detector", self.malicious_query_detector)
            graph.add_node("intent_detector", self.intent_detector)
            graph.add_node("ambiguous_question_answer", self.ambiguous_question_answer)
            graph.add_node("out_of_scope_answer", self.out_of_scope_answer)
            graph.add_node("save_message", self.save_message)
            graph.add_node("llm_reply", self.llm_reply)
            graph.add_node("sentiment_detection", self.sentiment_detection)
            graph.add_node("unsafe_fallback", self.unsafe_fallback)
            graph.add_node("log_final_metrics", self.log_final_metrics)

            graph.set_entry_point("malicious_query_detector")

            def route_from_malicious(state: MyStateSchema):
                moderation = (state.moderation or "").lower()
                if "unsafe" in moderation or "malicious" in moderation or "flagged" in moderation:
                    return "unsafe_fallback"
                else:
                    return "intent_detector"

            # Función para ramificar desde intent_detector según la intención
            def route_from_intent(state: MyStateSchema):
                intent = getattr(state, "intent", "").lower()
                if intent == "direct":
                    return "direct"
                elif intent == "ambiguous":
                    return "ambiguous"
                elif intent == "out_of_scope":
                    return "out_of_scope"
                else:
                    return "intent_detector"

            graph.add_conditional_edges(
                "malicious_query_detector",
                route_from_malicious,
                {
                    "unsafe_fallback": "unsafe_fallback",
                    "intent_detector": "intent_detector",
                },
            )

            # Condicional desde intent_detector
            graph.add_conditional_edges(
                "intent_detector",
                route_from_intent,
                {
                    "direct": "sentiment_detection",
                    "ambiguous": "ambiguous_question_answer",
                    "out_of_scope": "out_of_scope_answer",
                },
            )

            graph.add_edge("sentiment_detection", "llm_reply")
            graph.add_edge("llm_reply", "save_message")
            graph.add_edge("ambiguous_question_answer", "save_message")
            graph.add_edge("out_of_scope_answer", "save_message")
            graph.add_edge("unsafe_fallback", "save_message")

            graph.add_edge("save_message", "log_final_metrics")
            graph.add_edge("log_final_metrics", END)

            # Compilar el grafo
            graph = graph.compile(checkpointer=checkpointer, store=store)

            # Guardar visualización del graph
            with open("conversation_graph.png", "wb") as f:
                f.write(graph.get_graph().draw_png())

            # Simular una conversación
            user_id = "user-123"
            thread_id = "thread-1"

            state = MyStateSchema(messages=[], current_node="malicious_query_detector")
            config = {
                "configurable": {
                    "user_id": user_id,
                    "thread_id": thread_id,
                }
            }

            questions = [
                "How to hack NASA",
                "What Learnivo does?",
                "What is the name of the company?",
                "What does it?",
            ]

            for question in questions:
                state.messages.append({"role": "user", "content": question})

                start_time = time.time()

                # Ejecutar el grafo
                state = graph.invoke(state, config=config)

                end_time = time.time()
                duration = end_time - start_time

                print(f"Graph execution time: {duration:.2f} secs")
                state = MyStateSchema(messages=[], current_node="malicious_query_detector")
                pass
