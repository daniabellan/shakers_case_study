import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import psycopg
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.store.postgres import PostgresStore
from pydantic import BaseModel

from shakers_case_study.rag.pipelines.rag.prompts.prompts import (
    AMBIGUOUS_QUESTION_PROMPT,
    COMPANY_QA_PROMPT,
    INTENT_PROMPT,
    OUT_OF_SCOPE_PROMPT,
    RESOURCE_RECOMMENDATION_PROMPT,
)
from shakers_case_study.rag.vectorstore.qdrant_index import QdrantIndex
from shakers_case_study.utils.logging import get_logger

logger = get_logger()


class MyStateSchema(BaseModel):
    messages: List[Dict]
    intent: Optional[str] = None
    sentiment: Optional[str] = None


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

    def save_message(self, state: MyStateSchema, config: RunnableConfig, *, store):
        user_id = config["configurable"]["user_id"]
        thread_id = config["configurable"]["thread_id"]
        current_node = config["configurable"].get("current_node", "unknown")

        last_message = state.messages[-1]

        message_data = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "thread_id": thread_id,
            "role": last_message["role"],
            "content": last_message["content"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_name": current_node,
            "is_flagged": False,
            "metadata": {},
        }

        store.put((user_id, thread_id), message_data["id"], message_data)
        return state

    def sentiment_detection(self, state: MyStateSchema, config: RunnableConfig, *, store):
        user_message = state.messages[-1]["content"]

        # Simple prompt-based sentiment detection
        sentiment_prompt = f'Detect the sentiment of this message: "{user_message}". Return one word: Positive, Negative, or Neutral.'  # noqa: E501

        messages = [HumanMessage(content=sentiment_prompt)]
        response = self.llm.invoke(messages)

        sentiment = response.content.strip().lower()
        state.sentiment = sentiment

        return state

    def get_user_history(self, user_id, thread_id):
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

        return user_history

    def build_user_profile(self, user_id, thread_id, max_messages=10):
        user_history = self.get_user_history(user_id, thread_id)
        user_messages = [
            m.get("content") for m in user_history if m.get("role") == "user" and m.get("content")
        ]

        recent_messages = user_messages[-max_messages:]

        if not recent_messages:
            dummy_embedding = self.embedder.embed_query("")
            return np.zeros_like(dummy_embedding)

        weights = list(range(1, len(recent_messages) + 1))
        weights.reverse()

        weighted_embeddings = []
        for weight, msg in zip(weights, recent_messages):
            try:
                emb = self.embedder.embed_query(msg)
                emb_vec = np.array(emb)
                weighted_embeddings.append(emb_vec * weight)
            except Exception as e:
                print(f"Error embedding message: {msg[:30]}... - {e}")

        if not weighted_embeddings:
            return np.zeros_like(self.embedder.embed_query(""))

        profile_vector = np.sum(weighted_embeddings, axis=0) / sum(weights)

        return profile_vector

    def intent_detector(self, state: MyStateSchema, config: RunnableConfig, *, store):
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

        state.intent = response.content

        return state

    def out_of_scope_answer(self, state: MyStateSchema, config: RunnableConfig, *, store):
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

        state.messages.append({"role": "assistant", "content": response.content})

        return state

    def ambiguous_question_answer(self, state: MyStateSchema, config: RunnableConfig, *, store):
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

        state.messages.append({"role": "assistant", "content": response.content})

        return state

    def explain_recommendation(
        self, resource: dict, user_query: str, user_profile_summary: str
    ) -> str:
        """
        Usa el LLM para generar una explicación personalizada de por qué se recomienda este recurso.
        """
        prompt_template = PromptTemplate(
            input_variables=[
                "resource_title",
                "resource_summary",
                "user_query",
                "user_profile_summary",
            ],
            template=RESOURCE_RECOMMENDATION_PROMPT,
        )

        # Preparar la explicación
        formatted_prompt = prompt_template.format(
            resource_title=resource.get("source_file", "Recurso sin título"),
            resource_summary=resource.get("page_content", "Sin resumen"),
            user_query=user_query,
            user_profile_summary=user_profile_summary,
        )

        messages = [HumanMessage(content=formatted_prompt)]
        response = self.llm.invoke(messages)

        return response.content

    def build_user_profile_summary(self, user_id, thread_id, max_messages=10) -> str:
        """
        Opcional: Genera un resumen en texto natural del perfil del usuario
        basado en sus mensajes para dar contexto al LLM.
        """
        user_history = self.get_user_history(user_id, thread_id)
        user_messages = [
            m.get("content") for m in user_history if m.get("role") == "user" and m.get("content")
        ]

        recent_messages = user_messages[-max_messages:]
        if not recent_messages:
            return "No previous user information"

        # Podrías concatenar y/o hacer un resumen simple
        summary = "Recent user questions: " + ", ".join(set(recent_messages))
        return summary

    def recommend_resources(
        self,
        user_id: str,
        thread_id: str,
        user_question: str,
        alpha: float = 0.7,
        top_k: int = 10,
        n_recommendations: int = 3,
        diversity_threshold: float = 0.85,
    ) -> List[Tuple[float, dict]]:
        """
        Devuelve lista de (score, recurso_dict) donde recurso_dict tiene keys 'id' y 'text' mínimo.
        """

        profile_vector = self.build_user_profile(user_id, thread_id)
        question_embedding = np.array(self.embedder.embed_query(user_question))
        combined_vector = alpha * question_embedding + (1 - alpha) * profile_vector
        combined_vector /= np.linalg.norm(combined_vector)

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

        return recommendations

    # Nodo que procesa input usuario y genera respuesta con LLM
    def llm_reply(
        self,
        state: MyStateSchema,
        config: RunnableConfig,
        *,
        store,
        alpha: float = 0.7,
        max_hist_messages: int = 10,
    ):
        user_question = state.messages[-1]["content"]
        user_id = config["configurable"]["user_id"]
        thread_id = config["configurable"]["thread_id"]

        sentiment_tone = self.sentiment_tones.get(state.sentiment, self.sentiment_tones["neutral"])

        # Retrieval
        self.resources = self.vectorstore.similarity_search(user_question)

        # Answer user question
        # Formatear el prompt para la respuesta general del LLM
        company_info = "\n\n".join(
            f"# {doc.metadata.get('source_file', 'Unknown Resource').replace('_', ' ').title()}\n\n{doc.page_content.strip()}"  # noqa: E501
            for doc in self.resources
        )
        user_history = self.get_user_history(user_id, thread_id)
        previous_context = "\n".join(
            f'{msg["role"].capitalize()}: {msg["content"]}'
            for msg in user_history[-max_hist_messages:]
        )
        full_prompt_str = COMPANY_QA_PROMPT.format(
            company_info=company_info, sentiment_tone=sentiment_tone
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
        recommended = self.recommend_resources(user_id, thread_id, user_question)
        high_sim_recommendations = [(score, r) for score, r in recommended if score > 0.25]

        user_profile_summary = self.build_user_profile_summary(user_id, thread_id)

        explanations = []
        for score, resource in high_sim_recommendations:
            explanation = self.explain_recommendation(
                sentiment_tone, resource, user_question, user_profile_summary
            )
            explanations.append((resource, score, explanation))

        # Construir respuesta para el usuario que incluye las recomendaciones y las explicaciones
        response_text = "Recommendations:\n"
        for res, score, explanation in explanations:
            response_text += f"\n- {res['source_file'].replace('_', ' ').title()} (Similitud: {score:.2f}):\n{explanation}\n"  # noqa: E501

        final_response = llm_response.content + "\n\n" + response_text

        state.messages.append({"role": "assistant", "content": final_response})
        return state

    def run(self) -> str:
        with (
            PostgresStore.from_conn_string(self.db_uri) as store,
            PostgresSaver.from_conn_string(self.db_uri) as checkpointer,
        ):
            graph = StateGraph(state_schema=MyStateSchema)

            # Añadir todos los nodos
            graph.add_node("intent_detector", self.intent_detector)
            graph.add_node("ambiguous_question_answer", self.ambiguous_question_answer)
            graph.add_node("out_of_scope_answer", self.out_of_scope_answer)
            graph.add_node("save_message", self.save_message)
            graph.add_node("llm_reply", self.llm_reply)
            graph.add_node("sentiment_detection", self.sentiment_detection)

            graph.set_entry_point("intent_detector")

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

            graph.add_edge("sentiment_detection", "save_message")
            graph.add_edge("save_message", "llm_reply")

            # Desde ambiguous_question_answer y out_of_scope_answer vuelven a intent_detector para bucle # noqa: E501
            graph.add_edge("ambiguous_question_answer", END)
            graph.add_edge("out_of_scope_answer", END)

            # Desde save_message ir a llm_reply
            graph.add_edge("save_message", "llm_reply")

            # Desde llm_reply vuelve a save_message para guardar respuesta
            graph.add_edge("llm_reply", "save_message")

            # Desde save_message finaliza el flujo
            graph.add_edge("save_message", END)

            # Compilar el grafo
            graph = graph.compile(checkpointer=checkpointer, store=store)

            # Guardar visualización del graph
            with open("conversation_graph.png", "wb") as f:
                f.write(graph.get_graph().draw_png())

            # Simular una conversación
            user_id = "user-123"
            thread_id = "thread-1"

            state = MyStateSchema(messages=[])
            config = {
                "configurable": {
                    "user_id": user_id,
                    "thread_id": thread_id,
                    "current_node": "llm_reply",  # esto puedes actualizar dinámicamente
                }
            }

            questions = [
                "What Learnivo does?",
                "What is the name of the company?",
                "What does it?",
            ]

            for question in questions:
                state.messages.append({"role": "user", "content": question})

                # Ejecutar el grafo para cada input
                config["configurable"]["current_node"] = "llm_reply"
                graph.invoke(state, config=config)
                pass
