import json
import time
from contextlib import ExitStack

from backoff import expo, on_exception
from langchain.embeddings.base import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, StateGraph
from langgraph.store.postgres import PostgresStore
from ratelimit import RateLimitException, limits

from shakers_case_study.rag.pipelines.rag.enums.rag_schema import MyStateSchema
from shakers_case_study.rag.pipelines.rag.nodes.intent_detector import (
    ambiguous_question_answer,
    intent_detector,
    out_of_scope_answer,
)
from shakers_case_study.rag.pipelines.rag.nodes.log_llm_metrics import log_final_metrics
from shakers_case_study.rag.pipelines.rag.nodes.malicious_query_detector import (
    malicious_query_detector,
    unsafe_fallback,
)
from shakers_case_study.rag.pipelines.rag.nodes.qa_reply import qa_reply
from shakers_case_study.rag.pipelines.rag.nodes.save_message import save_message
from shakers_case_study.rag.pipelines.rag.nodes.sentiment_detection import sentiment_detection
from shakers_case_study.rag.vectorstore.qdrant_index import QdrantIndex
from shakers_case_study.utils.logging import get_logger

logger = get_logger()


class RAGPipeline:
    def __init__(
        self, vectorstore: QdrantIndex, llm: BaseChatModel, db_uri: str, embedder: Embeddings
    ):
        self.exit_stack = ExitStack()
        self.vectorstore = vectorstore
        self.llm = llm
        self.db_uri = db_uri
        self.embedder = embedder

        # Set up checkpoint and user history in BBDD
        self._initialize_checkpoint_memory()

        self.store = self.exit_stack.enter_context(PostgresStore.from_conn_string(self.db_uri))
        self.checkpointer = self.exit_stack.enter_context(
            PostgresSaver.from_conn_string(self.db_uri)
        )

        self.graph = self._build_graph()

    def _initialize_checkpoint_memory(self):
        with (
            PostgresStore.from_conn_string(self.db_uri) as store,
            PostgresSaver.from_conn_string(self.db_uri) as checkpointer,
        ):
            store.setup()
            checkpointer.setup()

    def _build_graph(
        self,
    ):
        graph = StateGraph(state_schema=MyStateSchema)

        # Añadir todos los nodos
        graph.add_node("malicious_query_detector", malicious_query_detector)
        graph.add_node("intent_detector", intent_detector)
        graph.add_node("ambiguous_question_answer", ambiguous_question_answer)
        graph.add_node("out_of_scope_answer", out_of_scope_answer)
        graph.add_node("save_message", save_message)
        graph.add_node("qa_reply", qa_reply)
        graph.add_node("sentiment_detection", sentiment_detection)
        graph.add_node("unsafe_fallback", unsafe_fallback)
        graph.add_node("log_final_metrics", log_final_metrics)

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

        graph.add_edge("sentiment_detection", "qa_reply")
        graph.add_edge("qa_reply", "save_message")
        graph.add_edge("ambiguous_question_answer", "save_message")
        graph.add_edge("out_of_scope_answer", "save_message")
        graph.add_edge("unsafe_fallback", "save_message")

        graph.add_edge("save_message", "log_final_metrics")
        graph.add_edge("log_final_metrics", END)

        # Compilar el grafo
        graph = graph.compile(checkpointer=self.checkpointer, store=self.store)

        # Guardar visualización del graph
        with open("conversation_graph.png", "wb") as f:
            f.write(graph.get_graph().draw_png())

        return graph

    # 10 requests per minute per user
    @on_exception(expo, RateLimitException, max_tries=3)
    @limits(calls=10, period=60)
    def run(self) -> str:
        profiles_path = "shakers_case_study/rag/pipelines/rag/profiles.json"
        with open(profiles_path, "r", encoding="utf-8") as f:
            profiles = json.load(f)

        total_time_all = 0
        total_questions_all = 0

        for profile in profiles:
            questions = profiles["mixed"]

            total_time_profile = 0
            total_questions_profile = len(questions)

            # Simular una conversación
            user_id = f"user-{profile}"
            thread_id = "thread-1"

            for question in questions:
                state = MyStateSchema(
                    messages=[{"role": "user", "content": question}],
                    current_node="malicious_query_detector",
                )
                config = {
                    "configurable": {
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "llm": self.llm,
                        "db_uri": self.db_uri,
                        "embedder": self.embedder,
                        "vectorstore": self.vectorstore,
                    }
                }

                start_time = time.time()
                state = self.graph.invoke(state, config=config)
                end_time = time.time()

                duration = end_time - start_time
                total_time_profile += duration

                print(f"[{profile}] Graph execution time for question: {duration:.3f} secs")

            avg_time_profile = (
                total_time_profile / total_questions_profile if total_questions_profile else 0
            )
            print(f"Average execution time for profile '{profile}': {avg_time_profile:.3f} secs\n")

            total_time_all += total_time_profile
            total_questions_all += total_questions_profile

        avg_time_all = total_time_all / total_questions_all if total_questions_all else 0
        print(f"Overall average execution time per question: {avg_time_all:.3f} secs")
        pass
