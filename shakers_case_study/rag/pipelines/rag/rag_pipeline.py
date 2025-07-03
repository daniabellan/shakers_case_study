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
    """
    Retrieval-Augmented Generation (RAG) pipeline that manages conversation flow using
    a state graph and integrates vectorstore retrieval, LLM calls, and database checkpoints.

    Attributes:
        vectorstore (QdrantIndex): Vector database for document similarity search.
        llm (BaseChatModel): Language model for generating responses.
        db_uri (str): Database connection URI for storing conversation state and checkpoints.
        embedder (Embeddings): Embedding model for vectorizing queries.
        exit_stack (ExitStack): Manages lifecycle of context-managed resources.
        store (PostgresStore): Postgres-based storage interface for conversation state.
        checkpointer (PostgresSaver): Manages checkpoint saving in Postgres.
        graph (StateGraph): Directed graph defining the pipeline's processing nodes and flow.
    """

    def __init__(
        self, vectorstore: QdrantIndex, llm: BaseChatModel, db_uri: str, embedder: Embeddings
    ):
        self.exit_stack = ExitStack()
        self.vectorstore = vectorstore
        self.llm = llm
        self.db_uri = db_uri
        self.embedder = embedder

        # Initialize database tables and checkpoint memory
        self._initialize_checkpoint_memory()

        # Open persistent store and checkpointer contexts
        self.store = self.exit_stack.enter_context(PostgresStore.from_conn_string(self.db_uri))
        self.checkpointer = self.exit_stack.enter_context(
            PostgresSaver.from_conn_string(self.db_uri)
        )

        # Build the state graph with all nodes and routing logic
        self.graph = self._build_graph()

    def _initialize_checkpoint_memory(self):
        """
        Sets up the necessary database tables for storing conversation state and checkpoints.
        This method ensures tables exist before the pipeline runs.
        """
        with (
            PostgresStore.from_conn_string(self.db_uri) as store,
            PostgresSaver.from_conn_string(self.db_uri) as checkpointer,
        ):
            store.setup()
            checkpointer.setup()

    def _build_graph(self) -> StateGraph:
        """
        Constructs the conversation state graph defining nodes and their routing logic.

        Returns:
            StateGraph: Compiled graph ready for invocation.
        """
        graph = StateGraph(state_schema=MyStateSchema)

        # Add all nodes representing pipeline steps
        graph.add_node("malicious_query_detector", malicious_query_detector)
        graph.add_node("intent_detector", intent_detector)
        graph.add_node("ambiguous_question_answer", ambiguous_question_answer)
        graph.add_node("out_of_scope_answer", out_of_scope_answer)
        graph.add_node("save_message", save_message)
        graph.add_node("qa_reply", qa_reply)
        graph.add_node("sentiment_detection", sentiment_detection)
        graph.add_node("unsafe_fallback", unsafe_fallback)
        graph.add_node("log_final_metrics", log_final_metrics)

        # Define entry point of the graph
        graph.set_entry_point("malicious_query_detector")

        # Routing function after malicious query detection
        def route_from_malicious(state: MyStateSchema) -> str:
            moderation = (state.moderation or "").lower()
            if any(keyword in moderation for keyword in ["unsafe", "malicious", "flagged"]):
                return "unsafe_fallback"
            return "intent_detector"

        # Routing function after intent detection node
        def route_from_intent(state: MyStateSchema) -> str:
            intent = getattr(state, "intent", "").lower()
            if intent == "direct":
                return "direct"
            elif intent == "ambiguous":
                return "ambiguous"
            elif intent == "out_of_scope":
                return "out_of_scope"
            else:
                # Default to intent_detector if intent is unrecognized
                return "intent_detector"

        # Conditional edges based on malicious detection result
        graph.add_conditional_edges(
            "malicious_query_detector",
            route_from_malicious,
            {
                "unsafe_fallback": "unsafe_fallback",
                "intent_detector": "intent_detector",
            },
        )

        # Conditional edges based on detected intent
        graph.add_conditional_edges(
            "intent_detector",
            route_from_intent,
            {
                "direct": "sentiment_detection",
                "ambiguous": "ambiguous_question_answer",
                "out_of_scope": "out_of_scope_answer",
            },
        )

        # Define direct edges between nodes
        graph.add_edge("sentiment_detection", "qa_reply")
        graph.add_edge("qa_reply", "save_message")
        graph.add_edge("ambiguous_question_answer", "save_message")
        graph.add_edge("out_of_scope_answer", "save_message")
        graph.add_edge("unsafe_fallback", "save_message")

        graph.add_edge("save_message", "log_final_metrics")
        graph.add_edge("log_final_metrics", END)

        # Compile the graph with checkpoint and store
        graph = graph.compile(checkpointer=self.checkpointer, store=self.store)

        # Save graph visualization for debugging and documentation
        with open("conversation_graph.png", "wb") as f:
            f.write(graph.get_graph().draw_png())

        return graph

    @on_exception(expo, RateLimitException, max_tries=3)
    @limits(calls=10, period=60)
    def run(self, state: MyStateSchema, config: dict) -> None:
        return self.graph.invoke(state, config=config)

    def profile_population(self) -> None:
        # FUNCION PARA RELLENAR LA BASE DE DATOS CON PERFILES DE PRUEBA
        profiles_path = "shakers_case_study/rag/pipelines/rag/profiles.json"
        with open(profiles_path, "r", encoding="utf-8") as f:
            profiles = json.load(f)

        total_time_all = 0
        total_questions_all = 0

        for profile in profiles:
            # 'mixed' questions for this profile
            questions = profiles["mixed"]

            total_time_profile = 0
            total_questions_profile = len(questions)

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

                logger.info(f"[{profile}] Graph execution time for question: {duration:.3f} secs")

            avg_time_profile = (
                total_time_profile / total_questions_profile if total_questions_profile else 0
            )
            logger.info(
                f"Average execution time for profile '{profile}': {avg_time_profile:.3f} secs\n"
            )

            total_time_all += total_time_profile
            total_questions_all += total_questions_profile

        avg_time_all = total_time_all / total_questions_all if total_questions_all else 0
        logger.info(f"Overall average execution time per question: {avg_time_all:.3f} secs")
