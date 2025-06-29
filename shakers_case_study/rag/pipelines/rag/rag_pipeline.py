from langchain_core.language_models.chat_models import BaseChatModel

from shakers_case_study.rag.vectorstore.qdrant_index import QdrantIndex
from shakers_case_study.utils.logging import get_logger

logger = get_logger()


class RAGPipeline:
    def __init__(self, vectorstore: QdrantIndex, llm: BaseChatModel):
        self.vectorstore = vectorstore
        self.llm = llm

    def run(self) -> None:
        # documents = self.vectorstore.similarity_search("HELLO", 10)
        pass
