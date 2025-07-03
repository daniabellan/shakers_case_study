import time

from langchain.schema import HumanMessage
from langchain_core.runnables.config import RunnableConfig

from shakers_case_study.rag.pipelines.rag.enums.rag_schema import MyStateSchema
from shakers_case_study.rag.pipelines.rag.nodes.log_llm_metrics import log_llm_metrics


def sentiment_detection(state: MyStateSchema, config: RunnableConfig):
    start_time = time.time()
    llm = config["configurable"]["llm"]
    user_message = state.messages[-1]["content"]

    # Simple prompt-based sentiment detection
    sentiment_prompt = f'Detect the sentiment of this message: "{user_message}". Return one word: Positive, Negative, or Neutral.'  # noqa: E501

    messages = [HumanMessage(content=sentiment_prompt)]
    response = llm.invoke(messages)
    state = log_llm_metrics(state, "sentiment_detection", start_time, response)

    sentiment = response.content.strip().lower()
    state.sentiment = sentiment

    return state
