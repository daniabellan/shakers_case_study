import time

from langchain.schema import HumanMessage
from langchain_core.runnables.config import RunnableConfig

from shakers_case_study.rag.pipelines.rag.enums.rag_schema import MyStateSchema
from shakers_case_study.rag.pipelines.rag.nodes.log_llm_metrics import log_llm_metrics


def sentiment_detection(state: MyStateSchema, config: RunnableConfig) -> MyStateSchema:
    """
    Detects the sentiment of the latest user message using an LLM prompt.

    The function sends the user's most recent message to the language model with a
    simple prompt asking to classify sentiment as one of: Positive, Negative, or Neutral.
    The detected sentiment is then stored in the state.

    Args:
        state (MyStateSchema): Current conversation state containing messages and metadata.
        config (RunnableConfig): Configuration object that includes the LLM to use.

    Returns:
        MyStateSchema: Updated state with the detected sentiment stored in `state.sentiment`.
    """
    start_time = time.time()
    llm = config["configurable"]["llm"]
    user_message = state.messages[-1]["content"]

    # Construct prompt for sentiment classification
    sentiment_prompt = (
        f'Detect the sentiment of this message: "{user_message}". '
        "Return one word: Positive, Negative, or Neutral."
    )

    messages = [HumanMessage(content=sentiment_prompt)]
    response = llm.invoke(messages)

    # Log latency and token usage metrics for this operation
    state = log_llm_metrics(state, "sentiment_detection", start_time, response)

    # Normalize and store the sentiment label in state
    sentiment = response.content.strip().lower()
    state.sentiment = sentiment

    return state
