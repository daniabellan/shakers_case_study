import time

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.runnables.config import RunnableConfig

from shakers_case_study.rag.pipelines.rag.enums.rag_schema import MyStateSchema
from shakers_case_study.rag.pipelines.rag.nodes.log_llm_metrics import log_llm_metrics
from shakers_case_study.rag.pipelines.rag.prompts.prompts import (
    MALICIOUS_DETECTOR_PROMPT,
    UNSAFE_FALLBACK_PROMPT,
)


def malicious_query_detector(state: MyStateSchema, config: RunnableConfig) -> MyStateSchema:
    """
    Detects if the user question is potentially malicious using an LLM prompt.

    Args:
        state (MyStateSchema): Current pipeline state containing conversation messages.
        config (RunnableConfig): Configuration including LLM and other resources.

    Returns:
        MyStateSchema: Updated state with moderation result set.
    """
    start_time = time.time()
    user_question = state.messages[-1]["content"]
    llm = config["configurable"]["llm"]

    # Prepare the prompt to detect malicious queries
    prompt = PromptTemplate(
        input_variables=["user_question"],
        template=MALICIOUS_DETECTOR_PROMPT,
    )
    formatted_prompt = prompt.format(user_question=user_question)

    messages = [HumanMessage(content=formatted_prompt)]
    response = llm.invoke(messages)

    # Log LLM usage metrics for this operation
    state = log_llm_metrics(state, "malicious_query_detector", start_time, response)

    # Store the moderation result in the state
    state.moderation = response.content

    print(f"malicious_query_detector time: {time.time() - start_time:.2f} secs")
    return state


def unsafe_fallback(state: MyStateSchema, config: RunnableConfig) -> MyStateSchema:
    """
    Provides a fallback response when a user query is detected as unsafe.

    Args:
        state (MyStateSchema): Current pipeline state containing conversation messages.
        config (RunnableConfig): Configuration including LLM and other resources.

    Returns:
        MyStateSchema: Updated state with fallback response appended and current_node set.
    """
    start_time = time.time()
    user_question = state.messages[-1]["content"]
    llm = config["configurable"]["llm"]

    # Prepare the fallback prompt for unsafe queries
    prompt = PromptTemplate(
        input_variables=["user_question"],
        template=UNSAFE_FALLBACK_PROMPT,
    )
    formatted_prompt = prompt.format(user_question=user_question)

    messages = [HumanMessage(content=formatted_prompt)]
    response = llm.invoke(messages)

    # Log LLM usage metrics for this fallback operation
    state = log_llm_metrics(state, "unsafe_fallback", start_time, response)

    # Update the state to indicate fallback and append the assistant’s response
    state.current_node = "unsafe_fallback"
    state.messages.append({"role": "assistant", "content": response.content})

    return state
