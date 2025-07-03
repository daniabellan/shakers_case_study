import time

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_core.runnables.config import RunnableConfig

from shakers_case_study.rag.pipelines.rag.enums.rag_schema import MyStateSchema
from shakers_case_study.rag.pipelines.rag.nodes.log_llm_metrics import log_llm_metrics
from shakers_case_study.rag.pipelines.rag.prompts.prompts import (
    AMBIGUOUS_QUESTION_PROMPT,
    INTENT_PROMPT,
    OUT_OF_SCOPE_PROMPT,
)
from shakers_case_study.utils.logging import get_logger

logger = get_logger()


def intent_detector(state: MyStateSchema, config: RunnableConfig) -> MyStateSchema:
    """
    Detects the user's intent based on the latest user message.

    Args:
        state (MyStateSchema): Current pipeline state including messages.
        config (RunnableConfig): Pipeline configuration containing the LLM instance.

    Returns:
        MyStateSchema: Updated state with detected intent and logged metrics.
    """
    llm = config["configurable"]["llm"]
    start_time = time.time()

    user_question = state.messages[-1]["content"]

    prompt = PromptTemplate(
        input_variables=["user_question"],
        template=INTENT_PROMPT,
    )
    formatted_prompt = prompt.format(user_question=user_question)
    messages = [HumanMessage(content=formatted_prompt)]

    response = llm.invoke(messages)

    state = log_llm_metrics(
        state, operation_name="intent_detector", start_time=start_time, response=response
    )
    state.intent = response.content

    logger.info(f"intent_detector completed in {time.time() - start_time:.2f} seconds")
    return state


def out_of_scope_answer(state: MyStateSchema, config: RunnableConfig) -> MyStateSchema:
    """
    Generates a response for user questions that are out of scope.

    Args:
        state (MyStateSchema): Current pipeline state including messages.
        config (RunnableConfig): Pipeline configuration containing the LLM instance.

    Returns:
        MyStateSchema: Updated state with out-of-scope response appended and logged metrics.
    """
    llm = config["configurable"]["llm"]
    start_time = time.time()

    user_question = state.messages[-1]["content"]

    prompt = PromptTemplate(
        input_variables=["user_question"],
        template=OUT_OF_SCOPE_PROMPT,
    )
    formatted_prompt = prompt.format(user_question=user_question)
    messages = [HumanMessage(content=formatted_prompt)]

    response = llm.invoke(messages)

    state = log_llm_metrics(
        state, operation_name="out_of_scope_answer", start_time=start_time, response=response
    )
    state.current_node = "out_of_scope"
    state.messages.append({"role": "assistant", "content": response.content})

    return state


def ambiguous_question_answer(state: MyStateSchema, config: RunnableConfig) -> MyStateSchema:
    """
    Generates a response for ambiguous user questions.

    Args:
        state (MyStateSchema): Current pipeline state including messages.
        config (RunnableConfig): Pipeline configuration containing the LLM instance.

    Returns:
        MyStateSchema: Updated state with ambiguous question response appended and logged metrics.
    """
    llm = config["configurable"]["llm"]
    start_time = time.time()

    user_question = state.messages[-1]["content"]

    prompt = PromptTemplate(
        input_variables=["user_question"],
        template=AMBIGUOUS_QUESTION_PROMPT,
    )
    formatted_prompt = prompt.format(user_question=user_question)
    messages = [HumanMessage(content=formatted_prompt)]

    response = llm.invoke(messages)

    state = log_llm_metrics(
        state, operation_name="ambiguous_question_answer", start_time=start_time, response=response
    )
    state.current_node = "ambiguous_question"
    state.messages.append({"role": "assistant", "content": response.content})

    return state
