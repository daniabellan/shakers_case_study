import time
from typing import Any, Dict, Optional, Union

from langchain_core.messages.ai import AIMessage

from shakers_case_study.rag.pipelines.rag.enums.rag_schema import MyStateSchema
from shakers_case_study.utils.logging import get_logger

logger = get_logger()


def log_llm_metrics(
    state: MyStateSchema,
    operation_name: str,
    start_time: float,
    response: Optional[Union[Dict[str, Any], AIMessage]] = None,
) -> MyStateSchema:
    """
    Logs latency and token usage metrics for a given LLM operation and updates the pipeline state.

    Args:
        state (MyStateSchema): The current state object containing metrics.
        operation_name (str): Name of the LLM operation being logged (e.g., 'intent_detector').
        start_time (float): Timestamp when the operation started (time.time()).
        response (Optional[Union[Dict, AIMessage]]): The LLM response object or dictionary
        containing usage metadata.

    Returns:
        MyStateSchema: The updated state with accumulated metrics.
    """
    latency = time.time() - start_time

    # Extract usage metadata from response if available
    if response is None:
        usage = {}
    elif isinstance(response, dict):
        usage = response.get("usage_metadata", {})
    else:
        # Assume AIMessage or similar object with usage_metadata attribute
        usage = getattr(response, "usage_metadata", {})

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", input_tokens + output_tokens)

    # Log metrics details
    logger.info(f"[LLM METRICS] Operation: {operation_name}")
    logger.info(f"  Latency: {latency:.2f}s")
    logger.info(f"  Input Tokens: {input_tokens}")
    logger.info(f"  Output Tokens: {output_tokens}")
    logger.info(f"  Total Tokens: {total_tokens}")

    # Update cumulative metrics in state
    state.metrics["total_input_tokens"] = state.metrics.get("total_input_tokens", 0) + input_tokens
    state.metrics["total_output_tokens"] = (
        state.metrics.get("total_output_tokens", 0) + output_tokens
    )
    state.metrics["total_tokens"] = state.metrics.get("total_tokens", 0) + total_tokens
    state.metrics["llm_latency"] = state.metrics.get("llm_latency", 0.0) + latency

    return state


def log_final_metrics(state: MyStateSchema) -> MyStateSchema:
    """
    Placeholder function to finalize metrics logging or processing.

    Args:
        state (MyStateSchema): The current pipeline state.

    Returns:
        MyStateSchema: The unmodified state (currently).
    """
    return state
