import time
from typing import Optional, Union

from langchain_core.messages.ai import AIMessage

from shakers_case_study.rag.pipelines.rag.enums.rag_schema import MyStateSchema
from shakers_case_study.utils.logging import get_logger

logger = get_logger()


def log_llm_metrics(
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

    state.metrics["total_input_tokens"] = state.metrics.get("total_input_tokens", 0) + input_tokens
    state.metrics["total_output_tokens"] = (
        state.metrics.get("total_output_tokens", 0) + output_tokens
    )
    state.metrics["total_tokens"] = state.metrics.get("total_tokens", 0) + total_tokens
    state.metrics["llm_latency"] = state.metrics.get("llm_latency", 0.0) + latency

    return state


def log_final_metrics(state: MyStateSchema):
    return state
