from typing import Dict, List, Optional

from pydantic import BaseModel


class MyStateSchema(BaseModel):
    """
    Schema representing the state within the LangGraph pipeline.

    Attributes:
        messages (List[Dict]): List of message dictionaries representing the
        conversation history or inputs.
        intent (Optional[str]): Detected user intent, if any.
        sentiment (Optional[str]): Sentiment analysis result of the input, if available.
        moderation (Optional[str]): Moderation status or flags related to the content.
        current_node (Optional[str]): Identifier of the current node in the
        pipeline execution graph.
        metrics (Dict[str, float]): Dictionary of numerical metrics related
        to the state or processing.
    """

    messages: List[Dict]
    intent: Optional[str] = None
    sentiment: Optional[str] = None
    moderation: Optional[str] = None
    current_node: Optional[str] = None
    metrics: Dict[str, float] = {}
