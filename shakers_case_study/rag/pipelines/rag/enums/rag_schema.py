from typing import Dict, List, Optional

from pydantic import BaseModel


class MyStateSchema(BaseModel):
    messages: List[Dict]
    intent: Optional[str] = None
    sentiment: Optional[str] = None
    moderation: Optional[str] = None
    current_node: Optional[str] = None
    metrics: Dict[str, float] = {}
