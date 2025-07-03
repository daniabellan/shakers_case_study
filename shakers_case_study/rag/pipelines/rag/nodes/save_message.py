import uuid
from datetime import datetime, timezone

from langchain_core.runnables.config import RunnableConfig

from shakers_case_study.rag.pipelines.rag.enums.rag_schema import MyStateSchema


def save_message(state: MyStateSchema, config: RunnableConfig, *, store):
    user_id = config["configurable"]["user_id"]
    thread_id = config["configurable"]["thread_id"]

    # Si el último mensaje es assistant y el penúltimo user, guardar ambos
    if (
        len(state.messages) >= 2
        and state.messages[-1]["role"] == "assistant"
        and state.messages[-2]["role"] == "user"
    ):
        messages_to_save = [state.messages[-2], state.messages[-1]]
    else:
        messages_to_save = [state.messages[-1]]

    for msg in messages_to_save:
        if msg["role"] == "assistant":
            current_node = state.current_node
        else:
            current_node = ""

        message_data = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "thread_id": thread_id,
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node_name": current_node,
            "is_flagged": False,
            "metadata": {},
            "sentiment": state.sentiment,
        }
        store.put((user_id, thread_id), message_data["id"], message_data)

    return state
