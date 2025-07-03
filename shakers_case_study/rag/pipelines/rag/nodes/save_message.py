import uuid
from datetime import datetime, timezone

from langchain_core.runnables.config import RunnableConfig

from shakers_case_study.rag.pipelines.rag.enums.rag_schema import MyStateSchema


def save_message(state: MyStateSchema, config: RunnableConfig, *, store) -> MyStateSchema:
    """
    Save the most recent user and assistant messages to the persistent store.

    If the last message is from the assistant and the one before it from the user,
    both messages are saved to maintain context. Otherwise, only the latest message
    is saved.

    Args:
        state (MyStateSchema): The current conversation state containing messages and metadata.
        config (RunnableConfig): Configuration object providing user and thread identifiers.
        store: An abstract storage interface with a `put` method accepting keys and message data.

    Returns:
        MyStateSchema: The unchanged state object for pipeline continuity.
    """
    user_id = config["configurable"]["user_id"]
    thread_id = config["configurable"]["thread_id"]

    # Determine which messages to save: last two if user followed by assistant, else just last
    if (
        len(state.messages) >= 2
        and state.messages[-1]["role"] == "assistant"
        and state.messages[-2]["role"] == "user"
    ):
        messages_to_save = [state.messages[-2], state.messages[-1]]
    else:
        messages_to_save = [state.messages[-1]]

    # Save each message with enriched metadata
    for msg in messages_to_save:
        current_node = state.current_node if msg["role"] == "assistant" else ""

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

        # Store message keyed by (user_id, thread_id) and message id
        store.put((user_id, thread_id), message_data["id"], message_data)

    return state
