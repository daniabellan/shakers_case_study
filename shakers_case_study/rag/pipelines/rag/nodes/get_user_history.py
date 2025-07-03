import psycopg
from langchain_core.runnables.config import RunnableConfig


def get_user_history(config: RunnableConfig) -> list:
    """
    Retrieve the full user message history from the database for a given user and thread.

    Args:
        config (RunnableConfig): Configuration containing user_id, thread_id, and database URI.

    Returns:
        list: A list of message records (dicts) sorted by timestamp ascending.
    """
    user_id = config["configurable"]["user_id"]
    thread_id = config["configurable"]["thread_id"]
    db_uri = config["configurable"]["db_uri"]

    prefix = f"{user_id}.{thread_id}"
    query = """
        SELECT value
        FROM store
        WHERE prefix = %s
        ORDER BY value->>'timestamp' ASC;
    """

    with psycopg.connect(db_uri) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (prefix,))
            rows = cur.fetchall()

    # Extract messages from query result
    user_history = [row[0] for row in rows]

    return user_history


def get_recent_user_messages(user_history: list, max_messages: int = 10) -> list:
    """
    Extracts the most recent user messages from the full user history.

    Args:
        user_history (list): List of message dicts from the user.
        max_messages (int, optional): Maximum number of recent user messages to retrieve.
        Defaults to 10.

    Returns:
        list: List of content strings of the most recent user messages.
    """
    # Filter user messages and extract 'content', then return the last max_messages entries
    recent_messages = [
        message.get("content")
        for message in user_history
        if message.get("role") == "user" and message.get("content")
    ][-max_messages:]

    return recent_messages
