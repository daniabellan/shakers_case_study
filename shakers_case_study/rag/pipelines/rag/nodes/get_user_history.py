import psycopg
from langchain_core.runnables.config import RunnableConfig


def get_user_history(config: RunnableConfig):
    user_id = config["configurable"]["user_id"]
    thread_id = config["configurable"]["thread_id"]
    db_uri = config["configurable"]["db_uri"]

    with psycopg.connect(db_uri) as conn:
        prefix = f"{user_id}.{thread_id}"
        query = """
            SELECT value
            FROM store
            WHERE prefix = %s
            ORDER BY value->>'timestamp' ASC;
        """
        with conn.cursor() as cur:
            cur.execute(query, (prefix,))
            rows = cur.fetchall()

        user_history = [row[0] for row in rows]

    return user_history


def get_recent_user_messages(user_history, max_messages=10):
    return [m.get("content") for m in user_history if m.get("role") == "user" and m.get("content")][
        -max_messages:
    ]
