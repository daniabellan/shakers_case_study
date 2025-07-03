import os
import uuid

import requests
import streamlit as st

# Backend API base URL loaded from environment variable
BACKEND_URL = os.getenv("BACKEND_URL")


def load_history(user_id: str, thread_id: str = "thread-1") -> list:
    """
    Fetch chat history for a given user and thread from the backend.

    Args:
        user_id (str): Identifier for the user.
        thread_id (str, optional): Identifier for the chat thread. Defaults to "thread-1".

    Returns:
        list: List of message dictionaries representing the chat history.
              Returns an empty list if loading fails or no history is found.
    """
    try:
        response = requests.get(
            f"{BACKEND_URL}/rag/history",
            params={"user_id": user_id, "thread_id": thread_id},
        )
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success":
            return data.get("history", [])
        else:
            return []
    except Exception as e:
        st.error(f"Error loading history: {e}")
        return []


def show():
    """
    Render the Streamlit chatbot interface, manage user session state,
    handle input, display conversation history, and send questions to the backend.
    """
    st.title("Chatbot")

    # Initialize session state with default values if not present
    if "user_id" not in st.session_state:
        st.session_state.user_id = "user-42"
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())[:8]
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_loaded_user" not in st.session_state:
        st.session_state.last_loaded_user = ""
    if "last_loaded_thread" not in st.session_state:
        st.session_state.last_loaded_thread = ""

    # Controlled inputs bound to session state
    user_id = st.text_input("User ID", value=st.session_state.user_id)
    thread_id = st.text_input("Chat ID", value=st.session_state.thread_id)

    # Reload chat history if user_id or thread_id has changed
    if (
        user_id != st.session_state.last_loaded_user
        or thread_id != st.session_state.last_loaded_thread
    ):
        st.session_state.user_id = user_id
        st.session_state.thread_id = thread_id
        st.session_state.chat_history = load_history(user_id, thread_id)
        st.session_state.last_loaded_user = user_id
        st.session_state.last_loaded_thread = thread_id

    # Display the chat history messages
    for message in st.session_state.chat_history:
        role = message.get("role", "user")
        content = message.get("content", "")
        if role == "user":
            st.markdown(f"**You:** {content}")
        else:
            st.markdown(f"**Assistant:** {content}")
        st.markdown("---")  # Separator between messages

    question = st.text_area("Your question:")

    # Handle the "Ask" button click event
    if st.button("Ask"):
        if not question.strip():
            st.warning("Please enter a question")
            return

        payload = {
            "question": question,
            "user_id": user_id,
            "thread_id": thread_id,
        }

        try:
            response = requests.post(f"{BACKEND_URL}/rag", json=payload)
            response.raise_for_status()
            data = response.json()
            answer = data.get("assistant_answer", "No answer")

            # Update the local chat history with user question and assistant answer
            st.session_state.chat_history.append({"role": "user", "content": question})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

            # Rerun the app to refresh the UI with updated chat history
            st.rerun()
        except Exception as e:
            st.error(f"Error contacting backend: {e}")
