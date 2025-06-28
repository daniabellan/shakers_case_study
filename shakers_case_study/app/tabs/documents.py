import os

import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")


def show():
    """
    Streamlit app function to display a list of technical documentation files
    and show the content of a selected document.

    Workflow:
    - Fetches a list of available documents from the backend.
    - Displays the list in a sidebar selectbox for user selection.
    - Fetches and displays the content of the selected document.
    - Provides a direct link to view the document in the browser.

    Handles errors gracefully with Streamlit messages.
    """

    st.title("Technical documentation")

    # Fetch the list of documents from the backend
    try:
        response = requests.get(f"{BACKEND_URL}/documents")
        response.raise_for_status()
        data = response.json()
        if data["status"] != "success":
            st.error(f"Server error: {data.get('message', 'Unknown')}")
            return
        files = sorted(data.get("payload", []))
    except Exception as e:
        st.error(f"Error fetching document list: {e}")
        return

    # Inform user if no documents are available
    if not files:
        st.info("No documents available yet.")
        return

    # Let user select a document from the sidebar
    selected_doc = st.sidebar.selectbox("Select a document", files)

    # Fetch the content of the selected document
    try:
        doc_resp = requests.get(f"{BACKEND_URL}/uploaded_docs/{selected_doc}")
        doc_resp.raise_for_status()
        content = doc_resp.text
    except Exception as e:
        st.error(f"Error fetching document content: {e}")
        return

    # Display document title and content
    st.subheader(f"{selected_doc}")
    st.markdown("---")
    st.markdown(content, unsafe_allow_html=True)

    # Provide a link to view the document in the browser
    st.markdown(f"[View in browser]({BACKEND_URL}/uploaded_docs/{selected_doc})")
