import os

import requests
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from a .env file (e.g., BACKEND_URL)
load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")


def show():
    """
    Streamlit app function to display and browse technical documentation files.

    Workflow:
    1. Retrieve the list of available documents from the backend API.
    2. Present the document list in a sidebar dropdown for user selection.
    3. Fetch and display the content of the selected document.
    4. Provide a clickable link to open the document directly in the browser.

    Handles exceptions gracefully, showing error messages on failure.
    """

    st.title("Technical Documentation")

    # Step 1: Retrieve document list from backend
    try:
        response = requests.get(f"{BACKEND_URL}/documents")
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "success":
            st.error(f"Server error: {data.get('message', 'Unknown error')}")
            return

        files = sorted(data.get("payload", []))

    except Exception as e:
        st.error(f"Error fetching document list: {e}")
        return

    # Inform the user if no documents are available
    if not files:
        st.info("No documents available yet.")
        return

    # Step 2: Display document selection dropdown in the sidebar
    selected_doc = st.sidebar.selectbox("Select a document", files)

    # Step 3: Fetch and display the content of the selected document
    try:
        doc_response = requests.get(f"{BACKEND_URL}/uploaded_docs/{selected_doc}")
        doc_response.raise_for_status()
        content = doc_response.text
    except Exception as e:
        st.error(f"Error fetching document content: {e}")
        return

    # Step 4: Render the selected document's title and content in the main area
    st.subheader(selected_doc)
    st.markdown("---")
    st.markdown(content, unsafe_allow_html=True)

    # Provide a direct link to view the document in a browser tab
    st.markdown(f"[View in browser]({BACKEND_URL}/uploaded_docs/{selected_doc})")
