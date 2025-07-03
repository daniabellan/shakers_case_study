import streamlit as st

from shakers_case_study.app.tabs import chatbot, documents

# Configure Streamlit app page settings: title and layout style
st.set_page_config(page_title="Shakers Case Study", layout="wide")


def main():
    """
    Main entry point for the Streamlit app.

    Displays a sidebar navigation menu to switch between different app tabs.
    Currently supports:
    - Chatbot interface
    - Documents viewer
    """
    st.sidebar.title("Navigation")

    # Sidebar radio buttons for tab selection
    page = st.sidebar.radio("Go to:", ["Chatbot", "Documents"])

    # Render the selected tab content
    if page == "Chatbot":
        chatbot.show()
    elif page == "Documents":
        documents.show()


if __name__ == "__main__":
    main()
