import streamlit as st
from shakers_case_study.app.tabs import documents

# Configure the Streamlit app page settings
st.set_page_config(page_title="Shakers Case Study", layout="wide")

def main():
    """
    Main function to run the Streamlit app.

    Displays a sidebar with navigation options.
    Currently supports navigation to the "Documents" tab.
    """
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to:", ["Documents"])

    if page == "Documents":
        documents.show()

if __name__ == "__main__":
    main()
