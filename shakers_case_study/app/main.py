import streamlit as st

from shakers_case_study.app.tabs import documents

st.set_page_config(page_title="Shakers Case Study", layout="wide")


def main():
    st.sidebar.title(" Navegación")
    page = st.sidebar.radio("Ir a:", [" Documentos"])

    if page == " Documentos":
        documents.show()


if __name__ == "__main__":
    main()
