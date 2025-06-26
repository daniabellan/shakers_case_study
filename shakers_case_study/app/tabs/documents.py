import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL")


def show():
    st.title("Documentación Técnica")

    try:
        response = requests.get(f"{BACKEND_URL}/documents")
        response.raise_for_status()
        data = response.json()
        if data["status"] != "success":
            st.error(f"Error del servidor: {data.get('message', 'Desconocido')}")
            return
        files = sorted(data.get("payload", []))
    except Exception as e:
        st.error(f"Error al obtener lista de documentos: {e}")
        return

    if not files:
        st.info("No hay documentos aún.")
        return

    selected_doc = st.sidebar.selectbox("Selecciona un documento", files)

    try:
        doc_resp = requests.get(f"{BACKEND_URL}/uploaded_docs/{selected_doc}")
        doc_resp.raise_for_status()
        content = doc_resp.text
    except Exception as e:
        st.error(f"Error al obtener contenido del documento: {e}")
        return

    st.subheader(f"{selected_doc}")
    st.markdown("---")
    st.markdown(content, unsafe_allow_html=True)

    st.markdown(f"[Ver en navegador]({BACKEND_URL}/uploaded_docs/{selected_doc})")
