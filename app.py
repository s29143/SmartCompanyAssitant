from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.loaders import load_text_documents
from src.chunking import split_documents
from src.vector_store import build_vector_store
from src.rag_chain import ask_rag

load_dotenv()

DATA_DIR = Path("data/documents")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_files(uploaded_files):
    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = DATA_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path.name)
    return saved_files


def ingest_documents():
    docs = load_text_documents(str(DATA_DIR))
    if not docs:
        return 0, 0

    chunks = split_documents(docs)
    build_vector_store(chunks)
    return len(docs), len(chunks)


st.set_page_config(page_title="Asystent firmowy RAG", layout="wide")

st.title("Asystent firmowy RAG")
st.write("Wgraj dokumenty, zbuduj indeks i zadawaj pytania na podstawie firmowej wiedzy.")

tab1, tab2 = st.tabs(["Indeksowanie dokumentów", "Pytania do dokumentów"])

with tab1:
    st.subheader("Dodaj dokumenty")
    uploaded_files = st.file_uploader(
        "Wgraj pliki TXT",
        type=["txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        saved_files = save_uploaded_files(uploaded_files)
        st.success(f"Zapisano pliki: {', '.join(saved_files)}")

    if st.button("Zbuduj indeks"):
        with st.spinner("Przetwarzanie dokumentów i budowanie bazy..."):
            docs_count, chunks_count = ingest_documents()

        if docs_count == 0:
            st.warning("Brak dokumentów do przetworzenia.")
        else:
            st.success(
                f"Gotowe. Wczytano {docs_count} dokumentów i utworzono {chunks_count} chunków."
            )

with tab2:
    st.subheader("Zadaj pytanie")
    question = st.text_input("Wpisz pytanie dotyczące dokumentów")

    if st.button("Zapytaj"):
        if not question.strip():
            st.warning("Wpisz pytanie.")
        else:
            with st.spinner("Szukam odpowiedzi..."):
                result = ask_rag(question)

            st.markdown("### Odpowiedź")
            st.write(result["answer"])

            with st.expander("Pokaż źródła"):
                for i, doc in enumerate(result["sources"], start=1):
                    st.markdown(f"**Źródło {i}:** {doc.metadata.get('source', 'unknown')}")
                    st.write(doc.page_content[:1000])
                    st.markdown("---")