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


st.set_page_config(
    page_title="Asystent firmowy RAG",
    page_icon="📘",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1000px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .subtitle {
            color: #94a3b8;
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }

        .answer-box {
            background: rgba(30, 41, 59, 0.55);
            border: 1px solid rgba(59, 130, 246, 0.25);
            border-left: 4px solid #3b82f6;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-top: 0.75rem;
            line-height: 1.7;
            color: #f8fafc;
        }

        .source-box {
            background: rgba(15, 23, 42, 0.45);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.8rem;
        }

        .source-title {
            font-weight: 600;
            margin-bottom: 0.4rem;
            color: #e2e8f0;
        }

        .source-text {
            color: #cbd5e1;
            font-size: 0.95rem;
            line-height: 1.6;
        }

        .stButton > button {
            border-radius: 10px;
            padding: 0.55rem 1rem;
            font-weight: 600;
        }

        .stTextInput > div > div > input {
            border-radius: 10px;
        }

        .stFileUploader {
            padding-top: 0.3rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">📘 Asystent firmowy RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Wgraj dokumenty, zbuduj indeks i zadawaj pytania na podstawie firmowej wiedzy.</div>',
    unsafe_allow_html=True,
)

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
    question = st.text_input(
        "Wpisz pytanie dotyczące dokumentów",
        placeholder="Np. Jakie usługi oferuje firma?"
    )

    if st.button("Zapytaj"):
        if not question.strip():
            st.warning("Wpisz pytanie.")
        else:
            with st.spinner("Szukam odpowiedzi..."):
                result = ask_rag(question)

            st.markdown("### Odpowiedź")
            st.markdown(
                f"<div class='answer-box'>{result['answer']}</div>",
                unsafe_allow_html=True,
            )

            st.markdown("### Źródła")
            for i, doc in enumerate(result["sources"], start=1):
                st.markdown(
                    f"""
                    <div class="source-box">
                        <div class="source-title">Źródło {i}: {doc.metadata.get('source', 'unknown')}</div>
                        <div class="source-text">{doc.page_content[:1000]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
