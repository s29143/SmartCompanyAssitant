from pathlib import Path
import html

import streamlit as st
from dotenv import load_dotenv

from src.loaders import load_text_documents
from src.chunking import split_documents
from src.questions import EXAMPLE_QUESTIONS
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
            white-space: pre-wrap;
        }

        .stButton > button {
            border-radius: 10px;
            padding: 0.55rem 1rem;
            font-weight: 600;
        }

        .stFileUploader {
            padding-top: 0.3rem;
        }
        
        .examples-wrapper {
            margin: 1rem 0 0.75rem 0;
        }
        
        .examples-title {
            font-size: 1.05rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 0.15rem;
        }
        
        .examples-subtitle {
            color: #94a3b8;
            font-size: 0.92rem;
            margin-bottom: 0.8rem;
        }
        
        div[data-testid="column"] .stButton > button {
            border-radius: 999px;
            background: rgba(59, 130, 246, 0.10);
            border: 1px solid rgba(59, 130, 246, 0.28);
            color: #dbeafe;
            font-weight: 500;
            padding: 0.55rem 0.9rem;
        }
        
        div[data-testid="column"] .stButton > button:hover {
            background: rgba(59, 130, 246, 0.18);
            border-color: rgba(96, 165, 250, 0.55);
            color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

st.markdown(
    '<div class="main-title">📘 Asystent firmowy RAG</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">Wgraj dokumenty, zbuduj indeks i rozmawiaj z asystentem na podstawie firmowej wiedzy.</div>',
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["Indeksowanie dokumentów", "Czat z dokumentami"])

with tab1:
    st.subheader("Dodaj dokumenty")

    uploaded_files = st.file_uploader(
        "Wgraj pliki TXT",
        type=["txt"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        saved_files = save_uploaded_files(uploaded_files)
        st.success(f"Zapisano pliki: {', '.join(saved_files)}")

    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Zbuduj indeks"):
            with st.spinner("Przetwarzanie dokumentów i budowanie bazy..."):
                docs_count, chunks_count = ingest_documents()

            if docs_count == 0:
                st.warning("Brak dokumentów do przetworzenia.")
            else:
                st.success(
                    f"Gotowe. Wczytano {docs_count} dokumentów "
                    f"i utworzono {chunks_count} chunków."
                )

    with col2:
        if st.button("Wyczyść rozmowę"):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.success("Rozmowa została wyczyszczona.")

with tab2:
    st.subheader("Czat z asystentem")

    example_questions = EXAMPLE_QUESTIONS

    if "show_examples" not in st.session_state:
        st.session_state.show_examples = True

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    selected_question = None

    if st.session_state.show_examples and not st.session_state.messages:
        st.markdown(
            """
            <div class="examples-wrapper">
                <div class="examples-title">Nie wiesz od czego zacząć?</div>
                <div class="examples-subtitle">Wybierz jedno z przykładowych pytań:</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        cols = st.columns(2)

        for index, question in enumerate(example_questions):
            with cols[index % 2]:
                if st.button(question, key=f"example_question_{index}"):
                    selected_question = question
                    st.session_state.show_examples = False

    user_question = st.chat_input("Zadaj pytanie dotyczące dokumentów...")

    final_question = selected_question or user_question

    if final_question:
        st.session_state.show_examples = False

        st.session_state.messages.append(
            {
                "role": "user",
                "content": final_question,
            }
        )

        with st.spinner("Szukam odpowiedzi w dokumentach..."):
            result = ask_rag(
                question=final_question,
                chat_history=st.session_state.messages[-6:],
            )

        assistant_answer = result["answer"]

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_answer,
            }
        )

        st.session_state.last_sources = result["sources"]

        st.rerun()

    if st.session_state.last_sources:
        with st.expander("Źródła ostatniej odpowiedzi"):
            for i, doc in enumerate(st.session_state.last_sources, start=1):
                source_name = html.escape(doc.metadata.get("source", "unknown"))
                source_text = html.escape(doc.page_content[:1000])

                st.markdown(
                    f"""
                    <div class="source-box">
                        <div class="source-title">Źródło {i}: {source_name}</div>
                        <div class="source-text">{source_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )