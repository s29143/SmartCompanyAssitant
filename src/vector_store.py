import shutil
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

DB_DIR = "db/chroma"


def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def reset_vector_store():
    db_path = Path(DB_DIR)
    if db_path.exists():
        shutil.rmtree(db_path)


def build_vector_store(chunks: list[Document], reset: bool = True):
    if reset:
        reset_vector_store()

    embeddings = get_embeddings()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    return vector_store


def load_vector_store():
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )