import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


DB_DIR = "db/chroma"


def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def build_vector_store(chunks: list[Document]):
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