from src.loaders import load_text_documents
from src.chunking import split_documents
from src.vector_store import build_vector_store
from dotenv import load_dotenv


def main():
    docs = load_text_documents("data/documents")
    print(f"Wczytano dokumentów: {len(docs)}")

    chunks = split_documents(docs)
    print(f"Utworzono chunków: {len(chunks)}")

    build_vector_store(chunks)
    print("Baza wektorowa została utworzona.")


if __name__ == "__main__":
    load_dotenv()
    main()