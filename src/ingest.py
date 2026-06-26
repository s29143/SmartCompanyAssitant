from src.loaders import load_text_documents, load_wordpress_documents
from src.chunking import split_documents
from src.vector_store import build_vector_store
from dotenv import load_dotenv

WP_POSTS_CSV = "data/notepads/data/wp_posts.csv"
WP_POSTMETA_CSV = "data/notepads/data/wp_postmeta.csv"


def main():
    docs = load_text_documents("data/documents")
    print(f"Wczytano dokumentów txt: {len(docs)}")

    wp_docs = load_wordpress_documents(WP_POSTS_CSV, WP_POSTMETA_CSV)
    print(f"Wczytano postów WordPress: {len(wp_docs)}")
    docs.extend(wp_docs)

    chunks = split_documents(docs)
    print(f"Utworzono chunków: {len(chunks)}")

    build_vector_store(chunks)
    print("Baza wektorowa została utworzona.")


if __name__ == "__main__":
    load_dotenv()
    main()