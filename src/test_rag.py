from src.rag_chain import ask_rag
from dotenv import load_dotenv


def main():
    while True:
        question = input("\nPytanie: ").strip()
        if question.lower() in {"exit", "quit"}:
            break

        result = ask_rag(question)

        print("\nOdpowiedź:")
        print(result["answer"])

        print("\nŹródła:")
        for doc in result["sources"]:
            print(f"- {doc.metadata.get('source', 'unknown')}")


if __name__ == "__main__":
    load_dotenv()
    main()