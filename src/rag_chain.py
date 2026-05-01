from langchain_openai import ChatOpenAI
from src.vector_store import load_vector_store
from src.prompts import RAG_PROMPT_TEMPLATE


def format_docs(docs):
    return "\n\n".join(
        f"Źródło: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )


def format_chat_history(chat_history):
    if not chat_history:
        return "Brak wcześniejszej historii rozmowy."

    formatted = []

    for message in chat_history:
        role = "Użytkownik" if message["role"] == "user" else "Asystent"
        formatted.append(f"{role}: {message['content']}")

    return "\n".join(formatted)


def ask_rag(question: str, chat_history=None):
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    docs = retriever.invoke(question)

    context = format_docs(docs)
    history_text = format_chat_history(chat_history)

    prompt = RAG_PROMPT_TEMPLATE.format(
        history_text=history_text,
        context=context,
        question=question
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": docs,
    }
