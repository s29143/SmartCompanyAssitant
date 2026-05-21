from langchain_openai import ChatOpenAI

from src.vector_store import load_vector_store
from src.prompts import RAG_PROMPT_TEMPLATE, QUERY_REWRITE_PROMPT


def format_docs(docs):
    return "\n\n".join(
        f"Źródło: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )


def format_chat_history(chat_history):
    if not chat_history:
        return "Brak historii rozmowy."

    formatted = []

    for message in chat_history:
        role = "Użytkownik" if message["role"] == "user" else "Asystent"
        formatted.append(f"{role}: {message['content']}")

    return "\n".join(formatted)


def rewrite_query(question: str, chat_history=None) -> str:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = QUERY_REWRITE_PROMPT.format(
        question=question,
        chat_history=format_chat_history(chat_history)
    )

    response = llm.invoke(prompt)

    return response.content.strip()


def ask_rag(question: str, chat_history=None):
    rewritten_query = rewrite_query(
        question=question,
        chat_history=chat_history
    )

    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(rewritten_query)

    context = format_docs(docs)

    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    response = llm.invoke(prompt)
    print(rewritten_query)
    return {
        "answer": response.content,
        "sources": docs,
        "rewritten_query": rewritten_query
    }