from langchain_openai import ChatOpenAI

from src.vector_store import load_vector_store
from src.prompts import RAG_PROMPT_TEMPLATE, QUERY_REWRITE_PROMPT, SOURCE_VERIFIER_PROMPT

def get_relevant_docs(query: str):
    vector_store = load_vector_store()

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 6,
            "fetch_k": 20,
            "lambda_mult": 0.5,
        }
    )

    return retriever.invoke(query)


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



def verify_answer(question: str, context: str, answer: str) -> str:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    prompt = SOURCE_VERIFIER_PROMPT.format(
        question=question,
        context=context,
        answer=answer
    )

    response = llm.invoke(prompt)

    return response.content.strip()

def ask_rag(question: str, chat_history=None):
    print(f"\n[USER QUESTION]")
    print(question)

    rewritten_query = rewrite_query(
        question=question,
        chat_history=chat_history
    )

    print(f"\n[QUERY REWRITE AGENT]")
    print(rewritten_query)

    docs = get_relevant_docs(rewritten_query)

    print(f"\n[RETRIEVER]")
    print(
        [doc.metadata.get("source", "unknown") for doc in docs]
    )

    context = format_docs(docs)
    history_text = format_chat_history(chat_history)

    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        chat_history=history_text
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    raw_response = llm.invoke(prompt)
    raw_answer = raw_response.content.strip()

    verified_answer = verify_answer(
        question=question,
        context=context,
        answer=raw_answer
    )

    print(f"\n[SOURCE VERIFIER AGENT]")
    print("Answer verified.")

    return {
        "answer": verified_answer,
        "raw_answer": raw_answer,
        "sources": docs,
        "rewritten_query": rewritten_query,
    }