from langchain_openai import ChatOpenAI
from src.vector_store import load_vector_store
from src.prompts import RAG_PROMPT_TEMPLATE


def format_docs(docs):
    return "\n\n".join(
        f"Źródło: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
        for doc in docs
    )


def ask_rag(question: str):
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    context = format_docs(docs)

    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": docs
    }