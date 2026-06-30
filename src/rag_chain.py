import logging
import re

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from src.vector_store import load_vector_store
from src.prompts import (
    RAG_SYSTEM_PROMPT,
    RAG_USER_TEMPLATE,
    QUERY_REWRITE_SYSTEM_PROMPT,
    QUERY_REWRITE_USER_TEMPLATE,
    SOURCE_VERIFIER_SYSTEM_PROMPT,
    SOURCE_VERIFIER_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = "Brak wystarczających informacji w dostępnych dokumentach."

# Te tagi ogradzają treść użytkownika w prompcie (patrz src/prompts.py).
# Treść kontrolowana przez anonimowego usera (pytanie, historia czatu) jest
# jedynym realnym wektorem injection w tym systemie - dokumenty z bazy
# wektorowej pochodzą wyłącznie ze strony firmy i są zaufane.
_FENCE_TAGS = ("chat_history", "user_question")


def _sanitize_for_prompt(text: str) -> str:
    """Neutralizuje tagi ogradzające w treści kontrolowanej przez użytkownika,
    żeby nie mogła przedwcześnie zamknąć swojego ogrodzenia w prompcie."""
    if not text:
        return text
    sanitized = text
    for tag in _FENCE_TAGS:
        sanitized = re.sub(
            rf"</?{tag}>",
            lambda m: m.group(0).replace("<", "‹").replace(">", "›"),
            sanitized,
            flags=re.IGNORECASE,
        )
    return sanitized


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
        content = _sanitize_for_prompt(str(message["content"]))
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)


def build_source_recommendations(docs, question: str | None = None, max_recommendations: int | None = None):
    recommendations = []
    seen_sources = set()
    question_text = (question or "").lower()
    keywords = set(re.findall(r"[a-ząćęłńóśźż0-9]+", question_text))

    def is_relevant(doc, source_name: str) -> bool:
        if not question_text:
            return True

        content = " ".join(
            [
                str(doc.page_content or ""),
                str(doc.metadata.get("source", "") or ""),
                str(doc.metadata.get("title", "") or ""),
            ]
        ).lower()

        if not content:
            return True

        content_tokens = set(re.findall(r"[a-ząćęłńóśźż0-9]+", content))
        overlap = keywords & content_tokens
        if overlap:
            return True

        strong_topic_terms = {
            "cena": {"cena", "ceny", "cenę", "koszt", "koszty"},
            "usługa": {"usługa", "usługi", "usługę", "oferta", "ofertę"},
            "kontakt": {"kontakt", "skontakt", "formularz"},
            "o": {"o", "nas"},
            "proces": {"proces", "wdrożenie", "współpraca", "etap"},
        }

        for topic, terms in strong_topic_terms.items():
            if topic in question_text:
                if any(term in content for term in terms):
                    return True

        return source_name in question_text or any(term in source_name for term in keywords if len(term) > 2)

    for doc in docs:
        source = str(doc.metadata.get("source", "unknown") or "unknown").strip()
        if not source or source in seen_sources:
            continue

        seen_sources.add(source)

        source_name = source.split("/")[-1].split("\\")[-1].rsplit(".", 1)[0]
        source_name = re.sub(r"[^a-z0-9]+", "-", source_name.lower()).strip("-")

        if not is_relevant(doc, source_name):
            continue

        for key in ("url", "link", "slug", "path"):
            raw_value = doc.metadata.get(key, "")
            if isinstance(raw_value, str) and raw_value.strip():
                candidate = raw_value.strip()
                if candidate.startswith("http://") or candidate.startswith("https://"):
                    url = candidate
                    break
                url = candidate if candidate.startswith("/") else f"/{candidate.strip('/')}/"
                break
        else:
            url = f"/{source_name}/" if source_name else "/"

        recommendations.append(
            {
                "source": source,
                "url": url,
                "type": str(doc.metadata.get("type", "unknown")),
            }
        )

        if max_recommendations is not None and len(recommendations) >= max_recommendations:
            break

    if max_recommendations is None:
        if len(recommendations) <= 2:
            return recommendations[:1]
        if len(recommendations) <= 5:
            return recommendations[:3]
        return recommendations[:4]

    return recommendations


def rewrite_query(question: str, chat_history=None) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=20, max_retries=1)

    messages = [
        SystemMessage(content=QUERY_REWRITE_SYSTEM_PROMPT),
        HumanMessage(content=QUERY_REWRITE_USER_TEMPLATE.format(
            question=_sanitize_for_prompt(question),
            chat_history=format_chat_history(chat_history),
        )),
    ]

    try:
        response = llm.invoke(messages)
    except Exception:
        logger.exception("Query rewrite LLM call failed")
        raise RuntimeError("Nie udało się przetworzyć zapytania.") from None

    return response.content.strip()


def verify_answer(question: str, context: str, answer: str) -> str:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=20, max_retries=1)

    messages = [
        SystemMessage(content=SOURCE_VERIFIER_SYSTEM_PROMPT),
        HumanMessage(content=SOURCE_VERIFIER_USER_TEMPLATE.format(
            question=_sanitize_for_prompt(question),
            context=context,
            answer=answer,
        )),
    ]

    try:
        response = llm.invoke(messages)
    except Exception:
        logger.exception("Answer verification LLM call failed")
        raise RuntimeError("Nie udało się zweryfikować odpowiedzi.") from None

    return response.content.strip()


def ask_rag(question: str, chat_history=None):
    logger.debug("Incoming question (truncated): %s", question[:200])

    rewritten_query = rewrite_query(question=question, chat_history=chat_history)
    logger.debug("Rewritten query: %s", rewritten_query)

    try:
        docs = get_relevant_docs(rewritten_query)
    except Exception:
        logger.exception("Retriever failed")
        raise RuntimeError("Nie udało się wyszukać informacji w bazie wiedzy.") from None

    logger.debug("Retrieved sources: %s", [d.metadata.get("source", "unknown") for d in docs])

    if not docs:
        return {
            "answer": FALLBACK_MESSAGE,
            "raw_answer": FALLBACK_MESSAGE,
            "sources": [],
            "rewritten_query": rewritten_query,
            "source_recommendations": [],
        }

    context = format_docs(docs)
    history_text = format_chat_history(chat_history)

    llm = ChatOpenAI(model="gpt-5-nano", temperature=0, timeout=30, max_retries=1)

    messages = [
        SystemMessage(content=RAG_SYSTEM_PROMPT),
        HumanMessage(content=RAG_USER_TEMPLATE.format(
            context=context,
            question=_sanitize_for_prompt(question),
            chat_history=history_text,
        )),
    ]

    try:
        raw_response = llm.invoke(messages)
    except Exception:
        logger.exception("Generation LLM call failed")
        raise RuntimeError("Nie udało się wygenerować odpowiedzi.") from None

    raw_answer = raw_response.content.strip()

    if raw_answer == FALLBACK_MESSAGE:
        verified_answer = raw_answer
    else:
        verified_answer = verify_answer(
            question=question,
            context=context,
            answer=raw_answer,
        )

    logger.debug("Answer verified")

    source_recommendations = build_source_recommendations(docs, question=question)

    return {
        "answer": verified_answer,
        "raw_answer": raw_answer,
        "sources": docs,
        "rewritten_query": rewritten_query,
        "source_recommendations": source_recommendations,
    }