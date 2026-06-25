from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag_chain import ask_rag

app = FastAPI(title="SmartCompany RAG API")


class QueryRequest(BaseModel):
    question: str
    chat_history: list[dict] | None = None


class QueryResponse(BaseModel):
    answer: str
    raw_answer: str
    sources: list[str]
    rewritten_query: str


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    try:
        result = ask_rag(
            question=request.question,
            chat_history=request.chat_history,
        )
        return QueryResponse(
            answer=result["answer"],
            raw_answer=result["raw_answer"],
            sources=[
                doc.metadata.get("source", "unknown")
                for doc in result["sources"]
            ],
            rewritten_query=result["rewritten_query"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
