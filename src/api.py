import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from dotenv import load_dotenv
from src.rag_chain import ask_rag

load_dotenv()
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="SmartCompany RAG API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)


class ChatMessage(BaseModel):
    role: str
    content: str = Field(max_length=2000)

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v):
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    chat_history: list[ChatMessage] | None = Field(default=None, max_length=10)


class QueryResponse(BaseModel):
    answer: str
    raw_answer: str
    sources: list[str]
    rewritten_query: str
    source_recommendations: list[dict[str, str]]


@app.post("/ask", response_model=QueryResponse)
@limiter.limit("10/minute")
def ask(request: Request, body: QueryRequest):
    try:
        result = ask_rag(
            question=body.question,
            chat_history=(
                [m.model_dump() for m in body.chat_history]
                if body.chat_history
                else None
            ),
        )
        return QueryResponse(
            answer=result["answer"],
            raw_answer=result["raw_answer"],
            sources=[
                doc.metadata.get("source", "unknown")
                for doc in result["sources"]
            ],
            rewritten_query=result["rewritten_query"],
            source_recommendations=result.get("source_recommendations", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}
