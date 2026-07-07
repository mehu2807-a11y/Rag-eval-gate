"""FastAPI service exposing the RAG pipeline.

Run with:
    uvicorn app.main:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app import config
from app.ingest import build_index
from app.retrieval import retrieve
from app.generate import generate_answer

app = FastAPI(title="RAG-Eval", version="0.1.0")


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None


class Source(BaseModel):
    source: str
    chunk_index: int
    distance: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    contexts = retrieve(req.question, top_k=req.top_k)
    answer = generate_answer(req.question, contexts)
    sources = [
        Source(source=c["source"], chunk_index=c["chunk_index"], distance=c["distance"])
        for c in contexts
    ]
    return AskResponse(answer=answer, sources=sources)


@app.post("/reindex")
def reindex():
    n = build_index(reset=True)
    return {"chunks_indexed": n}
