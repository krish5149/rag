from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any
from pathlib import Path
import shutil

from app.upload_fie import router as upload_router
from app.graphs.query_graph import query_app as rag_graph

class QueryRequest(BaseModel):
    query: str

app = FastAPI(title="LangGraph RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload_router, prefix="/upload", tags=["Upload"])

static_dir = Path(__file__).resolve().parent / "static"
DATA_DIR = Path(__file__).resolve().parent / "data"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("shutdown")
def cleanup_storage() -> None:
    for path in (CHROMA_DIR, DATA_DIR):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


def normalize_reranked_docs(reranked_docs: list[Any]) -> list[Any]:
    normalized = []
    for item in reranked_docs:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            text, score = item
            try:
                score = float(score)
            except (TypeError, ValueError):
                pass
            normalized.append({"document": text, "score": score})
        else:
            normalized.append(item)
    return normalized


@app.get("/")
def root() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.exception_handler(Exception)
async def internal_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query_rag(request: QueryRequest) -> dict[str, Any]:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    state = {
        "query": request.query,
        "retrieved_chunks": [],
        "reranked_docs": [],
        "answer": "",
        "relevent_or_not": None,
        "query_rewrite_count": 0
    }

    result = rag_graph.invoke(state,{"configurable": {"thread_id": "thread_id_123"}})  #type: ignore
    answer = result.get("answer") or result.get("generation") or ""
    reranked_docs = normalize_reranked_docs(result.get("reranked_docs", []))

    return {
        "query": request.query,
        "answer": answer,
        "relevant": result.get("relevent_or_not"),
        "reranked_docs": reranked_docs
    }
