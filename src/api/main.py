from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.retrieval.rag import query_rag
import logging

# Setup
app = FastAPI(
    title="RAG Pipeline API",
    description="Ask questions over AI research papers",
    version="1.0.0"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request and Response models
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks_used: int

# Routes
@app.get("/health")
def health():
    return {"status": "ok", "message": "RAG Pipeline is running!"}

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        logger.info(f"Question received: {request.question}")
        result = query_rag(request.question, request.top_k)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=result["chunks_used"]
        )
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))