from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.rag_service import rag_service
from app.services.groq_service import groq_service
from app.database.pinecone_client import vector_store

router = APIRouter()

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message text")

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question or query")
    history: Optional[List[ChatMessage]] = Field(default=[], description="Recent conversation history")

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    model: str
    vector_mode: str
    total_sources_retrieved: int

@router.post("/", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Execute a RAG query over uploaded documents using Groq LLM."""
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history]
    try:
        response = rag_service.answer_query(request.query, history_dicts)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@router.post("/stream")
def chat_stream(request: ChatRequest):
    """Stream live tokens from Groq LLM as Server-Sent Events (SSE)."""
    history_dicts = [{"role": msg.role, "content": msg.content} for msg in request.history]
    
    return StreamingResponse(
        rag_service.stream_query(request.query, history_dicts),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no"
        }
    )

@router.get("/status")
def get_chat_status():
    """Returns AI model and vector store status."""
    return {
        "groq_configured": groq_service.is_configured(),
        "groq_model": groq_service.model,
        "vector_store_mode": vector_store.client_mode,
        "pinecone_connected": vector_store.client_mode == "pinecone"
    }
