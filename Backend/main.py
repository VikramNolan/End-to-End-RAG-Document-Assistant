import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.api import chat, documents
from app.config import settings
from app.database.pinecone_client import vector_store
from app.services.groq_service import groq_service

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="High-performance RAG Assistant powered by FastAPI, Groq, FastEmbed & Pinecone",
    version="1.0.0"
)

# CORS Setup (Allows frontend to talk to backend from any port/origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])

@app.get("/api/health")
@app.get("/health")
def health_check():
    return {
        "status": "online",
        "system": settings.PROJECT_NAME,
        "groq_configured": groq_service.is_configured(),
        "vector_store_mode": vector_store.client_mode,
        "embedding_model": settings.EMBEDDING_MODEL
    }

@app.get("/api/stats")
def get_system_stats():
    from app.services.document_service import document_service
    docs = document_service.list_documents()
    total_chunks = sum(d.get("chunk_count", 0) for d in docs)
    return {
        "total_documents": len(docs),
        "total_chunks": total_chunks,
        "vector_mode": vector_store.client_mode,
        "embedding_model": settings.EMBEDDING_MODEL
    }

# Frontend static files hosting
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists() and (FRONTEND_DIR / "index.html").exists():
    css_dir = FRONTEND_DIR / "css"
    if css_dir.exists():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
    js_dir = FRONTEND_DIR / "js"
    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend_root():
        return FileResponse(FRONTEND_DIR / "index.html")
else:
    @app.get("/")
    def default_root():
        return {"status": "online", "system": settings.PROJECT_NAME}