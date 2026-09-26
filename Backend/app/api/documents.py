from typing import List, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Response
from app.services.document_service import document_service
from app.database.pinecone_client import vector_store
from app.config import settings

router = APIRouter()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (PDF, DOCX, TXT, MD) to be chunked, embedded, and stored in vector DB."""
    try:
        result = await document_service.process_document(file)
        return {
            "success": True,
            "message": f"Successfully processed and indexed '{result['filename']}'",
            "document": result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@router.get("/")
def list_documents():
    """Retrieve all uploaded documents and their chunk counts."""
    docs = document_service.list_documents()
    return {
        "success": True,
        "count": len(docs),
        "documents": docs
    }

@router.get("/stats")
def get_stats():
    """System vector database stats and configuration."""
    docs = document_service.list_documents()
    total_chunks = sum(d.get("chunk_count", 0) for d in docs)
    total_size = sum(d.get("file_size", 0) for d in docs)
    return {
        "total_documents": len(docs),
        "total_chunks": total_chunks,
        "total_file_size": total_size,
        "vector_mode": vector_store.client_mode,
        "index_name": vector_store.index_name,
        "embedding_model": settings.EMBEDDING_MODEL,
        "chunk_size": settings.CHUNK_SIZE,
        "chunk_overlap": settings.CHUNK_OVERLAP
    }

@router.delete("/clear")
def clear_all_documents():
    """Clear the entire knowledge base (all documents and vector embeddings)."""
    document_service.clear_all_documents()
    return {
        "success": True,
        "message": "Entire knowledge base has been cleared successfully."
    }

@router.get("/{doc_id}/preview")
def get_document_preview(doc_id: str):
    """Retrieve document full text preview and breakdown of all chunk segments."""
    preview = document_service.get_document_preview(doc_id)
    if not preview:
        raise HTTPException(status_code=404, detail="Document preview not found.")
    return {
        "success": True,
        "preview": preview
    }

@router.get("/{doc_id}/download")
def download_document_text(doc_id: str):
    """Download the extracted text of the document as a text file."""
    preview = document_service.get_document_preview(doc_id)
    if not preview or not preview.get("full_text"):
        raise HTTPException(status_code=404, detail="Document text not found.")
    
    filename = preview.get("filename", f"document_{doc_id}")
    clean_filename = f"{filename.rsplit('.', 1)[0]}_extracted.txt"
    
    return Response(
        content=preview["full_text"],
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{clean_filename}"'
        }
    )

@router.delete("/{doc_id}")
def delete_document(doc_id: str):
    """Delete a document and all its embedded chunks from the vector database."""
    success = document_service.delete_document(doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found or already deleted.")
    return {
        "success": True,
        "message": f"Document '{doc_id}' successfully removed from vector store."
    }