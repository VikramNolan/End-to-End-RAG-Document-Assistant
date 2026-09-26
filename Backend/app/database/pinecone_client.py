import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from app.config import settings

logger = logging.getLogger(__name__)

class VectorStoreClient:
    def __init__(self):
        self.api_key = settings.PINECONE_API_KEY.strip() or os.getenv("PINECONE_API_KEY", "").strip()
        self.index_name = settings.effective_index_name
        self.client_mode = "local"
        self.pc = None
        self.index = None
        
        # Local persistence directory
        self.local_data_dir = Path(__file__).resolve().parent.parent / "data"
        self.local_data_dir.mkdir(parents=True, exist_ok=True)
        self.local_store_file = self.local_data_dir / "local_vectors.json"
        
        self.local_vectors: List[Dict[str, Any]] = []
        self._load_local_store()

        if self.api_key:
            try:
                from pinecone import Pinecone, ServerlessSpec
                self.pc = Pinecone(api_key=self.api_key)
                
                existing_indexes = [idx["name"] for idx in self.pc.list_indexes()]
                
                # If target index is in existing indexes, check dimension
                target_idx = self.index_name
                if "rag-chatbot" in existing_indexes:
                    target_idx = "rag-chatbot"
                elif target_idx not in existing_indexes:
                    logger.info(f"Creating Pinecone index: {target_idx}")
                    self.pc.create_index(
                        name=target_idx,
                        dimension=384,
                        metric="cosine",
                        spec=ServerlessSpec(cloud="aws", region=settings.effective_environment)
                    )

                self.index_name = target_idx
                self.index = self.pc.Index(self.index_name)
                self.client_mode = "pinecone"
                logger.info(f"Connected to Pinecone index: {self.index_name}")
            except Exception as e:
                logger.warning(f"Failed to connect to Pinecone: {e}. Falling back to Local Vector Store.")
                self.client_mode = "local"
        else:
            logger.info("No PINECONE_API_KEY found. Operating in Local Vector Store mode.")

    def _load_local_store(self):
        if self.local_store_file.exists():
            try:
                with open(self.local_store_file, "r", encoding="utf-8") as f:
                    self.local_vectors = json.load(f)
            except Exception as e:
                logger.error(f"Error loading local vector store: {e}")
                self.local_vectors = []

    def _save_local_store(self):
        try:
            with open(self.local_store_file, "w", encoding="utf-8") as f:
                json.dump(self.local_vectors, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving local vector store: {e}")

    def upsert_vectors(self, vectors: List[Dict[str, Any]]) -> bool:
        if not vectors:
            return True

        # Always update local vectors list for document tracking and local queries
        existing_ids = {v["id"] for v in self.local_vectors}
        for vec in vectors:
            if vec["id"] in existing_ids:
                self.local_vectors = [v for v in self.local_vectors if v["id"] != vec["id"]]
            self.local_vectors.append(vec)
        self._save_local_store()

        # If Pinecone is active, also upsert to Pinecone
        if self.client_mode == "pinecone" and self.index:
            try:
                batch_size = 100
                for i in range(0, len(vectors), batch_size):
                    batch = vectors[i:i + batch_size]
                    self.index.upsert(vectors=batch)
                return True
            except Exception as e:
                logger.error(f"Pinecone upsert failed: {e}. Kept in local store.")

        return True

    def query(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        if self.client_mode == "pinecone" and self.index:
            try:
                res = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
                matches = []
                for match in res.get("matches", []):
                    metadata = match.get("metadata", {})
                    matches.append({
                        "id": match["id"],
                        "score": round(float(match.get("score", 0.0)), 4),
                        "text": metadata.get("text", ""),
                        "metadata": metadata
                    })
                if matches:
                    return matches
            except Exception as e:
                logger.error(f"Pinecone query failed: {e}. Falling back to local store.")

        # Local vector cosine similarity search
        if not self.local_vectors or not query_embedding:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        results = []
        for item in self.local_vectors:
            emb = np.array(item["values"], dtype=np.float32)
            emb_norm = np.linalg.norm(emb)
            if emb_norm == 0:
                score = 0.0
            else:
                score = float(np.dot(q_vec, emb) / (q_norm * emb_norm))
            
            metadata = item.get("metadata", {})
            results.append({
                "id": item["id"],
                "score": round(score, 4),
                "text": metadata.get("text", ""),
                "metadata": metadata
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete_document(self, doc_id: str) -> bool:
        deleted = False
        if self.client_mode == "pinecone" and self.index:
            try:
                # Find vector IDs matching doc_id
                ids_to_delete = [
                    v["id"] for v in self.local_vectors
                    if v.get("metadata", {}).get("doc_id") == doc_id
                ]
                if ids_to_delete:
                    self.index.delete(ids=ids_to_delete)
                deleted = True
            except Exception as e:
                logger.error(f"Pinecone delete failed: {e}")

        # Local vector store delete
        initial_len = len(self.local_vectors)
        self.local_vectors = [
            v for v in self.local_vectors
            if v.get("metadata", {}).get("doc_id") != doc_id
        ]
        if len(self.local_vectors) < initial_len:
            self._save_local_store()
            deleted = True

        return deleted

    def clear_all(self) -> bool:
        """Clear all indexed vectors from vector store."""
        if self.client_mode == "pinecone" and self.index:
            try:
                self.index.delete(delete_all=True)
            except Exception as e:
                logger.error(f"Pinecone delete_all failed: {e}")
        self.local_vectors = []
        self._save_local_store()
        return True

    def list_documents(self) -> List[Dict[str, Any]]:
        docs: Dict[str, Dict[str, Any]] = {}
        for item in self.local_vectors:
            meta = item.get("metadata", {})
            doc_id = meta.get("doc_id")
            if not doc_id:
                continue
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("filename", "Unknown document"),
                    "chunk_count": 0,
                    "uploaded_at": meta.get("uploaded_at", ""),
                    "file_type": meta.get("file_type", "txt"),
                    "file_size": meta.get("file_size", 0)
                }
            docs[doc_id]["chunk_count"] += 1
        return list(docs.values())

vector_store = VectorStoreClient()
