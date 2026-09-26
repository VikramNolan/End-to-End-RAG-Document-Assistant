from typing import List, Optional
import numpy as np
from app.config import settings

class EmbeddingService:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None
        self._dimension = 384  # Default for BAAI/bge-small-en-v1.5 and all-MiniLM-L6-v2

    @property
    def model(self):
        if self._model is None:
            from fastembed import TextEmbedding
            # Handles models like BAAI/bge-small-en-v1.5
            self._model = TextEmbedding(model_name=self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimension

    def get_embedding_model(self):
        return self.model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = list(self.model.embed(texts))
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> List[float]:
        if not text.strip():
            return [0.0] * self.dimension
        embeddings = list(self.model.embed([text]))
        return embeddings[0].tolist()

embedding_service = EmbeddingService()
