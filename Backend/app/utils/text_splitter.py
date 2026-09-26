import re
from typing import List, Dict, Any, Optional

class RecursiveCharacterTextSplitter:
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)

        good_splits: List[str] = []
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_splits = self._split_text(s, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    # Fallback slice if no more separators
                    for j in range(0, len(s), self.chunk_size - self.chunk_overlap):
                        good_splits.append(s[j:j + self.chunk_size])

        # Merge splits up to chunk_size with chunk_overlap
        current_chunk = []
        current_length = 0

        for piece in good_splits:
            piece_len = len(piece) + (len(separator) if current_chunk else 0)
            if current_length + piece_len <= self.chunk_size:
                current_chunk.append(piece)
                current_length += piece_len
            else:
                if current_chunk:
                    merged = separator.join(current_chunk).strip()
                    if merged:
                        final_chunks.append(merged)
                    
                    # Backtrack for overlap
                    overlap_chunk = []
                    overlap_length = 0
                    for p in reversed(current_chunk):
                        p_len = len(p) + (len(separator) if overlap_chunk else 0)
                        if overlap_length + p_len <= self.chunk_overlap:
                            overlap_chunk.insert(0, p)
                            overlap_length += p_len
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_length = overlap_length

                current_chunk.append(piece)
                current_length += len(piece) + (len(separator) if len(current_chunk) > 1 else 0)

        if current_chunk:
            merged = separator.join(current_chunk).strip()
            if merged:
                final_chunks.append(merged)

        return final_chunks

    def split_text(self, text: str) -> List[str]:
        cleaned = text.replace("\r\n", "\n").strip()
        if not cleaned:
            return []
        return self._split_text(cleaned, self.separators)

    def create_chunks(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        text_chunks = self.split_text(text)
        metadata = metadata or {}
        doc_id = metadata.get("doc_id", "doc")
        
        chunks = []
        for idx, chunk_text in enumerate(text_chunks):
            chunk_id = f"{doc_id}_chunk_{idx}"
            chunk_metadata = {
                **metadata,
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "total_chunks": len(text_chunks),
                "text": chunk_text
            }
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        return chunks
