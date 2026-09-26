import json
from typing import List, Dict, Any, Generator, Optional
from app.config import settings
from app.services.embedding_service import embedding_service
from app.database.pinecone_client import vector_store
from app.services.groq_service import groq_service

SYSTEM_PROMPT = """You are a highly capable and intelligent AI RAG Assistant.
Your task is to answer the user's inquiry accurately and comprehensively, prioritizing the retrieved context documents provided below.

Guidelines:
1. Always reference and cite the source document name when drawing information from the context.
2. If the retrieved context contains relevant information, synthesize it into a clear, direct, and well-structured answer (use bullet points or headers if appropriate).
3. If the context does not contain sufficient details to answer the question, state that politely, and then provide the best answer possible based on general knowledge while making it clear it was not found in the documents.
4. Keep your answer professional, concise, and helpful."""

class RAGService:
    def __init__(self):
        self.top_k = settings.TOP_K

    def retrieve_context(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        query_embedding = embedding_service.embed_query(query)
        matches = vector_store.query(query_embedding, top_k=k)
        return matches

    def build_context_prompt(self, matches: List[Dict[str, Any]]) -> str:
        if not matches:
            return "No relevant context documents found in the database."

        context_parts = []
        for i, match in enumerate(matches, 1):
            meta = match.get("metadata", {})
            filename = meta.get("filename", "Document")
            chunk_idx = meta.get("chunk_index", 0)
            score = match.get("score", 0.0)
            text = match.get("text", "")
            context_parts.append(
                f"[Source {i}: {filename} (Chunk #{chunk_idx}, Similarity: {score})]\n{text}"
            )
        return "\n\n---\n\n".join(context_parts)

    def prepare_messages(
        self,
        query: str,
        matches: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        context_str = self.build_context_prompt(matches)
        
        system_content = f"{SYSTEM_PROMPT}\n\n### RETRIEVED CONTEXT:\n{context_str}"
        messages = [{"role": "system", "content": system_content}]

        # Append recent history if provided (limit to last 6 turns)
        if chat_history:
            recent_history = chat_history[-6:]
            for msg in recent_history:
                role = msg.get("role")
                content = msg.get("content")
                if role in ["user", "assistant"] and content:
                    messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": query})
        return messages

    def format_sources(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sources = []
        for m in matches:
            meta = m.get("metadata", {})
            sources.append({
                "chunk_id": m.get("id"),
                "doc_id": meta.get("doc_id", ""),
                "filename": meta.get("filename", "Unknown"),
                "score": m.get("score", 0.0),
                "text_snippet": m.get("text", "")[:280] + ("..." if len(m.get("text", "")) > 280 else ""),
                "chunk_index": meta.get("chunk_index", 0)
            })
        return sources

    def answer_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        matches = self.retrieve_context(query)
        sources = self.format_sources(matches)
        messages = self.prepare_messages(query, matches, chat_history)

        answer = groq_service.generate_completion(messages)

        return {
            "answer": answer,
            "sources": sources,
            "model": groq_service.model,
            "vector_mode": vector_store.client_mode,
            "total_sources_retrieved": len(sources)
        }

    def stream_query(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Generator[str, None, None]:
        matches = self.retrieve_context(query)
        sources = self.format_sources(matches)
        messages = self.prepare_messages(query, matches, chat_history)

        # First yield the metadata event containing sources
        meta_event = {
            "type": "metadata",
            "sources": sources,
            "model": groq_service.model,
            "vector_mode": vector_store.client_mode
        }
        yield f"data: {json.dumps(meta_event)}\n\n"

        # Stream LLM tokens
        for token in groq_service.generate_completion_stream(messages):
            token_event = {"type": "token", "content": token}
            yield f"data: {json.dumps(token_event)}\n\n"

        # End event
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

rag_service = RAGService()