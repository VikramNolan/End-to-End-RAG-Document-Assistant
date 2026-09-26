import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Locate .env in project root or current working dir
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = ROOT_DIR / ".env"
if not ENV_FILE.exists():
    ENV_FILE = Path(".env")

class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG ChatBot Assistant"
    
    # LLM Settings (Groq)
    GROQ_API_KEY: str = Field(default="", validation_alias="GROQ_API_KEY")
    GROQ_MODEL: str = Field(default="qwen/qwen3.8-27b", validation_alias="GROQ_MODEL")
    
    # Vector Database Settings (Pinecone)
    PINECONE_API_KEY: str = Field(default="", validation_alias="PINECONE_API_KEY")
    PINECONE_INDEX: str = Field(default="rag-chatbot")
    PINECONE_INDEX_NAME: str = Field(default="")
    PINECONE_ENVIRONMENT: str = Field(default="", validation_alias="PINECONE_ENVIRONMENT")
    PINECONE_ENV: str = Field(default="")
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    
    # Retrieval & Chunking Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100
    TOP_K: int = 4

    @property
    def effective_index_name(self) -> str:
        name = self.PINECONE_INDEX_NAME.strip() or self.PINECONE_INDEX.strip() or "rag-chatbot"
        if "://" in name:
            # Extract index name from host URL, e.g. https://rag-chatbot-80hfsbf... -> rag-chatbot
            host_part = name.split("://")[-1].split(".")[0]
            # If host_part has suffix hash like ragweb-80hfsbf or rag-chatbot-80hfsbf
            parts = host_part.rsplit("-", 1)
            return parts[0] if len(parts) > 1 else host_part
        return name

    @property
    def effective_environment(self) -> str:
        return self.PINECONE_ENV.strip() or self.PINECONE_ENVIRONMENT.strip() or "us-east-1"

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE) if ENV_FILE.exists() else ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
