from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    chroma_dir: str = "./data/chroma"
    collection_name: str = "rag_docs"
    documents_dir: str = "./data/documents"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    ollama_host: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"

    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    def ensure_dirs(self) -> None:
        Path(self.chroma_dir).mkdir(parents=True, exist_ok=True)
        Path(self.documents_dir).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
