from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes import chat, ingest

app = FastAPI(
    title="Local RAG API",
    version="1.0.0",
    description="A local RAG backend with multi-agent orchestration. "
                "Embeddings: sentence-transformers. LLM: Ollama. Vector DB: ChromaDB.",
)

# Open CORS so any frontend (Streamlit, React, curl, ...) can hit the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Local RAG API",
        "llm": settings.llm_model,
        "embeddings": settings.embedding_model,
        "collection": settings.collection_name,
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
