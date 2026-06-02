from typing import Any
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = None
    history: list[dict[str, str]] = Field(default_factory=list)


class Source(BaseModel):
    id: str
    document: str
    chunk: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentStep(BaseModel):
    name: str
    detail: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    trace: list[AgentStep]


class IngestResponse(BaseModel):
    indexed_files: list[str]
    chunks: int


class CollectionInfo(BaseModel):
    name: str
    count: int
