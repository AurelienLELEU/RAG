from fastapi import APIRouter, HTTPException

from ..agents.orchestrator import run as run_agents
from ..models import AgentStep, ChatRequest, ChatResponse, Source

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    try:
        state = run_agents(req.question, history=req.history, top_k=req.top_k)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Agent pipeline failed: {e}") from e

    docs = state.get("relevant_documents") or state.get("documents") or []
    sources = [
        Source(
            id=d["id"],
            document=d.get("metadata", {}).get("source", "unknown"),
            chunk=d["chunk"],
            score=d.get("score", 0.0),
            metadata=d.get("metadata", {}),
        )
        for d in docs
    ]
    trace = [AgentStep(**s) for s in state.get("trace", [])]
    return ChatResponse(answer=state.get("answer", ""), sources=sources, trace=trace)
