"""
Multi-agent RAG orchestration using LangGraph.

Pipeline:
    router  -> decides if retrieval is needed
        |--(no)--> generator (direct answer, no sources)
        |--(yes)-> refiner -> retriever -> grader -> [rewrite? loop once] -> generator

The refiner uses the LLM to turn the raw user question (which may rely on
conversational context, be vague, or use synonyms) into one self-contained,
keyword-rich search query plus a couple of paraphrases. The retriever then
fans out across those queries and merges the results, giving the grader and
the generator a much better recall surface than a single literal query.

Each node is a small "agent" with a single responsibility. The graph state
accumulates a trace so the API can return what happened.
"""
from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from langgraph.graph import StateGraph, END

from ..config import settings
from ..llm import complete
from ..vectorstore import query as vs_query


# --- State --------------------------------------------------------------------

class RAGState(TypedDict, total=False):
    question: str
    original_question: str
    history: list[dict[str, str]]
    top_k: int
    needs_retrieval: bool
    search_queries: list[str]
    documents: list[dict[str, Any]]
    relevant_documents: list[dict[str, Any]]
    rewrites: int
    answer: str
    trace: list[dict[str, str]]


def _log(state: RAGState, name: str, detail: str) -> None:
    state.setdefault("trace", []).append({"name": name, "detail": detail})


# --- Agent: Router ------------------------------------------------------------

ROUTER_SYSTEM = (
    "You are a routing agent for a RAG system. Decide whether the user's question "
    "requires looking up information in the indexed document collection. "
    "Answer strictly with JSON of the form {\"retrieve\": true} or {\"retrieve\": false}. "
    "Choose false ONLY for pure chit-chat, greetings, or trivial reformulations."
)


def router_node(state: RAGState) -> RAGState:
    q = state["question"]
    try:
        raw = complete(f"Question: {q}\nJSON:", system=ROUTER_SYSTEM, temperature=0.0)
        # Extract a JSON object from the LLM output
        start, end = raw.find("{"), raw.rfind("}")
        decision = json.loads(raw[start : end + 1]) if start != -1 else {"retrieve": True}
        needs = bool(decision.get("retrieve", True))
    except Exception:
        needs = True
    state["needs_retrieval"] = needs
    _log(state, "router", f"needs_retrieval={needs}")
    return state


# --- Agent: Refiner (prompt/query refactoring) -------------------------------

REFINER_SYSTEM = (
    "You are a query-refinement agent for a Retrieval-Augmented Generation "
    "system. Your job is to turn the user's last message into queries that "
    "maximize vector-search recall over a document collection.\n\n"
    "Rules:\n"
    "- Resolve pronouns and references using the conversation history so each "
    "  query is self-contained.\n"
    "- Keep the user's language (French stays French, English stays English).\n"
    "- Strip greetings, politeness and meta-talk; keep only the information need.\n"
    "- Produce 1 primary rewrite (concise, keyword-rich, declarative) and 0-2 "
    "  paraphrases that use synonyms or alternative phrasings likely to match "
    "  the wording used in technical documents.\n"
    "- Do NOT invent facts. Do NOT answer the question.\n\n"
    "Respond with ONLY a JSON object of the form:\n"
    '{"primary": "...", "paraphrases": ["...", "..."]}'
)


def _format_history(history: list[dict[str, str]], max_turns: int = 4) -> str:
    if not history:
        return "(no prior turns)"
    last = history[-max_turns * 2 :]
    return "\n".join(f"{m.get('role', '?').upper()}: {m.get('content', '')}" for m in last)


def refiner_node(state: RAGState) -> RAGState:
    original = state["question"]
    hist = _format_history(state.get("history", []))
    user_prompt = (
        f"Conversation so far:\n{hist}\n\n"
        f"User's last message:\n{original}\n\n"
        "Return the JSON object now."
    )
    queries: list[str] = []
    try:
        raw = complete(user_prompt, system=REFINER_SYSTEM, temperature=0.1)
        start, end = raw.find("{"), raw.rfind("}")
        data = json.loads(raw[start : end + 1]) if start != -1 else {}
        primary = (data.get("primary") or "").strip()
        paras = [p.strip() for p in (data.get("paraphrases") or []) if isinstance(p, str) and p.strip()]
        if primary:
            queries.append(primary)
        queries.extend(paras[:2])
    except Exception:
        queries = []

    # Always keep the original as a safety net (deduplicated, case-insensitive).
    seen: set[str] = set()
    final: list[str] = []
    for q in [*queries, original]:
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            final.append(q.strip())

    state["search_queries"] = final
    # The primary refined query becomes the working `question` for retrieval.
    state["question"] = final[0]
    _log(state, "refiner", f"{len(final)} query/queries: {final!r}")
    return state


# --- Agent: Retriever ---------------------------------------------------------

def _merge_results(batches: list[list[dict[str, Any]]], k: int) -> list[dict[str, Any]]:
    """Merge several top-K result lists, keeping the best score per chunk id."""
    best: dict[str, dict[str, Any]] = {}
    for batch in batches:
        for d in batch:
            cur = best.get(d["id"])
            if cur is None or d["score"] > cur["score"]:
                best[d["id"]] = d
    return sorted(best.values(), key=lambda x: x["score"], reverse=True)[:k]


def retriever_node(state: RAGState) -> RAGState:
    k = state.get("top_k") or settings.top_k
    queries = state.get("search_queries") or [state["question"]]
    batches = [vs_query(q, top_k=k) for q in queries]
    docs = _merge_results(batches, k)
    state["documents"] = docs
    _log(
        state,
        "retriever",
        f"retrieved {len(docs)} unique chunks from {len(queries)} query/queries (top_k={k})",
    )
    return state


# --- Agent: Grader ------------------------------------------------------------

GRADER_SYSTEM = (
    "You are a relevance grader. Given a user question and a document chunk, "
    "reply with a single token: YES if the chunk is relevant to answering the "
    "question, otherwise NO. Be strict."
)


def grader_node(state: RAGState) -> RAGState:
    q = state["question"]
    kept: list[dict[str, Any]] = []
    for d in state.get("documents", []):
        prompt = f"Question:\n{q}\n\nChunk:\n{d['chunk']}\n\nAnswer YES or NO:"
        try:
            verdict = complete(prompt, system=GRADER_SYSTEM, temperature=0.0).upper()
        except Exception:
            verdict = "YES"  # fail-open: keep the chunk
        if verdict.startswith("YES"):
            kept.append(d)
    state["relevant_documents"] = kept
    _log(state, "grader", f"kept {len(kept)}/{len(state.get('documents', []))} chunks")
    return state


# --- Agent: Query rewriter ----------------------------------------------------

REWRITER_SYSTEM = (
    "You rewrite a user question to improve retrieval from a vector database. "
    "Produce a single, self-contained, keyword-rich rewrite. Output the rewrite only."
)


def rewriter_node(state: RAGState) -> RAGState:
    state["rewrites"] = state.get("rewrites", 0) + 1
    original = state.get("original_question", state["question"])
    try:
        new_q = complete(f"Original: {original}", system=REWRITER_SYSTEM, temperature=0.2)
    except Exception:
        new_q = original
    new_q = new_q.strip().strip('"')
    state["question"] = new_q
    # Replace the query set so the retriever re-runs against the new phrasing
    # (plus the original as a safety net).
    state["search_queries"] = list(dict.fromkeys([new_q, original]))
    _log(state, "rewriter", f"rewrite #{state['rewrites']}: {new_q!r}")
    return state


# --- Agent: Generator ---------------------------------------------------------

GENERATOR_SYSTEM = (
    "You are a helpful assistant answering strictly from the provided context. "
    "Cite the sources you used inline using bracketed indices like [1], [2] that "
    "match the numbered context blocks. If the context does not contain the "
    "answer, say so explicitly. Answer in the user's language."
)

DIRECT_SYSTEM = (
    "You are a concise assistant. Answer the user directly. "
    "Do not invent sources."
)


def _format_context(docs: list[dict[str, Any]]) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        src = d.get("metadata", {}).get("source", "unknown")
        blocks.append(f"[{i}] (source: {src})\n{d['chunk']}")
    return "\n\n".join(blocks)


def generator_node(state: RAGState) -> RAGState:
    docs = state.get("relevant_documents") or state.get("documents") or []
    history = state.get("history", [])
    if not docs:
        msgs = [{"role": "system", "content": DIRECT_SYSTEM}, *history,
                {"role": "user", "content": state["question"]}]
        from ..llm import chat as llm_chat
        answer = llm_chat(msgs, temperature=0.3)
        _log(state, "generator", "answered without context")
    else:
        context = _format_context(docs)
        user = (
            f"Context:\n{context}\n\n"
            f"Question: {state.get('original_question', state['question'])}\n\n"
            "Write a grounded answer and cite sources as [n]."
        )
        msgs = [{"role": "system", "content": GENERATOR_SYSTEM}, *history,
                {"role": "user", "content": user}]
        from ..llm import chat as llm_chat
        answer = llm_chat(msgs, temperature=0.2)
        _log(state, "generator", f"answered using {len(docs)} chunks")
    state["answer"] = answer
    return state


# --- Edges --------------------------------------------------------------------

def route_after_router(state: RAGState) -> Literal["refiner", "generator"]:
    return "refiner" if state.get("needs_retrieval", True) else "generator"


def route_after_grader(state: RAGState) -> Literal["generator", "rewriter"]:
    if state.get("relevant_documents"):
        return "generator"
    if state.get("rewrites", 0) < 1:
        return "rewriter"
    return "generator"  # give up, generator will say "not found"


# --- Build graph --------------------------------------------------------------

def build_graph():
    g = StateGraph(RAGState)
    g.add_node("router", router_node)
    g.add_node("refiner", refiner_node)
    g.add_node("retriever", retriever_node)
    g.add_node("grader", grader_node)
    g.add_node("rewriter", rewriter_node)
    g.add_node("generator", generator_node)

    g.set_entry_point("router")
    g.add_conditional_edges("router", route_after_router,
                            {"refiner": "refiner", "generator": "generator"})
    g.add_edge("refiner", "retriever")
    g.add_edge("retriever", "grader")
    g.add_conditional_edges("grader", route_after_grader,
                            {"generator": "generator", "rewriter": "rewriter"})
    g.add_edge("rewriter", "retriever")
    g.add_edge("generator", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run(question: str, history: list[dict[str, str]] | None = None,
        top_k: int | None = None) -> RAGState:
    initial: RAGState = {
        "question": question,
        "original_question": question,
        "history": history or [],
        "top_k": top_k or settings.top_k,
        "rewrites": 0,
        "trace": [],
    }
    return get_graph().invoke(initial)
