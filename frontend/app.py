"""Streamlit frontend. Talks to the FastAPI backend over HTTP only."""
from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Local RAG", page_icon="📚", layout="wide")
st.title("📚 Local RAG")
st.caption(f"Backend: `{API_URL}`")


# --- Sidebar: ingestion & collection management -------------------------------

with st.sidebar:
    st.header("Collection")
    try:
        info = requests.get(f"{API_URL}/api/ingest/info", timeout=10).json()
        st.metric("Indexed chunks", info.get("count", 0))
    except Exception as e:  # noqa: BLE001
        st.error(f"API unreachable: {e}")

    st.divider()
    st.subheader("Upload documents")
    uploads = st.file_uploader(
        "PDF, DOCX, TXT, MD, HTML",
        type=["pdf", "docx", "txt", "md", "markdown", "html", "htm"],
        accept_multiple_files=True,
    )
    if st.button("Ingest uploads", disabled=not uploads, use_container_width=True):
        files = [("files", (u.name, u.getvalue(), u.type or "application/octet-stream")) for u in uploads]
        with st.spinner("Indexing..."):
            r = requests.post(f"{API_URL}/api/ingest/upload", files=files, timeout=600)
        if r.ok:
            data = r.json()
            st.success(f"Indexed {len(data['indexed_files'])} file(s), {data['chunks']} chunks.")
        else:
            st.error(r.text)

    st.divider()
    st.subheader("Maintenance")
    if st.button("Re-scan documents folder", use_container_width=True):
        with st.spinner("Scanning..."):
            r = requests.post(f"{API_URL}/api/ingest/scan", timeout=600)
        st.success(r.json()) if r.ok else st.error(r.text)

    if st.button("Clear collection", type="secondary", use_container_width=True):
        r = requests.delete(f"{API_URL}/api/ingest", timeout=30)
        st.success(r.json()) if r.ok else st.error(r.text)

    st.divider()
    top_k = st.slider("Top-K chunks", 1, 12, 4)


# --- Chat ---------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []  # list[{"role","content","sources","trace"}]


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for i, s in enumerate(sources, start=1):
            st.markdown(f"**[{i}] {s['document']}**  ·  score `{s['score']:.3f}`")
            st.code(s["chunk"][:1200] + ("..." if len(s["chunk"]) > 1200 else ""))


def render_trace(trace: list[dict[str, str]]) -> None:
    if not trace:
        return
    with st.expander("Agent trace", expanded=False):
        for step in trace:
            st.markdown(f"- **{step['name']}** — {step['detail']}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))
            render_trace(msg.get("trace", []))


prompt = st.chat_input("Ask a question about your documents...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
        if m["role"] in ("user", "assistant")
    ]
    payload = {"question": prompt, "top_k": top_k, "history": history}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(f"{API_URL}/api/chat", json=payload, timeout=600)
                r.raise_for_status()
                data = r.json()
            except Exception as e:  # noqa: BLE001
                st.error(f"Request failed: {e}")
                data = {"answer": "_Error contacting backend._", "sources": [], "trace": []}
        st.markdown(data["answer"])
        render_sources(data.get("sources", []))
        render_trace(data.get("trace", []))
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": data["answer"],
                "sources": data.get("sources", []),
                "trace": data.get("trace", []),
            }
        )
