"""Document loading + chunking + indexing."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

from .config import settings
from .vectorstore import add_chunks


# --- Loaders ------------------------------------------------------------------

def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _load_docx(path: Path) -> str:
    import docx  # python-docx

    d = docx.Document(str(path))
    return "\n".join(p.text for p in d.paragraphs)


def _load_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    return soup.get_text("\n")


LOADERS = {
    ".txt": _load_txt,
    ".md": _load_txt,
    ".markdown": _load_txt,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".html": _load_html,
    ".htm": _load_html,
}


def load_document(path: Path) -> str:
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"Unsupported file type: {path.suffix}")
    return loader(path)


# --- Chunking -----------------------------------------------------------------
#
# Recursive, structure-aware splitter:
#   paragraphs  ->  sentences  ->  words  ->  characters
# We never cut a unit in half unless the unit itself is bigger than chunk_size.
# Adjacent chunks share a sentence-aligned overlap so cross-paragraph context
# is preserved without duplicating arbitrary character slices.

# Order matters: try paragraph breaks first, then line breaks, then sentence
# terminators, then whitespace, then characters as the last resort.
_SEPARATORS: list[str] = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
_PARAGRAPH_RE = re.compile(r"\n\s*\n+")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÉÈÀÂÊÎÔÛÇ0-9])")


def _split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]


def _split_sentences(paragraph: str) -> list[str]:
    parts = _SENTENCE_RE.split(paragraph.strip())
    return [s.strip() for s in parts if s.strip()]


def _recursive_split(text: str, chunk_size: int, seps: list[str]) -> list[str]:
    """Langchain-style recursive splitter that respects natural boundaries."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text] if text else []

    sep = seps[0] if seps else ""
    if sep == "":
        # Last resort: hard char split.
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep) if sep else [text]
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(part) <= chunk_size:
            out.append(part)
        else:
            out.extend(_recursive_split(part, chunk_size, seps[1:]))
    return out


def _pack(units: list[str], chunk_size: int, joiner: str) -> list[str]:
    """Greedily pack atomic units into chunks <= chunk_size."""
    chunks: list[str] = []
    buf = ""
    for u in units:
        if not buf:
            buf = u
        elif len(buf) + len(joiner) + len(u) <= chunk_size:
            buf = buf + joiner + u
        else:
            chunks.append(buf)
            buf = u
    if buf:
        chunks.append(buf)
    return chunks


def _sentence_aligned_overlap(prev_chunk: str, overlap: int) -> str:
    """Take the last sentences of the previous chunk, up to ~overlap chars."""
    if overlap <= 0 or not prev_chunk:
        return ""
    sents = _split_sentences(prev_chunk) or [prev_chunk]
    out = ""
    for s in reversed(sents):
        candidate = (s + " " + out).strip() if out else s
        if len(candidate) > overlap and out:
            break
        out = candidate
        if len(out) >= overlap:
            break
    return out


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Paragraph-aware chunking with sentence-aligned overlap."""
    text = (text or "").strip()
    if not text:
        return []

    # 1) Split into paragraphs. Each oversized paragraph is recursively split
    #    on sentence -> word -> char boundaries, never mid-word when avoidable.
    atomic: list[str] = []
    for para in _split_paragraphs(text):
        if len(para) <= chunk_size:
            atomic.append(para)
        else:
            atomic.extend(_recursive_split(para, chunk_size, _SEPARATORS))

    # 2) Pack atomic units into chunks, separating paragraphs with a blank line.
    chunks = _pack(atomic, chunk_size, joiner="\n\n")

    # 3) Prepend a sentence-aligned overlap from the previous chunk.
    if overlap > 0 and len(chunks) > 1:
        merged: list[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = _sentence_aligned_overlap(chunks[i - 1], overlap)
            merged.append(((tail + "\n\n") if tail else "") + chunks[i])
        chunks = merged

    return chunks


# --- Indexing -----------------------------------------------------------------

def _doc_id(source: str, idx: int, content: str) -> str:
    h = hashlib.sha1(f"{source}:{idx}:{content[:64]}".encode()).hexdigest()[:16]
    return f"{Path(source).stem}-{idx}-{h}"


def index_paths(paths: Iterable[Path]) -> tuple[list[str], int]:
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []
    indexed: list[str] = []

    for path in paths:
        if not path.is_file() or path.suffix.lower() not in LOADERS:
            continue
        try:
            text = load_document(path)
        except Exception as e:  # noqa: BLE001
            print(f"[ingest] skip {path.name}: {e}")
            continue
        chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
        for i, c in enumerate(chunks):
            ids.append(_doc_id(path.name, i, c))
            docs.append(c)
            metas.append({"source": path.name, "path": str(path), "chunk_index": i})
        if chunks:
            indexed.append(path.name)

    add_chunks(ids, docs, metas)
    return indexed, len(ids)


def index_directory(directory: str | Path | None = None) -> tuple[list[str], int]:
    base = Path(directory or settings.documents_dir)
    files = [p for p in base.rglob("*") if p.is_file()]
    return index_paths(files)
