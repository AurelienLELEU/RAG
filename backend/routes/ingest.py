import shutil
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from ..config import settings
from ..ingestion import index_directory, index_paths
from ..models import CollectionInfo, IngestResponse
from ..vectorstore import collection_count, reset_collection

router = APIRouter(tags=["ingest"])


@router.post("/ingest/scan", response_model=IngestResponse)
def ingest_scan() -> IngestResponse:
    """Re-index every file in the configured documents directory."""
    files, chunks = index_directory()
    return IngestResponse(indexed_files=files, chunks=chunks)


@router.post("/ingest/upload", response_model=IngestResponse)
async def ingest_upload(files: list[UploadFile] = File(...)) -> IngestResponse:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    saved: list[Path] = []
    docs_dir = Path(settings.documents_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        dest = docs_dir / Path(f.filename).name
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(dest)
    indexed, chunks = index_paths(saved)
    return IngestResponse(indexed_files=indexed, chunks=chunks)


@router.delete("/ingest", response_model=CollectionInfo)
def clear_collection() -> CollectionInfo:
    reset_collection()
    return CollectionInfo(name=settings.collection_name, count=collection_count())


@router.get("/ingest/info", response_model=CollectionInfo)
def info() -> CollectionInfo:
    return CollectionInfo(name=settings.collection_name, count=collection_count())
