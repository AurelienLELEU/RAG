"""CLI: index every supported file under ./data/documents (or a custom path)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a script from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.ingestion import index_directory  # noqa: E402
from backend.vectorstore import collection_count, reset_collection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Index documents into ChromaDB.")
    parser.add_argument("--path", default=None, help="Directory to scan (default: data/documents).")
    parser.add_argument("--reset", action="store_true", help="Wipe the collection before indexing.")
    args = parser.parse_args()

    if args.reset:
        print("Resetting collection...")
        reset_collection()

    files, chunks = index_directory(args.path)
    print(f"Indexed {len(files)} files, {chunks} chunks. Total in collection: {collection_count()}")
    for f in files:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
