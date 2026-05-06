import os
from pathlib import Path

def setup():
    vectorstore_path = Path("data/vectorstore")
    if vectorstore_path.exists() and any(vectorstore_path.iterdir()):
        print("Vector store found ✅ Skipping rebuild.")
    else:
        print("Vector store not found. Building...")
        from src.embeddings.embedder import embed_chunks
        from src.embeddings.vector_store import build_vector_store
        import json
        with open("data/processed/chunks.json") as f:
            chunks = json.load(f)
        embedded = embed_chunks(chunks)
        build_vector_store(embedded)
        print("Vector store built! ✅")

if __name__ == "__main__":
    setup()