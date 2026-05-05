import os
from pathlib import Path

def setup():
    vectorstore_path = Path("data/vectorstore")
    chunks_path = Path("data/processed/chunks.json")
    
    if not vectorstore_path.exists() or not any(vectorstore_path.iterdir()):
        print("Vector store not found. Building...")
        if chunks_path.exists():
            from src.embeddings.embedder import embed_chunks
            from src.embeddings.vector_store import build_vector_store
            import json
            with open(chunks_path) as f:
                chunks = json.load(f)
            embedded = embed_chunks(chunks)
            build_vector_store(embedded)
            print("Vector store built! ✅")
        else:
            print("No chunks found. Running full pipeline...")
            from src.ingestion.loader import load_all_documents
            from src.ingestion.cleaner import clean_all_documents
            from src.ingestion.chunker import chunk_documents
            from src.embeddings.embedder import embed_chunks
            from src.embeddings.vector_store import build_vector_store
            import json
            os.makedirs("data/raw", exist_ok=True)
            os.makedirs("data/processed", exist_ok=True)
            docs = load_all_documents("data/raw/")
            with open("data/processed/raw_docs.json", "w") as f:
                json.dump(docs, f)
            clean_all_documents(
                "data/processed/raw_docs.json",
                "data/processed/cleaned_docs.json"
            )
            with open("data/processed/cleaned_docs.json") as f:
                docs = json.load(f)
            chunks = chunk_documents(docs)
            with open("data/processed/chunks.json", "w") as f:
                json.dump(chunks, f)
            embedded = embed_chunks(chunks)
            build_vector_store(embedded)
            print("Full pipeline complete! ✅")
    else:
        print("Vector store found ✅")

if __name__ == "__main__":
    setup()