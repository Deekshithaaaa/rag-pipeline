import chromadb
import json

def build_vector_store(chunks: list[dict], collection_name="rag_docs"):
    client = chromadb.PersistentClient(path="data/vectorstore")
    
    # Delete existing collection if it exists
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["content"] for c in chunks],
        metadatas=[{
            "source": c["source"],
            "chunk_index": c["chunk_index"]
        } for c in chunks]
    )
    
    print(f"Stored {len(chunks)} chunks in ChromaDB ✅")
    return collection

if __name__ == "__main__":
    with open("data/processed/embedded_chunks.json") as f:
        chunks = json.load(f)
    
    collection = build_vector_store(chunks)
    print(f"Vector store built successfully! ✅")