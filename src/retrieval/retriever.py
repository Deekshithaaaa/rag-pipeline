import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def get_collection(collection_name="rag_docs"):
    db = chromadb.PersistentClient(path="data/vectorstore")
    return db.get_collection(collection_name)

def retrieve(query: str, collection, top_k=5) -> list[dict]:
    # Convert question to embedding
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    # Search ChromaDB for similar chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i]
        })
    
    return chunks

if __name__ == "__main__":
    collection = get_collection()
    results = retrieve("What is the attention mechanism?", collection)
    print(f"Found {len(results)} relevant chunks:\n")
    for i, r in enumerate(results):
        print(f"Chunk {i+1} (from {r['source']}):")
        print(f"{r['content'][:200]}...")
        print()