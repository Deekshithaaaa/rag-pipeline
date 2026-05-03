from openai import OpenAI
from dotenv import load_dotenv
import json
import os

load_dotenv()
client = OpenAI()

def embed_chunks(chunks: list[dict], batch_size=100) -> list[dict]:
    embedded = []
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [c["content"] for c in batch]
        
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        
        for j, item in enumerate(response.data):
            chunk = batch[j].copy()
            chunk["embedding"] = item.embedding
            embedded.append(chunk)
        
        batch_num = i // batch_size + 1
        print(f"Embedded batch {batch_num}/{total_batches} ✅")
    
    return embedded

if __name__ == "__main__":
    with open("data/processed/chunks.json") as f:
        chunks = json.load(f)
    
    print(f"Embedding {len(chunks)} chunks...")
    embedded = embed_chunks(chunks)
    
    with open("data/processed/embedded_chunks.json", "w") as f:
        json.dump(embedded, f)
    
    print(f"\nTotal embedded: {len(embedded)} chunks")
    print("Saved to data/processed/embedded_chunks.json ✅")