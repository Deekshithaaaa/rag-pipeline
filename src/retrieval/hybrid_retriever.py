import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

def load_chunks(chunks_path="data/processed/chunks.json") -> list[dict]:
    with open(chunks_path) as f:
        return json.load(f)

def build_bm25_index(chunks: list[dict]):
    tokenized = [chunk["content"].lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)

def semantic_search(query: str, collection, top_k=10) -> list[dict]:
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return [
        {
            "chunk_id": results["ids"][0][i],
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "score": 1 - results["distances"][0][i]
        }
        for i in range(len(results["documents"][0]))
    ]

def bm25_search(query: str, chunks: list[dict], bm25, top_k=10) -> list[dict]:
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [
        {
            "chunk_id": chunks[i]["chunk_id"],
            "content": chunks[i]["content"],
            "source": chunks[i]["source"],
            "score": float(scores[i])
        }
        for i in top_indices
    ]

def reciprocal_rank_fusion(semantic_results: list[dict], bm25_results: list[dict], k=20, semantic_weight=0.8, bm25_weight=0.2) -> list[dict]:
    scores = {}
    docs = {}

    for rank, doc in enumerate(semantic_results):
        cid = doc["chunk_id"]
        scores[cid] = scores.get(cid, 0) + semantic_weight * (1 / (k + rank + 1))
        docs[cid] = doc

    for rank, doc in enumerate(bm25_results):
        cid = doc["chunk_id"]
        scores[cid] = scores.get(cid, 0) + bm25_weight * (1 / (k + rank + 1))
        docs[cid] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [docs[cid] for cid, _ in ranked]

def hybrid_retrieve(query: str, collection, chunks: list[dict], bm25, top_k=8) -> list[dict]:
    semantic_results = semantic_search(query, collection, top_k=10)
    bm25_results = bm25_search(query, chunks, bm25, top_k=10)
    fused = reciprocal_rank_fusion(semantic_results, bm25_results)
    return fused[:top_k]

if __name__ == "__main__":
    db = chromadb.PersistentClient(path="data/vectorstore")
    collection = db.get_collection("rag_docs")
    chunks = load_chunks()
    bm25 = build_bm25_index(chunks)

    query = "What is the attention mechanism?"
    print(f"Query: {query}\n")

    print("=== SEMANTIC ONLY ===")
    from src.retrieval.retriever import retrieve
    semantic = retrieve(query, collection, top_k=5)
    for i, r in enumerate(semantic):
        print(f"{i+1}. {r['source']}: {r['content'][:100]}...")

    print("\n=== HYBRID (BM25 + Semantic) ===")
    hybrid = hybrid_retrieve(query, collection, chunks, bm25, top_k=5)
    for i, r in enumerate(hybrid):
        print(f"{i+1}. {r['source']}: {r['content'][:100]}...")