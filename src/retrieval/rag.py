from openai import OpenAI
from dotenv import load_dotenv
import chromadb
import json
from src.retrieval.hybrid_retriever import (
    hybrid_retrieve, build_bm25_index, load_chunks
)

load_dotenv()
client = OpenAI()

# Initialize once
db = chromadb.PersistentClient(path="data/vectorstore")
collection = db.get_collection("rag_docs")
chunks = load_chunks()
bm25 = build_bm25_index(chunks)

def build_prompt(query: str, context_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join([c["content"] for c in context_chunks])
    return f"""You are a helpful AI assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {query}

Answer:"""

def query_rag(question: str, top_k=5) -> dict:
    chunks_result = hybrid_retrieve(question, collection, chunks, bm25, top_k)
    prompt = build_prompt(question, chunks_result)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return {
        "answer": response.choices[0].message.content,
        "sources": list(set([c["source"] for c in chunks_result])),
        "chunks_used": len(chunks_result)
    }

if __name__ == "__main__":
    questions = [
        "What is the attention mechanism?",
        "What is LoRA and how does it work?",
        "What is chain of thought prompting?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        print("-" * 50)
        result = query_rag(question)
        print(f"💬 Answer: {result['answer']}")
        print(f"📄 Sources: {result['sources']}")
        print("=" * 50)