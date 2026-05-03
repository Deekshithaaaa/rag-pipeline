from openai import OpenAI
from dotenv import load_dotenv
from src.retrieval.retriever import retrieve, get_collection

load_dotenv()
client = OpenAI()

def build_prompt(query: str, context_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join([c["content"] for c in context_chunks])
    return f"""You are a helpful AI assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer this."

Context:
{context}

Question: {query}

Answer:"""

def query_rag(question: str, top_k=5) -> dict:
    collection = get_collection()
    chunks = retrieve(question, collection, top_k)
    prompt = build_prompt(question, chunks)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return {
        "answer": response.choices[0].message.content,
        "sources": list(set([c["source"] for c in chunks])),
        "chunks_used": len(chunks)
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
        print(f"🔢 Chunks used: {result['chunks_used']}")
        print("=" * 50)