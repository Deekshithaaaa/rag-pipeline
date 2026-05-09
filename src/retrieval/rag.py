from openai import OpenAI
from dotenv import load_dotenv
from src.retrieval.retriever import retrieve, get_collection

load_dotenv()
client = OpenAI()
collection = get_collection()

def build_prompt(query: str, context_chunks: list[dict]) -> str:
    context = "\n\n---\n\n".join([c["content"] for c in context_chunks])
    return f"""You are a precise research assistant answering questions about AI research papers.

STRICT RULES:
- Answer using ONLY the context provided below
- Do NOT use any outside knowledge or training data
- If the answer is not in the context, respond exactly: "I cannot find this in the provided papers."
- Be specific and cite which paper the information comes from
- Keep answers concise and factual

Context:
{context}

Question: {query}

Answer based strictly on the context above:"""

def query_rag(question: str, top_k=5) -> dict:
    chunks = retrieve(question, collection, top_k)
    prompt = build_prompt(question, chunks)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a strict research assistant. You ONLY answer from provided context. You NEVER use outside knowledge. If context doesn't contain the answer, say 'I cannot find this in the provided papers.' No exceptions."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )
    
    return {
        "answer": response.choices[0].message.content,
        "sources": list(set([c["source"] for c in chunks])),
        "chunks_used": len(chunks)
    }