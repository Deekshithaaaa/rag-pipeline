from langchain_text_splitters import RecursiveCharacterTextSplitter
import json

def chunk_documents(docs: list[dict], chunk_size=800, chunk_overlap=150) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["content"])
        for i, chunk in enumerate(splits):
            chunks.append({
                "chunk_id": f"{doc['source']}_chunk_{i}",
                "source": doc["source"],
                "content": chunk,
                "chunk_index": i
            })
        print(f"Document: {doc['source']} → {len(splits)} chunks")
    
    return chunks

if __name__ == "__main__":
    with open("data/processed/cleaned_docs.json") as f:
        docs = json.load(f)
    
    chunks = chunk_documents(docs)
    
    print(f"\nTotal chunks created: {len(chunks)}")
    print(f"Average chunks per document: {len(chunks) // len(docs)}")
    
    with open("data/processed/chunks.json", "w") as f:
        json.dump(chunks, f)
    
    print("Saved to data/processed/chunks.json ✅")