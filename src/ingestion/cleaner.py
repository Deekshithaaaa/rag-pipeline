import re
import json

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)              # remove extra whitespace
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)   # remove non-ASCII
    text = re.sub(r'\n{3,}', '\n\n', text)        # collapse blank lines
    return text.strip()

def clean_all_documents(input_path: str, output_path: str):
    with open(input_path) as f:
        docs = json.load(f)
    
    for doc in docs:
        doc["content"] = clean_text(doc["content"])
    
    with open(output_path, "w") as f:
        json.dump(docs, f)
    
    print(f"Cleaned {len(docs)} documents ✅")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    clean_all_documents(
        "data/processed/raw_docs.json",
        "data/processed/cleaned_docs.json"
    )