from pathlib import Path
import fitz  # PyMuPDF
import json

def load_pdf(filepath: str) -> dict:
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return {
        "source": Path(filepath).name,
        "content": text,
        "type": "pdf"
    }

def load_all_documents(folder: str) -> list[dict]:
    docs = []
    for path in Path(folder).glob("*.pdf"):
        print(f"Loading: {path.name}")
        docs.append(load_pdf(str(path)))
    return docs

if __name__ == "__main__":
    docs = load_all_documents("data/raw/")
    print(f"\nTotal documents loaded: {len(docs)}")
    with open("data/processed/raw_docs.json", "w") as f:
        json.dump(docs, f)
    print("Saved to data/processed/raw_docs.json ✅")