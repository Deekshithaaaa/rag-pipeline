# 🧠 RAG Pipeline — LLM-Powered Document Q&A

> Ask natural language questions over AI research papers using Retrieval Augmented Generation (RAG)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Faithfulness](https://img.shields.io/badge/Faithfulness-97%25-brightgreen)

## 📌 What is this?

An end-to-end RAG pipeline that ingests PDF research papers, converts them into semantic vector embeddings, and answers natural language questions using GPT-4 — with source citations.

**Example:**
Question: "What is the attention mechanism?"
Answer: "The attention mechanism maps a query and key-value pairs
to an output using weighted sums..."
Source: 1706_03762.pdf (Attention Is All You Need)

## 🏗️ Architecture
📄 PDF Documents
↓
📝 Text Extraction (PyMuPDF)
↓
✂️  Chunking (LangChain - 500 chars, 50 overlap)
↓
🔢 Embeddings (OpenAI text-embedding-3-small)
↓
🗄️  Vector Store (ChromaDB - cosine similarity)
↓
🔍 Semantic Retrieval (top-k=5)
↓
💬 Answer Generation (GPT-4o-mini)
↓
🌐 FastAPI REST API

## 📊 Evaluation Results

| Metric | Score |
|--------|-------|
| Faithfulness | **97%** |
| Framework | RAGAs |
| Test Questions | 5 |

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| PDF Extraction | PyMuPDF |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | ChromaDB |
| LLM | GPT-4o-mini |
| API | FastAPI + Uvicorn |
| Evaluation | RAGAs |
| Containerization | Docker + Docker Compose |

## 📁 Project Structure
rag-pipeline/
├── data/
│   ├── raw/          # Original PDF papers
│   ├── processed/    # Cleaned text, chunks
│   └── vectorstore/  # ChromaDB persistent store
├── src/
│   ├── ingestion/    # PDF loading & cleaning
│   ├── embeddings/   # Embedding & vector store
│   ├── retrieval/    # RAG query engine
│   └── api/          # FastAPI backend
├── tests/            # RAGAs evaluation
├── Dockerfile
├── docker-compose.yml
└── requirements.txt

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- OpenAI API key
- Docker (optional)

### Local Setup

```bash
# Clone the repo
git clone https://github.com/Deekshithaaaa/rag-pipeline.git
cd rag-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Add your OpenAI API key to .env

# Download dataset
python download_papers.py

# Build vector store
python src/embeddings/embedder.py
python src/embeddings/vector_store.py

# Start API
uvicorn src.api.main:app --reload
```

### Docker Setup

```bash
docker-compose up --build
```

### API Usage

Visit `http://localhost:8000/docs` for interactive documentation.

**Query endpoint:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the attention mechanism?", "top_k": 5}'
```

**Response:**
```json
{
  "answer": "The attention mechanism...",
  "sources": ["1706_03762.pdf"],
  "chunks_used": 5
}
```

## 📚 Dataset

10 landmark AI research papers from ArXiv:
- Attention Is All You Need (Transformers)
- RAG — Retrieval Augmented Generation
- GPT-4 Technical Report
- LLaMA & LLaMA 2
- LoRA
- InstructGPT
- ReAct
- Chain of Thought Prompting
- LLM Survey

## 🔮 Future Improvements
- Add hybrid search (semantic + BM25 keyword)
- Implement re-ranking with cross-encoders
- Add streaming responses
- Support more document formats (Word, HTML)
- Build a Streamlit frontend

## 👩‍💻 Author
**Deekshitha Adishesha Raje Urs**  
[GitHub](https://github.com/Deekshithaaaa)