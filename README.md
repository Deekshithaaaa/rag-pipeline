# 🧠 RAG Pipeline — LLM-Powered Document Q&A

> Ask natural language questions over AI research papers using Retrieval Augmented Generation (RAG)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![Faithfulness](https://img.shields.io/badge/Faithfulness-79%25-brightgreen)
![HybridSearch](https://img.shields.io/badge/Search-Hybrid_BM25+Semantic-purple)

🌐 **Live API:** https://rag-pipeline-production-737f.up.railway.app/docs

🎨 **Live Demo UI:** https://rag-paper-assistant-2026.streamlit.app

---

## 📌 What is this?

An end-to-end RAG pipeline that ingests PDF research papers, converts them into semantic vector embeddings, and answers natural language questions using GPT-4 — with source citations.

**Example:**
```
Question: "What is the attention mechanism?"
Answer: "The attention mechanism maps a query and key-value pairs 
         to an output using weighted sums..."
Source: 1706_03762.pdf (Attention Is All You Need)
```

## 🏗️ Architecture

```
📄 PDF Documents
      ↓
📝 Text Extraction (PyMuPDF)
      ↓
✂️  Chunking (LangChain - 800 chars, 150 overlap)
      ↓
🔢 Embeddings (OpenAI text-embedding-3-small)
      ↓
🗄️  Vector Store (ChromaDB - cosine similarity)
      ↓
🔀 Hybrid Search (BM25 + Semantic + RRF)
      ↓
💬 Answer Generation (GPT-4o-mini)
      ↓
🌐 FastAPI REST API
```

## 📊 Evaluation Results

| Metric | Score |
|--------|-------|
| Faithfulness | **79%** |
| Answer Relevancy | **71%** |
| Framework | RAGAs |
| Test Questions | 25 |

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| PDF Extraction | PyMuPDF |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | ChromaDB |
| Retrieval | Hybrid Search (BM25 + ChromaDB + RRF) |
| LLM | GPT-4o-mini |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Evaluation | RAGAs |
| Containerization | Docker + Docker Compose |

## 📁 Project Structure

```
rag-pipeline/
├── data/
│   ├── raw/              # Original PDF papers
│   ├── processed/        # Cleaned text, chunks
│   └── vectorstore/      # ChromaDB persistent store
├── src/
│   ├── ingestion/        # PDF loading & cleaning
│   ├── embeddings/       # Embedding & vector store
│   ├── retrieval/        # RAG + Hybrid search engine
│   └── api/              # FastAPI backend
├── tests/                # RAGAs evaluation
├── app.py                # Streamlit frontend
├── startup.py            # Auto-setup on deployment
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

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

### Streamlit UI

```bash
python -m streamlit run app.py
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
- LLaMA
- LLaMA 2
- LoRA
- InstructGPT
- ReAct
- Chain of Thought Prompting
- A Survey of Large Language Models (Zhao et al., 2023)

## 🔍 Hybrid Search

This pipeline uses **Hybrid Search** combining:
- **Semantic Search** — finds conceptually similar content using embeddings
- **BM25 Keyword Search** — finds exact keyword matches
- **Reciprocal Rank Fusion (RRF)** — merges both result sets intelligently

This gives better retrieval than pure semantic search alone.

## 📈 Evaluation Journey

| Version | Faithfulness | What Changed |
|---|---|---|
| v1 (5 questions) | 97% | Small sample |
| v2 (25 questions) | 80% | More questions |
| v3 (hybrid search) | 58% | Hybrid added |
| v4 (tuned RRF) | 63% | Better weights |
| v5 (bigger chunks) | 67% | Chunk size 800 |
| v6 (strict prompt) | **79%** | System message |

## 🔮 Future Improvements
- Implement re-ranking with cross-encoders
- Add streaming responses
- Support more document formats (Word, HTML)
- Add conversation memory for multi-turn Q&A

## 👩‍💻 Author
**Deekshitha Adishesha Raje Urs**
[GitHub](https://github.com/Deekshithaaaa) | [LinkedIn](#)