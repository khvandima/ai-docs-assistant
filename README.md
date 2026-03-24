# Course RAG Agent

An intelligent RAG (Retrieval-Augmented Generation) agent for answering questions based on course materials. Built with LangGraph, Qdrant, and local LLMs via Ollama.

## Architecture

```
User Question
     ↓
  Router (LLM)
     ├── chat → Direct Answer
     └── rag  → Hybrid Search (Dense + BM25)
                     ↓
               Reranking (CrossEncoder)
                     ↓
               Grade (LLM)
                     ├── relevant   → Generate Answer
                     └── irrelevant → No Answer
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Agent framework | LangGraph |
| LLM | Ollama (local) / Groq / OpenAI / Anthropic |
| Vector database | Qdrant |
| Embeddings | multilingual-e5-large (Ollama) |
| Reranking | CrossEncoder (sentence-transformers) |
| Hybrid search | Dense + BM25 via Qdrant |
| API | FastAPI |
| Tracing | LangSmith |

## Features

- **Hybrid search** — combines semantic (dense) and keyword (BM25) search with RRF fusion
- **Reranking** — CrossEncoder reranks retrieved chunks for better relevance
- **Multi-provider LLM** — switch between Ollama, OpenAI, Anthropic, Groq via `.env`
- **Conversation memory** — maintains chat history within a session
- **Auto-indexing** — automatically indexes files from `uploads/` on startup
- **LangSmith tracing** — full observability of agent execution
- **Health checks** — `/health` and `/health/detailed` endpoints

## Quick Start

### Prerequisites

- Python 3.12+
- Docker
- Ollama (for local LLM)

### 1. Clone and configure

```bash
git clone https://github.com/yourusername/rag-agent.git
cd rag-agent
cp .env.example .env
# Edit .env with your settings
```

### 2. Start Qdrant

```bash
docker run -d --name qdrant -p 6333:6333 qdrant/qdrant
```

### 3. Pull embedding model

```bash
ollama pull jeffh/intfloat-multilingual-e5-large:q8_0
```

### 4. Install dependencies and run

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in your browser.

### Docker Compose (recommended)

```bash
docker-compose up --build
```

## Configuration

All settings are managed via `.env`. Key options:

```bash
# Switch LLM provider
PROVIDER=ollama        # or: openai, anthropic, groq
MODEL=my-model

# Tune retrieval
SCORE_THRESHOLD=0.60   # minimum relevance score
RERANK_TOP_K=3         # chunks to keep after reranking
CHUNK_SIZE=512         # chunk size in characters
```

## Usage

1. Open the web UI at `http://localhost:8000`
2. Go to **Files** tab and upload your `.md` course files
3. Go to **Chat** tab and ask questions

## Running Tests

```bash
# Unit tests (no external dependencies)
pytest tests/ -v

# Integration tests (require running Qdrant and Ollama)
pytest tests/ -v --integration
```

## RAG Evaluation Results

Evaluated on 6 questions from real course materials using [Ragas](https://ragas.io):

| Metric | Score | Description |
|--------|-------|-------------|
| **Faithfulness** | 0.854 | Answer grounded in retrieved context |
| **Answer Relevancy** | 0.907 | Answer addresses the question |
| **Context Recall** | 0.917 | Relevant chunks retrieved |

> Evaluated with `llama-3.1-8b-instant` via Groq API. Run `python -m tests.eval` to reproduce.

## RAG Evaluation

```bash
python -m tests.eval
```

Evaluates the pipeline using Ragas metrics:
- **Faithfulness** — answer grounded in context
- **Answer relevancy** — answer addresses the question
- **Context precision** — relevant chunks ranked first
- **Context recall** — all relevant chunks retrieved

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI |
| POST | `/ask` | Ask a question |
| POST | `/upload` | Upload MD file |
| GET | `/files` | List uploaded files |
| DELETE | `/files/{filename}` | Delete file |
| POST | `/files/{filename}/reindex` | Reindex file |
| GET | `/health` | Quick health check |
| GET | `/health/detailed` | Detailed component status |

## Project Structure

```
├── app/
│   ├── agent.py          # LangGraph graph
│   ├── ingestion.py      # MD file indexing pipeline
│   ├── embeddings.py     # Embedding model
│   ├── llm_factory.py    # Multi-provider LLM factory
│   ├── reranker.py       # CrossEncoder reranking
│   ├── retry.py          # Retry logic
│   ├── config.py         # Settings via pydantic-settings
│   ├── logger.py         # Structured logging
│   ├── main.py           # FastAPI application
│   └── static/
│       └── index.html    # Web UI
├── tests/
│   ├── eval.py           # Ragas evaluation pipeline
│   ├── test_agent.py
│   ├── test_ingestion.py
│   └── test_main.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── requirements.txt
```

## License

MIT