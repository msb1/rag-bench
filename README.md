# rag-bench

A comprehensive evaluation framework for Retrieval-Augmented Generation (RAG) systems using the Open RAG Benchmark dataset. This project provides end-to-end tools for building, evaluating, and analyzing RAG pipelines with multimodal PDF-based documents.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Configuration](#configuration)
- [Available Scripts](#available-scripts)
- [Results & Output Files](#results--output-files)
- [Metrics Explained](#metrics-explained)
- [Usage Examples](#usage-examples)
- [Environment Variables](#environment-variables)
- [License](#license)

## Overview

rag-bench is designed to evaluate RAG systems using the Open RAG Benchmark, a dataset of 1000 arXiv papers with 3045 question-answer pairs. The framework implements:

- **Document Processing**: PDF-to-markdown conversion and intelligent chunking
- **Vector Store Integration**: Qdrant-based hybrid retrieval (dense + sparse embeddings)
- **RAG Fusion Pipeline**: Multi-query retrieval with reranking
- **Evaluation Metrics**: RAGAS-style metrics computed via LLM-as-a-judge approach
- **Analysis Dashboard**: Visualization and failure analysis tools

## Project Structure

```
rag-bench/
├── src/                          # Main source code
│   ├── __init__.py
│   ├── analyze.py               # Results visualization and failure analysis
│   ├── chunker.py               # Document chunking and indexing
│   ├── convert.py               # PDF to Markdown conversion
│   ├── eval.py                  # Row-by-row evaluation script
│   ├── evaluate.py              # Core metric functions (Faithfulness, Relevancy, etc.)
│   ├── migrate.py               # S3 bucket migration utility
│   ├── rag.py                   # Complete RAG pipeline implementation
│   ├── models/
│   │   └── reranker.py          # FastAPI reranker server for jina-reranker-v3.5
│   └── utils/
│       ├── __init__.py
│       ├── clients.py           # S3, OpenAI, and embedding client factories
│       ├── helper.py            # S3 utility functions
│       └── qdrt.py              # Qdrant client factory
│
├── openrag/                      # Open RAG Benchmark dataset
│   ├── README.md                # Dataset documentation
│   ├── answers.json             # Ground truth answers (3045 QA pairs)
│   ├── pdf_urls.json            # arXiv PDF URLs
│   ├── qrels.json               # Query-relevance labels
│   └── queries.json             # Evaluation queries (3045 queries)
│
├── results/                      # Evaluation outputs
│   ├── output.jsonl             # Generated answers with metrics
│   ├── results.jsonl            # Raw RAG pipeline outputs
│   ├── hallucinations.jsonl     # Filtered: severe hallucination failures
│   └── retrieval_misses.jsonl   # Filtered: retrieval failure cases
│
├── .env                         # Environment configuration
├── pyproject.toml               # Python dependencies
├── progress.json                # Processing checkpoint
└── LICENSE                      # MIT License
```

## Prerequisites

- Python 3.14+
- Qdrant Vector Database
- LM Studio with compatible models
- MinIO/S3-compatible object storage (optional, for PDF retrieval)

### Required Models

The framework requires several local LLMs:

| Model | Purpose | Endpoint Config |
|-------|---------|-----------------|
| `text-embedding-embeddinggemma-300m` | Embedding generation | `OPENAI_LOCAL_ENDPOINT` |
| `qwen/qwen3-4b-2507` | Query generation (RAG Fusion) | `OPENAI_REMOTE_ENDPOINT` |
| `qwen2.5-7b-instruct-mlx` | Answer generation | `OPENAI_LOCAL_ENDPOINT` |
| `meta-llama-3.1-8b-instruct` | Evaluation (Faithfulness, Relevancy, etc.) | `OPENAI_LOCAL_ENDPOINT` |

### Dependencies

```bash
pip install -e .
# or using uv
uv pip install -e .
```

Key dependencies:
- `langchain`, `langchain-openai`, `langchain-qdrant` - RAG framework
- `fastembed` - Fast embedding models
- `qdrant-client` - Vector database
- `pymupdf4llm` - PDF to Markdown conversion
- `ragas` - Evaluation metrics foundation
- `matplotlib`, `seaborn` - Visualization

## Configuration

### Environment Setup

1. Copy `.env` template and configure endpoints:

```bash
# Ensure LM Studio is running with your models
# Configure S3 endpoints if using MinIO
```

2. Start required services:

```bash
# Qdrant (standalone or Docker)
docker run -p 6333:6333 qdrant/qdrant

# LM Studio - Load models and enable local API
# Tools → Server → Start server

# FastAPI reranker (optional, for jina-reranker-v3.5)
uvicorn models.reranker:app --host 0.0.0.0 --port 8000
```

## Available Scripts

### 1. PDF Conversion (`convert.py`)

Converts PDFs from S3 to Markdown format:

```bash
python -m src.convert
```

**Purpose**: Transforms PDF documents into markdown for easier chunking and retrieval.

### 2. Document Chunking (`chunker.py`)

Chunks markdown documents and indexes them in Qdrant:

```bash
python -m src.chunker
```

**Features**:
- Recursive character splitting (512 tokens, 51 overlap)
- Markdown header-aware splitting
- Table extraction and preservation
- BM25 + dense hybrid retrieval

### 3. RAG Pipeline (`rag.py`)

Executes the complete RAG pipeline:

```bash
python -m src.rag
```

**Pipeline Steps**:
1. Generates query variations using LLM
2. Performs multi-query retrieval (k=12)
3. Deduplicates and reranks results (top 5)
4. Generates final answer using retrieved context

**Output**: Writes to `results/results.jsonl` with question, answer, contexts, and ground truth.

### 4. Evaluation (`eval.py`)

Evaluates outputs using RAGAS-style metrics:

```bash
python -m src.eval
```

**Metrics Computed**:
- `faithfulness` - Answer grounded in context
- `answer_relevancy` - Answer addresses question
- `context_recall` - Context contains ground truth
- `context_precision` - Context is relevant to question
- `answer_correctness` - F1 score of factual match

**Output**: Writes to `results/output.jsonl` with all metrics.

### 5. Analysis (`analyze.py`)

Visualizes results and identifies failures:

```bash
python -m src.analyze
```

**Generates**:
- `rag_metrics_dashboard.png` - Performance visualization
- `hallucinations.jsonl` - Filtered faithfulness < 0.30 failures
- `retrieval_misses.jsonl` - Filtered context_recall < 0.30 failures

### 6. S3 Migration (`migrate.py`)

Migrates data between S3-compatible stores:

```bash
python -m src.migrate
```

## Results & Output Files

| File | Description |
|------|-------------|
| `results/results.jsonl` | Raw RAG pipeline outputs (question, answer, contexts, ground_truth) |
| `results/output.jsonl` | Evaluation results with 5 metrics per record |
| `results/hallucinations.jsonl` | Rows where answer isn't grounded in context (faithfulness < 0.30) |
| `results/retrieval_misses.jsonl` | Rows where retriever missed key facts (context_recall < 0.30) |

Each line is a JSON object. Example output record:

```json
{
  "question": "What are the challenges in estimating output impedance?",
  "answer": "The main challenges include...",
  "contexts": ["Chunk 1: Content...", "Chunk 2: Content..."],
  "ground_truth": "Estimating output impedance is challenging due to..."
}
```

## Metrics Explained

| Metric | Description | Range | Good Score |
|--------|-------------|-------|------------|
| **Faithfulness** | Generated answer sticks to retrieved context | 0.0 - 1.0 | ≥ 0.70 |
| **Answer Relevancy** | Answer addresses the question | 0.0 - 1.0 | ≥ 0.70 |
| **Context Recall** | Context contains all ground truth facts | 0.0 - 1.0 | ≥ 0.70 |
| **Context Precision** | Retrieved context is relevant | 0.0 - 1.0 | ≥ 0.70 |
| **Answer Correctness** | Factual match with ground truth (TP/(TP+0.5×(FP+FN))) | 0.0 - 1.0 | ≥ 0.70 |

**Implementation Note**: All metrics use LLM-as-a-judge approach via local OpenAI-compatible API endpoints.

## Usage Examples

### End-to-End Evaluation

```bash
# 1. Process PDFs (if not already converted)
python -m src.convert

# 2. Chunk and index documents
python -m src.chunker

# 3. Run RAG pipeline
python -m src.rag

# 4. Evaluate results
python -m src.eval

# 5. Generate analysis
python -m src.analyze
```

### Resume Partial Runs

The framework supports resuming from checkpoints via `progress.json`:

```bash
# After interruption, re-running rag.py will continue from last processed query
python -m src.rag
```

### Custom Chunking Parameters

Modify `src/chunker.py` to customize chunking:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,      # Adjust for your context window
    chunk_overlap=51,    # Overlap for context continuity
)
```

### Adjust Evaluation Thresholds

Modify `src/analyze.py` to change failure thresholds:

```python
FAITHFULNESS_THRESHOLD = 0.30      # Lower = stricter
CONTEXT_RECALL_THRESHOLD = 0.30    # Lower = stricter
```

## Environment Variables

Set in `.env` file:

| Variable | Description | Example |
|----------|-------------|---------|
| `S3_ENDPOINT_URL` | S3-compatible storage endpoint | `--------------` |
| `S3_BUCKET` | Bucket name | `--------------` |
| `S3_ACCESS_KEY` | Access key | `--------------` |
| `S3_SECRET_KEY` | Secret key | `--------------` |
| `PDF_URLS_FILE` | Path to PDF URLs JSON | `.../pdf_urls.json` |
| `QUERIES_FILE` | Path to queries JSON | `.../queries.json` |
| `ANSWERS_FILE` | Path to answers JSON | `.../answers.json` |
| `DATASET_FILE` | Output JSONL path | `.../results.jsonl` |
| `OUTPUT_FILE` | Evaluation output path | `.../output.jsonl` |
| `EMBEDDING_MODEL` | Embedding model name | `text-embedding-embeddinggemma-300m` |
| `CODING_MODEL` | Query generation model | `qwen/qwen3-4b-2507` |
| `RAG_MODEL` | Answer generation model | `qwen2.5-7b-instruct` |
| `EVAL_MODEL` | Evaluation model | `meta-llama-3.1-8b-instruct` |
| `QDRANT_COLLECTION` | Vector collection name | `--------------` |
| `OPENAI_LOCAL_ENDPOINT` | Local LLM endpoint | `--------------` |
| `OPENAI_REMOTE_ENDPOINT` | Remote LLM endpoint | `--------------` |

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

**Note**: This project relies on local model serving via LM Studio. Ensure all required models are downloaded and the server is running before executing scripts.
