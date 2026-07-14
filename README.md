# RAG Against the Machine 🤖🔍

A high-performance, source-grounded **Retrieval-Augmented Generation (RAG)** pipeline built from scratch to query complex codebases. This system utilizes a custom tokenized **BM25s** search engine optimized for code syntax extraction, paired with a locally hosted **Qwen 0.6B LLM** to generate highly precise, factual, and hallucination-free technical answers.

---

## 🚀 Performance Metrics
Evaluated natively via automated integration test suites (`moulinette`), the retrieval core successfully surpasses standard baseline thresholds:

* **Recall@1:** `65.0%` (High-precision single-shot match)
* **Recall@5:** `81.0%` (Exceeds mandatory 80% baseline threshold)
* **Recall@10:** `87.0%`
* **Cold Start Latency:** `< 5 seconds` (With cached model parameters)

---

## 🛠️ System Architecture

1. **Codebase Chunker (`index.py`)**: Traverses source directories recursively to slice `.py` and `.md` files. It employs custom regular expression boundaries (`def`, `class`, and `\n\n`) to retain code-block integrity, tracking precise character coordinates (`first_character_index` to `last_character_index`).
2. **Retrieval Core (`bm25s`)**: Tokenizes the raw text corpus, strips English stopwords, compiles a highly sparse keyword inverted index, and flushes persistent binary indices to disk alongside relative POSIX metadata profiles.
3. **Local Context Generator (`answer_dataset.py`)**: Coordinates cross-platform path mapping, reconstructs text fragments dynamically from disk slices to preserve RAM footprint, compiles structured ChatML conversation role templates, and runs inference using local weights.

---

## 📁 Directory Structure

```text
rag-agains-machine/
├── data/
│   ├── raw/vllm-0.10.1/          # Target codebase repository chunks
│   └── processed/bm25_index/     # Persistent BM25 lookup vectors & metadata
├── student/
│   ├── __init__.py
│   ├── __main__.py               # CLI entrypoint exposure layer (Python Fire)
│   ├── index.py                  # Chunking core and keyword compiler logic
│   ├── answer_dataset.py         # Batch LLM inference and context processor
│   └── models.py                 # Structured Pydantic validation contracts
├── pyproject.toml                # Project configurations managed by uv
└── README.md
```

## 🔧 Installation & Environment Setup

This project uses **`uv`**, a fast Python package installer and resolver written in Rust.

### 1. Prerequisites
If you are running on **Windows**, it is highly recommended to clone this repository inside **WSL (Windows Subsystem for Linux - Ubuntu)** to eliminate filesystem mapping overhead and ensure optimal `drvfs` indexing speeds.

### 2. Clone and Synchronize Workspace
```bash
# Clone the repository
git clone [https://github.com/qirall79/rag-agains-machine.git](https://github.com/qirall79/rag-agains-machine.git)
cd rag-agains-machine

# Create a virtual environment and sync locked dependencies
uv venv
uv sync
```

### 3. Clone and Synchronize Workspace
Force the initial download of the Qwen weights to satisfy cold start pipeline time constraints:
```
uv run python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)"
```

## 💻 Command Line Interface (CLI) Execution

The system exposes unified pipeline methods managed via the `python-fire` wrapper ecosystem.

### Phase 1: Build the Vector Indexes
Chunk the codebase text and compile your persistent local inverted lookup schemas:
```bash
uv run python -m student index --max_chunk_size 2000
```

### Phase 2: Standalone Single-Query Execution
Query the RAG pipeline directly from your shell terminal with customized document recall limits:
```bash
uv run python -m student answer "How to configure OpenAI server?" --k 10
```

### Phase 3: Batch Evaluation Suite
Process a comprehensive validation dataset in bulk to output structured retrieval files:
```bash
uv run python -m student answer_dataset --student_search_results_path "data/output/search_results.json"
```

## 🎯 Verification and Grading

To run structural consistency checks and calculate recall accuracies against the gold standard ground truth validation targets, execute the corresponding native testing binaries:

```bash
# Evaluate retrieval matching capabilities
./moulinette/moulinette-ubuntu evaluate_student_search_results \
  --student_answer_path data/output/search_results.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10

# Evaluate answer generation fidelity
./moulinette/moulinette-ubuntu evaluate_student_answers \
  --student_answer_path data/output/answers.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```
