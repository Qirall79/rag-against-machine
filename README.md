*This project has been created as part of the 42 curriculum by wbelfatm.*

# RAG Against the Machine

A retrieval-augmented generation system that answers questions about the vLLM codebase. It indexes ~1,900 source and documentation files, retrieves the passages most likely to contain the answer, and generates a grounded response from them with a local Qwen3-0.6B.

The interesting part is not the pipeline — it is the retrieval. The system runs two independent retrievers, one lexical and one semantic, and fuses their rankings. Neither is strong alone. Together they beat both.

---

## Results

Measured with the official evaluation binary against the public question sets. Recall@k counts a source as found when a retrieved chunk lies in the same file and overlaps the reference character span.

| Retriever | docs recall@5 | code recall@5 |
|---|---:|---:|
| BM25 (lexical only) | 0.810 | 0.576 |
| MiniLM (semantic only) | 0.510 | 0.333 |
| **Hybrid, RRF fusion** | **0.800** | **0.606** |

Thresholds are 80% on docs and 50% on code. Both are met.

Full breakdown of the shipped hybrid configuration:

| | @1 | @3 | @5 | @10 |
|---|---:|---:|---:|---:|
| docs | 0.500 | 0.700 | 0.800 | 0.850 |
| code | 0.384 | 0.515 | 0.606 | 0.657 |

**The result worth explaining:** the semantic retriever is substantially worse than BM25 on both question sets — 0.51 against 0.81 on docs, 0.33 against 0.58 on code. Fusing a clearly inferior retriever into a stronger one still improved code retrieval by three points. The two methods fail differently, so the weak one contributes correct chunks the strong one ranks nowhere. That is the entire argument for hybrid retrieval, and it only shows up if you measure each leg separately.

The trade is not free. Docs retrieval loses a point (0.810 to 0.800) because fusion dilutes BM25's top-ranked exact matches with weaker semantic hits. Shipping the hybrid means accepting a docs score sitting exactly on the 80% threshold in exchange for the code gain.

---

## Architecture

```
                 ┌──────────────────────────────────────┐
  data/raw/  ──► │ Chunker                              │
                 │  .py  → def/class boundaries         │
                 │  .md  → paragraph boundaries         │
                 └──────────────┬───────────────────────┘
                                │  one ordered chunk list
                 ┌──────────────┴───────────────┐
                 ▼                              ▼
        ┌─────────────────┐            ┌─────────────────┐
        │ BM25 index      │            │ MiniLM embeddings│
        │ (bm25s, sparse) │            │ (384-d, .npy)    │
        └────────┬────────┘            └────────┬─────────┘
                 │ ranked ids                   │ ranked ids
                 └──────────────┬───────────────┘
                                ▼
                    ┌───────────────────────┐
                    │ Reciprocal Rank Fusion│
                    └───────────┬───────────┘
                                ▼
                   top-k chunks ──► Qwen3-0.6B ──► answer
```

Both indexes are built from **the same ordered chunk list**, so a document id means the same thing to both retrievers and maps to the same metadata entry — file path plus character span. That shared ordering is what makes fusion possible; if the two indexes chunked independently, their ids would be incomparable.

---

## Chunking strategy

A Python file and a Markdown page do not break apart the same way, so the chunker dispatches on extension.

- **Python** splits at top-level `def` and `class` declarations, keeping a function or class intact rather than cutting mid-body.
- **Markdown** splits at blank lines, keeping paragraphs intact.

In both cases a greedy accumulator walks the candidate split positions and extends the current chunk while it fits under `max_chunk_size` (default 2000 characters, configurable via CLI). When the next split would overflow, it cuts at the last valid split and restarts with an overlap window so context is not lost at the seam. A single definition longer than the limit is hard-sliced at exactly `max_chunk_size`.

Every chunk carries the character range it occupies in the original file. Those coordinates are what the evaluator scores, so they are computed from the source text and never from any transformed version of it.

---

## Retrieval

**Lexical — BM25.** Term frequency, inverse document frequency, length normalisation. Exact token matching: it is excellent when a question quotes an identifier verbatim and blind to paraphrase. `car` does not match `automobile`.

**Semantic — all-MiniLM-L6-v2.** Chunks are embedded once at index time into normalised 384-dimensional vectors, persisted as a NumPy array. At query time the question is embedded and ranked by cosine similarity. This catches paraphrase that BM25 cannot, at the cost of precision on rare exact tokens.

**Fusion — Reciprocal Rank Fusion.** Each retriever returns a ranked list. A document's fused score is the sum over both lists of `1 / (k + rank)`, with `k = 60` and rank starting at 1; a document absent from a list contributes nothing from it. Results are re-sorted by fused score.

Fusion operates on **rank, not score**, and that is the point. A BM25 score and a cosine similarity are not comparable quantities — one is unbounded term-weight arithmetic, the other is a bounded dot product. Normalising them onto a common scale requires assumptions that do not hold. Rank position is directly comparable, and RRF only needs the ordering.

---

## Answer generation

The top-k retrieved sources are re-read from disk by character span rather than carried in memory, joined into a single context block, and passed to Qwen3-0.6B behind a ChatML prompt that instructs the model to answer only from the supplied context. Reasoning traces are stripped from the output before it is returned.

Qwen3-0.6B has real reasoning limits at this size. Grounding and retrieval quality carry the answer; the model's job is to phrase what retrieval already found.

---

## Design decisions

**Two indexes, one chunk list.** Building both retrievers from a single ordered pass is what makes their document ids interchangeable. It also means re-chunking is a single change that propagates to both.

**Embeddings stored as `.npy`, not JSON.** A 13k × 384 float matrix in JSON is large, slow to parse, and lossy on round-trip. NumPy's binary format is exact and loads as a ready array.

**Embeddings normalised at index time.** With unit vectors, cosine similarity reduces to a dot product, so query-time scoring is a single matrix multiply.

**No vector database.** Brute-force cosine over ~13k rows is a millisecond-scale operation. FAISS and friends solve a problem this corpus does not have, at the cost of another dependency and another index format to keep in sync.

**No agent or RAG framework.** The chunker, both retrievers, the fusion, and the CLI are written directly against `bm25s`, `sentence-transformers`, and NumPy. The retrieval behaviour is the thing being measured, so it is not delegated to a library that hides it.

---

## Challenges

**Identifier expansion looked obvious and was not.** Questions paraphrase; code uses identifiers. Splitting `AsyncLLMEngine` into `async llm engine` at index time should manufacture the lexical overlap BM25 needs. Applied symmetrically to corpus and query, it did exactly that — code recall@5 rose from 0.586 to 0.717.

It also dropped docs recall@5 from 0.83 to 0.72, because splitting identifiers in natural-language questions fragments terms the prose never fragmented.

Applying it to only one side was far worse than not applying it at all: expanding the corpus but not the query collapsed code recall to 0.182, and the reverse to 0.202. If the corpus stores `async llm engine` and the query still asks for `AsyncLLMEngine`, the token that matters no longer exists on either side of the match.

The fix appeared to be applying it per dataset — expand code queries, leave docs queries alone — which recovered both (0.82 docs, 0.72 code). That configuration turned out to be unreachable at inference time. Docs and code questions are lexically indistinguishable: a CamelCase-and-underscore heuristic fires on 98% of docs questions as well as 99% of code questions, because documentation is full of identifiers too. The only clean signal is the file type of the gold answer, which is precisely what retrieval is trying to find.

The technique was abandoned. Semantic retrieval matches `AsyncLLMEngine` to "asynchronous engine" natively, without needing to mangle the corpus — identifier expansion was a lexical workaround for a problem that has a semantic solution.

**Fusion weights.** RRF was swept with weighted variants favouring the stronger lexical leg. Equal weighting outperformed every asymmetric configuration tried, so the shipped fusion weights both retrievers equally.

**Truncation limits the semantic leg.** MiniLM accepts roughly 256 word-pieces. Chunks run to 2000 characters, so the tail of a long chunk is never embedded. This is the most likely explanation for semantic retrieval performing worse on code (0.333) than on docs (0.510) — code chunks are denser and the answer is less likely to sit in the first few hundred tokens. Smaller chunks for the semantic index alone would test this.

---

## Installation

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/qirall79/rag-agains-machine.git
cd rag-agains-machine
uv sync
```

The vLLM corpus goes in `data/raw/vllm-0.10.1/`. Model weights download on first run.

---

## Usage

**Build both indexes.** Chunks the corpus, builds the BM25 index and the embedding matrix, writes both to `data/processed/`.

```bash
uv run python -m student index --max_chunk_size 2000
```

**Search a single query.**

```bash
uv run python -m student search "How to configure the OpenAI server?" --k 5
```

**Search a dataset.** Writes a `StudentSearchResults` JSON file.

```bash
uv run python -m student search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

**Answer a single query** using the retrieved context.

```bash
uv run python -m student answer "How to configure the OpenAI server?" --k 5
```

**Answer a dataset.** Consumes search results, produces `StudentSearchResultsAndAnswer`.

```bash
uv run python -m student answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

**Score retrieval** with the official evaluator. Student results first, ground truth second.

```bash
./moulinette/moulinette-ubuntu evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 10 --max_context_length 2000
```

The binary is a Linux x86-64 executable. On macOS or ARM, run it inside a `linux/amd64` container with the repository bind-mounted.

---

## Layout

```
rag-agains-machine/
├── data/
│   ├── raw/vllm-0.10.1/       target corpus
│   ├── processed/             BM25 index, embeddings.npy, metadata.json
│   ├── datasets/              question sets
│   └── output/                search results and answers
├── student/
│   ├── __main__.py            CLI (Python Fire)
│   ├── index.py               chunking, BM25 index, embedding index
│   ├── search_dataset.py      retrieval and RRF fusion
│   ├── answer_dataset.py      context assembly and generation
│   ├── evaluator.py           local recall@k, for iteration
│   └── models.py              pydantic schemas
├── pyproject.toml
└── uv.lock
```

---

## What I would change

- **Chunk separately for each index.** One chunk size serves both retrievers today, and it suits neither. BM25 benefits from larger chunks; MiniLM cannot see past ~256 word-pieces of them.
- **Measure latency per leg.** Recall is reported; retrieval cost is not. Fusion doubles the work and nothing in the current numbers says what that costs.
- **A reranker.** A cross-encoder over the fused top-50 is the obvious next lever, and the one most likely to fix the docs recall@1 of 0.500.
- **The 80% docs margin is zero.** Shipping the hybrid trades a point of docs recall for three points of code recall, which leaves no headroom against the threshold. A reranker or per-index chunking would buy that margin back.

---

## Resources

[BM25 (Robertson & Zaragoza)](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) · [Reciprocal Rank Fusion (Cormack et al.)](https://plg.uwaterloo.ca/~gvcormack/cormacksigir09-rrf.pdf) · [sentence-transformers](https://www.sbert.net/) · [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) · [bm25s](https://github.com/xhluca/bm25s)

**AI usage.** AI assistance was used to review the retrieval code, to help design the ablation methodology behind the identifier-expansion experiment, and to review this README. The chunking strategy, the fusion implementation, and the experimental conclusions are mine.