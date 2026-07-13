# GOT-QA: A RAG-Based Q&A System over A Song of Ice and Fire

Ask any question about the *A Song of Ice and Fire* book series and get a grounded, source-attributed answer — powered by FAISS retrieval, CrossEncoder reranking, and Claude Haiku.

---

## Architecture

```
                        ┌─────────────────────────────────┐
                        │         User Question            │
                        └────────────────┬────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────┐
                        │   MPNet Embedding Model          │
                        │  (all-mpnet-base-v2, 768-dim)   │
                        │   Encodes query → dense vector   │
                        └────────────────┬────────────────┘
                                         │
                                         ▼
                        ┌─────────────────────────────────┐
                        │   FAISS IndexFlatIP              │
                        │   7,394 chunk vectors            │
                        │   Cosine similarity → top-30    │
                        └────────────────┬────────────────┘
                                         │  30 candidates
                                         ▼
                        ┌─────────────────────────────────┐
                        │   CrossEncoder Reranker          │
                        │  (ms-marco-MiniLM-L-6-v2)       │
                        │   Re-scores all 30 pairs         │
                        │   → keeps top 3                  │
                        └────────────────┬────────────────┘
                                         │  3 anchor chunks
                                         ▼
                        ┌─────────────────────────────────┐
                        │   Context Window Expansion       │
                        │   ±4 chunks per anchor           │
                        │   (same book only)               │
                        │   → up to ~27 chunks             │
                        └────────────────┬────────────────┘
                                         │  ~27 chunks
                                         ▼
                        ┌─────────────────────────────────┐
                        │   Claude Haiku                   │
                        │  (claude-haiku-4-5, 200K ctx)   │
                        │   Grounded answer + source attr  │
                        └─────────────────────────────────┘
```

---

## Pipeline Steps

| Step | What Happens |
|---|---|
| **Load & Clean** | 5 `.txt` book files loaded; `clean_text()` strips control chars, BOM, normalises whitespace |
| **Tokenize & Chunk** | MPNet tokenizer splits text into 400-token chunks with 80-token overlap; 7,394 total chunks |
| **Embed** | `all-mpnet-base-v2` encodes all chunks in batches of 128 → 768-dim float32 vectors |
| **Index** | L2-normalised vectors stored in `faiss.IndexFlatIP` for exact cosine similarity search |
| **Retrieve** | Query embedded → FAISS returns top-30 candidate chunks |
| **Rerank** | CrossEncoder scores all 30 query-chunk pairs → top-3 retained |
| **Expand** | Each of the 3 anchors expands to ±4 neighbouring chunks (same book) → ~27 chunks |
| **Answer** | All ~27 chunks passed to Claude Haiku (200K context window); answer is 1–3 sentences with source citation |

---

## Key Numbers

| Parameter | Value |
|---|---|
| Books indexed | 5 (A Song of Ice and Fire, Books 1–5) |
| Total chunks | 7,394 |
| Chunk size | 400 tokens (MPNet tokenizer) |
| Chunk overlap | 80 tokens |
| Embedding dimension | 768 |
| FAISS index type | `IndexFlatIP` (exact, cosine via L2 norm) |
| FAISS top-k | 30 |
| Reranker top-n | 3 |
| Context window expansion | ±4 chunks |
| Max context to LLM | ~27 chunks |
| LLM | Claude Haiku (`claude-haiku-4-5-20251001`) |
| Approx cost per query | ~$0.0014 |

---

## What I Learned Building This

### 1. Text Preprocessing Matters More Than It Looks

The `.txt` files carried years of encoding artifacts — byte order marks (BOM), control characters, broken Unicode sequences from OCR or copy-paste. Early runs produced chunks with junk characters that contaminated embeddings and sometimes caused retrieval failures on otherwise straightforward questions.

The fix was a targeted `clean_text()` function:

```python
def clean_text(text):
    text = re.sub(r"\s+", " ", text)                     # collapse whitespace
    text = text.replace("\n", " ")                        # fix broken line breaks
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", "", text)  # strip control chars
    text = text.replace('﻿', '')                     # strip BOM
    return text.strip()
```

Books are opened with `encoding="utf-8", errors="ignore"` to silently discard any remaining malformed bytes rather than crashing at ingest time.

### 2. Chunking Strategy Is Where You Win or Lose Meaning

Books break at page boundaries, and meaning often spans those breaks — a character's name introduced at the end of one page, their action described at the top of the next. Naive word-count chunking missed these spans entirely.

The solution was using the **actual model tokenizer** (`AutoTokenizer` from `sentence-transformers/all-mpnet-base-v2`) rather than word counts. George R.R. Martin's character names tokenize heavily — "Daenerys" becomes 3 tokens, "Targaryen" another 3 — so word-level estimates consistently underestimated chunk sizes.

Chosen parameters after experimentation:
- **max_tokens = 400** (MPNet's hard limit is 512; headroom prevents truncation during embedding)
- **overlap_tokens = 80** (~20% overlap) — enough to bridge page-end breaks without duplicating too much content

Each chunk carries metadata: `{book_id, chunk_id, position}` — essential for the context expansion step and for source attribution in answers.

### 3. Context Window Was the Real Bottleneck

The initial approach used `deepset/roberta-base-squad2` and then progressively larger variants (`roberta-large-squad2`) as the reader model. Despite good retrieval — the right chunks were being surfaced — the answers were consistently poor or incomplete.

The root cause: RoBERTa's context window is **512 tokens**. After retrieval and reranking, the top-3 chunks totalled ~1,200 tokens. The model was receiving hard-truncated context, often cutting off mid-sentence — so even when retrieval was perfect, the reader never saw the full answer.

Switching to **Claude Haiku** (200,000-token context window) resolved this completely. The full 27-chunk expanded context (~10,000+ tokens) passes to the model intact, and answer quality improved dramatically. The lesson: retrieval quality and reader capacity are both necessary; optimising one without the other yields diminishing returns.

### 4. Context Window Expansion Adds Coherence

After reranking, the top-3 chunks are anchor points — but answers in a novel rarely fit neatly within a 400-token window. The expansion step retrieves ±4 neighbouring chunks per anchor (within the same book), deduplicates, and sorts by position. This reconstructs a coherent reading flow around each relevant passage and eliminates the choppy, out-of-context fragment problem that plagued early answers.

---

## How to Run

### Prerequisites

- Python 3.10+
- GPU recommended (CUDA) — CPU works but embedding ~7,400 chunks takes ~10–15 minutes
- Anthropic API key ([get one here](https://console.anthropic.com))

### Setup

```bash
git clone https://github.com/gyanaranjanmishra/GOT-QA.git
cd GOT-QA
pip install -r requirements.txt
```

> **Note:** The book `.txt` files are not included in this repo for copyright reasons. You will need your own copies of the five *A Song of Ice and Fire* books as plain text files, named `001ssb.txt` through `005ssb.txt`, placed in the directory specified in the notebook's **Load Books** cell.

### Environment

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### Run

Open `Improved_GOT_QA.ipynb` in Jupyter. Execute cells in order:

1. **First run only:** Run all cells — this generates `embeddings/embeddings_0.npy` and `chunk_metadata.pkl` (~80 seconds on GPU)
2. **Subsequent runs:** Skip the embedding/encoding cells (Sections 6–8); load the saved files directly
3. **Ask questions:** Use `run_cli()` for an interactive loop, or call `answer_question("your question")` directly

### Dependencies

```
torch
faiss-cpu          # or faiss-gpu for GPU acceleration
sentence-transformers
anthropic
numpy
nltk
tqdm
transformers
```

---

## Stack

| Component | Tool |
|---|---|
| Embedding | `sentence-transformers/all-mpnet-base-v2` |
| Vector store | FAISS `IndexFlatIP` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM reader | Claude Haiku (`claude-haiku-4-5-20251001`) |
| Language | Python 3.11 |
