# RegLens — Local Regulation Analyst

A document question-answering application that runs entirely on your machine using
**Microsoft Foundry Local**. It answers questions from uploaded documents, shows the
exact source passages behind every answer, and is tuned for structured EU legislation —
specifically the **Medical Device Regulation (EU) 2017/745 (MDR)** and the
**In Vitro Diagnostic Regulation (EU) 2017/746 (IVDR)**.

General TXT, Markdown, PDF and DOCX documents work too; the legal-aware features simply
switch off when a document has no article structure.

No cloud account, no API key, no network calls at inference time.

---

## What it does

- Upload up to 20 documents (`.pdf`, `.docx`, `.txt`, `.md`, 10 MB each).
- Parse EU legislation into its real hierarchy — chapters, sections, articles,
  numbered paragraphs, lettered points, annexes — and cite it precisely
  (*"MDR, Article 10(9), p. 23"*).
- Retrieve with a **hybrid** of semantic embeddings, BM25 keyword ranking, and
  legal structure signals.
- Return **definitions and enumerated lists verbatim from the source**, skipping the
  chat model entirely when synthesis would only add latency and distortion.
- Show a **confidence score** and **highlight the source sentences** that support the
  answer.
- **Compare mode**: ask how MDR and IVDR differ and get a grouped, side-by-side answer.
- Refuse to answer when the documents do not cover the question.
- Keep a session history and export it as Markdown with citations intact.

---

## Architecture

```
                       ┌──────────────────────────────┐
   Question ──────────▶│  Question planner            │  type + scope + article refs
                       └──────────────┬───────────────┘
                                      ▼
        ┌───────────────────────────────────────────────────────┐
        │  Hybrid retrieval                                     │
        │    semantic (embeddings)  0.6                         │
        │  + keyword   (BM25)       0.3   + explicit-reference  │
        │  + structure (legal)      0.1     bonus               │
        └──────────────┬────────────────────────────────────────┘
                       ▼
             ┌──────────────────────┐
             │  Context expansion   │  pull in sibling paragraphs of the same article
             └──────────┬───────────┘
                        ▼
        ┌───────────────────────────────────────┐
        │  Answer                               │
        │   • direct  → quote the source        │  definitions, lists
        │   • synthesis → Foundry Local chat    │  everything else
        │   • refused → "not in the documents"  │  low confidence
        └──────────────┬────────────────────────┘
                       ▼
             ┌──────────────────────┐
             │  Validate + highlight│  citation check, repetition guard, span matching
             └──────────────────────┘
```

Storage is SQLite; retrieval scores the corpus in memory (capped at 20 documents).

---

## Technology

| Area | Choice |
| --- | --- |
| Frontend | React 18, TypeScript, Vite |
| Backend | FastAPI, Uvicorn |
| Local AI runtime | Microsoft Foundry Local |
| Chat model | `phi-3.5-mini` (configurable) |
| Embedding model | `qwen3-embedding-0.6b` |
| Retrieval | Embeddings + BM25 (own implementation) + legal-aware scoring |
| Storage | SQLite |
| Document processing | PyMuPDF, python-docx |

---

## Setup

### 1. Install the Foundry Local runtime

```bash
winget install Microsoft.FoundryLocal
```

macOS: `brew install microsoft/foundrylocal/foundrylocal`

### 2. Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

The first request downloads the chat and embedding models (~3 GB combined) and can take
several minutes. Subsequent starts load from cache in about 20 seconds.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

### 4. Get the regulations

The app ships with no documents. Download the two reference regulations and upload them
through the interface:

- **MDR** — Regulation (EU) 2017/745:
  https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32017R0745
- **IVDR** — Regulation (EU) 2017/746:
  https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32017R0746

Indexing a 175-page regulation takes several minutes on CPU (it embeds ~600 chunks);
the document list shows live progress and the app stays usable meanwhile.

---

## Configuration

Every setting in `backend/app/config.py` can be overridden with a `REGLENS_`-prefixed
environment variable:

```bash
set REGLENS_CHAT_MODEL_ALIAS=phi-4-mini
set REGLENS_TOP_K=8
```

| Setting | Default | Notes |
| --- | --- | --- |
| `chat_model_alias` | `phi-3.5-mini` | `phi-4-mini` is stronger but ~2× the size and slower on CPU |
| `embedding_model_alias` | `qwen3-embedding-0.6b` | 1024-dimension vectors |
| `force_cpu_variant` | `true` | See "GPU" below |
| `model_cache_dir` | `~/.foundry-shared/cache/models` | Shared across projects — see below |
| `top_k` | `6` | Chunks passed to the answer layer |
| `semantic_weight` / `keyword_weight` / `structure_weight` | `0.6 / 0.3 / 0.1` | Hybrid blend |

### Shared model cache

Foundry Local defaults its cache to `~/.{app_name}/cache/models`, so **every application
using the SDK downloads its own copy of the same multi-gigabyte models**. RegLens points
at a shared directory instead. If you run several Foundry Local projects, point them all
at the same path and they will share one copy.

### GPU

On some machines the OpenVINO GPU execution provider fails to initialise:

```
Could not find an implementation for EPContext(1) node with name 'ContextNode'
```

RegLens therefore selects each model's **CPU variant** by default. If your GPU works,
set `REGLENS_FORCE_CPU_VARIANT=false` for a significant speed-up.

---

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Backend + Foundry Local status |
| `GET` | `/api/documents` | List indexed documents |
| `POST` | `/api/documents` | Upload and index a document |
| `DELETE` | `/api/documents/{id}` | Delete a document and its chunks |
| `POST` | `/api/chat` | Ask a question (`stream: true` for SSE) |
| `GET` | `/api/history` | Session history |
| `DELETE` | `/api/history` | Clear history |
| `GET` | `/api/history/export` | Export the session as Markdown |

Interactive docs at http://localhost:8000/docs.

---

## Tests

```bash
cd backend
.venv\Scripts\python -m pytest tests/ -q
```

86 tests covering the legal parser, chunker, BM25, question planner, hybrid retrieval,
answer paths, validation, highlighting, and every API endpoint. They use a deterministic
stub embedder, so **no model download is required** and the suite finishes in ~3 seconds.

Two manual scripts need the real PDFs in `backend/data/samples/`:

```bash
python tests/check_real_documents.py   # parser diagnostics on the real regulations
python tests/e2e_real_models.py        # full pipeline with real Foundry Local models
```

---

## Design notes and limitations

- **Retrieval is brute-force over the corpus in memory.** At 20 documents (~1,200 chunks)
  this is well under a second and far simpler than maintaining a vector index. Beyond
  that scale, `sqlite-vec` or a dedicated vector database is the right move.
- **Embeddings are stored as JSON text** in SQLite. A binary format would be more compact,
  but this keeps the database inspectable, which matters more at this size.
- **Answer latency is dominated by the chat model on CPU** — expect roughly 20–40 seconds
  per synthesised answer on a laptop without GPU acceleration. Two things mitigate it:
  definitions and lists take the direct path and return in under a second, and the
  synthesis path streams tokens so the answer appears as it is written.
- **The parser targets EUR-Lex drafting conventions.** Regulations from other publishers
  will fall back to heading-based chunking, which still works but yields coarser citations.
- **Confidence is a retrieval signal, not a correctness guarantee.** It combines semantic
  strength, the margin over the runner-up, and whether the chunk type matches the question
  type. High confidence means the right passage was probably found — not that the model
  read it correctly, which is exactly why the source panel shows the underlying text.

---

## License

[MIT](LICENSE) © Elifnur Demirezen

The EU regulations themselves are published by the European Union and are not included
in this repository; download them from EUR-Lex using the links above.
