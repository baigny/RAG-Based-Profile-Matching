# RAG Based Profile Matching — Plan

## Context
Assignment: RAG pipeline that matches resumes against job descriptions. Ingest resumes, chunk by section, embed + store in a vector DB, retrieve by semantic + keyword hybrid search, score 0-100, and generate LLM reasoning for each match. Reuses `fs_tools.py` from the earlier `LLM-Powered-File-System-Assistant` project as the file-ingestion layer — no reimplementing PDF/DOCX reads.

Stack: **Ollama** (local, no API key) for LLM calls (segmentation, metadata extraction, reasoning) and embeddings, **ChromaDB** for the vector store.

## Folder structure
```
RAG-Based-Profile-Matching/
├── backend/
│   ├── __init__.py
│   ├── fs_tools.py           # ported from previous project (read/write/list/search files)
│   ├── chunking.py           # section-based resume chunking
│   ├── embeddings.py         # Ollama embedding wrapper
│   ├── vector_store.py       # ChromaDB client + collection helpers
│   └── metadata_extractor.py # LLM call -> structured JSON (name, skills, years, education)
├── ai/
│   ├── resume_rag.py         # ingestion pipeline: fs_tools -> chunk -> embed -> store
│   └── job_matcher.py        # JD embed -> hybrid retrieval -> score -> LLM reasoning
├── data/
│   ├── resumes/              # 30+ synthetic resumes (.txt)
│   └── job_descriptions/     # 5+ synthetic JDs (.txt)
├── scripts/
│   └── generate_data.py      # LLM-generates synthetic resumes + JDs
├── eval/
│   ├── ground_truth.json     # manually labeled resume<->JD matches
│   └── evaluate.py           # precision@10 + latency per query
├── chroma_db/                # persisted vector store (gitignored)
├── output/                   # scratch output (match reports)
├── requirements.txt
├── PLAN.md
├── README.md
└── .gitignore
```

## Phase 0 — Environment check
- Confirm Ollama installed + reachable: `ollama --version`, `ollama list`.
- Pull models: `ollama pull llama3.1` (reasoning/segmentation/extraction), `ollama pull nomic-embed-text` (embeddings).
- Python 3.10+, venv.
- Verify: venv activates clean, both models show in `ollama list`.

## Phase 1 — Scaffold
- Create folder structure above, empty `__init__.py`, `.gitignore` (`venv/`, `__pycache__/`, `*.pyc`, `chroma_db/`, `output/*` with `.gitkeep`).
- `requirements.txt`: `ollama`, `chromadb`, `pypdf`, `python-docx`.
- Verify: `pip install -r requirements.txt` succeeds.

## Phase 2 — Port `fs_tools.py`
- Copy `fs_tools.py` from `LLM-Powered-File-System-Assistant` into `backend/`.
- Confirm `read_file` preserves line breaks between sections (needed for chunking) — patch if it flattens whitespace.
- Verify: standalone smoke test against `data/resumes/`.

## Phase 3 — Data generation
- `scripts/generate_data.py`: prompts `llama3.1` to generate 30+ synthetic resumes (varied seniority, skill sets, formats) and 5+ JDs, writes to `data/resumes/` and `data/job_descriptions/`.
- Verify: 30+ resume files, 5+ JD files, spot-check diversity.

## Phase 4 — Chunking
- `backend/chunking.py`: LLM call per resume (via `llama3.1`) segments raw text into sections (Education, Experience, Skills, etc.), returns list of `{section, text}`.
- Verify: run against a few sample resumes, confirm sane section boundaries.

## Phase 5 — Embeddings + vector store
- `backend/embeddings.py`: wraps `ollama.embeddings(model="nomic-embed-text", ...)`.
- `backend/vector_store.py`: ChromaDB persistent client, one collection, one entry per chunk.
- `backend/metadata_extractor.py`: LLM call returns structured JSON (name, skills, years_experience, education) per resume; stored as chunk metadata.
- `ai/resume_rag.py`: full ingestion pipeline — `fs_tools.list_files` -> `fs_tools.read_file` -> chunk -> extract metadata -> embed -> upsert into Chroma.
- Verify: ingest all resumes, confirm collection count matches chunk count, basic similarity query returns sane results.

## Phase 6 — Hybrid search + scoring + reasoning
- `ai/job_matcher.py`:
  1. Embed JD text, query Chroma top-K (K=10).
  2. Extract must-have keywords from JD (LLM or simple heuristic), run keyword match via `fs_tools.search_in_file`-style logic against candidate resumes, merge/re-rank with semantic results (hybrid).
  3. Score 0-100: normalize similarity, boost for must-have keyword presence, penalize for absence.
  4. Metadata filter support (e.g. `years_experience >= N`) applied before/after retrieval.
  5. LLM reasoning call per top match: JD + matched chunks -> 1-2 sentence justification.
- CLI entrypoint: `python ai/job_matcher.py "path/to/jd.txt"`.
- Verify: run against a JD, confirm ranked list with scores + reasoning.

## Phase 7 — Eval
- `eval/ground_truth.json`: manually labeled which resumes should match each JD.
- `eval/evaluate.py`: computes precision@10 and latency per query against ground truth.
- Verify: script runs, prints metrics table.

## Phase 8 — Docs
- `README.md`: setup (Ollama install + pulls, venv, pip install), project structure, how to run ingestion + matching, example output, eval results.
- Finalize `requirements.txt`.

## Phase 9 — Demo video
- Manual step (not automated): 2-3 min screen capture running ingestion + a match query + eval script.
