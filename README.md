# RAG-Based Profile Matching

A local, zero-cost RAG pipeline that matches resumes against job descriptions. Resumes are chunked by section, embedded, and stored in a vector DB; a job description is matched via hybrid (semantic + keyword) search, scored 0-100, and given an LLM-generated reasoning per match. Everything runs on **Ollama** (local, no API key) and **ChromaDB** (local persistent vector store).

## Project structure

```
RAG-Based-Profile-Matching/
├── backend/
│   ├── fs_tools.py           # file read/write/list/search (ported from LLM-Powered-File-System-Assistant)
│   ├── chunking.py           # LLM-based section chunking for resumes
│   ├── embeddings.py         # Ollama embedding wrapper (nomic-embed-text)
│   ├── vector_store.py       # ChromaDB client + collection helpers
│   └── metadata_extractor.py # LLM call -> structured JSON metadata per resume
├── ai/
│   ├── resume_rag.py         # ingestion pipeline: fs_tools -> chunk -> embed -> store
│   └── job_matcher.py        # JD embed -> hybrid retrieval -> score -> LLM reasoning
├── data/
│   ├── resumes/               # synthetic resumes (.txt)
│   └── job_descriptions/      # synthetic job descriptions (.txt)
├── scripts/
│   └── generate_data.py      # LLM-generates synthetic resumes + JDs
├── eval/
│   ├── ground_truth.json     # manually labeled resume<->JD matches
│   └── evaluate.py           # precision@10 + latency per query
├── chroma_db/                # persisted vector store (gitignored)
├── output/                   # scratch output
└── requirements.txt
```

## Setup

### 1. Install Ollama and pull the models

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

Confirm:

```bash
ollama --version
ollama list
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate synthetic data (optional — sample data already included)

```bash
venv\Scripts\python.exe scripts\generate_data.py
```

Generates 32 synthetic resumes across varied roles/seniority and 6 job descriptions into `data/resumes/` and `data/job_descriptions/`.

### 2. Ingest resumes into the vector store

```bash
venv\Scripts\python.exe ai\resume_rag.py
```

For each resume: reads via `fs_tools`, LLM-chunks into sections (Summary, Skills, Experience, Education, ...), LLM-extracts metadata (name, skills, years of experience, education), embeds each chunk with `nomic-embed-text`, and upserts into a persistent Chroma collection (`chroma_db/`).

### 3. Match a job description against ingested resumes

```bash
venv\Scripts\python.exe ai\job_matcher.py "data\job_descriptions\jd_01_senior_backend_engineer.txt"
```

Optional flags: `--top N` (default 10), `--min-years N` (metadata filter applied at retrieval time).

Pipeline:
1. Embed the JD, query Chroma for the top candidate chunks, aggregate to one best-matching chunk set per resume.
2. LLM extracts must-have keywords from the JD; keyword coverage is measured against each candidate's full resume text.
3. Final score = `0.6 * semantic_similarity + 0.4 * keyword_match`, scaled to 0-100.
4. LLM generates a 1-2 sentence reasoning per top match, grounded in the actual matched excerpts.

### 4. Run evaluation

```bash
venv\Scripts\python.exe eval\evaluate.py
```

Runs every JD in `eval/ground_truth.json` through the matcher, reports precision@10 and latency per query, plus averages.

Results (32 resumes, 6 JDs, local CPU inference):

| JD | P@10 | Latency (s) |
|---|---|---|
| jd_01_senior_backend_engineer | 0.3 | 39.2 |
| jd_02_frontend_developer | 0.4 | 27.1 |
| jd_03_data_scientist | 0.3 | 44.0 |
| jd_04_devops_engineer | 0.3 | 68.7 |
| jd_05_machine_learning_engineer | 0.2 | 23.3 |
| jd_06_full_stack_developer | 0.2 | 35.8 |

Average precision@10: **0.283**, average latency: **39.7s**.

P@10 is capped low by design — ground truth lists only 2-4 relevant resumes per JD out of 32 candidates (e.g. `jd_06` has 2 relevant → 0.2 is the ceiling, and it hit exactly that, meaning both relevant resumes landed in the top 10). Latency is dominated by a single LLM keyword-extraction call per query on CPU; `eval/evaluate.py` runs with `include_reasoning=False` since reasoning text isn't part of either metric.

Also run `eval/verify_matching.py` for quick sanity checks (experience filtering, keyword coverage) — both pass.

## Notes

- All LLM calls (chunking, metadata extraction, keyword extraction, reasoning) use `llama3.1` locally via `ollama.generate(..., format="json")` for structured outputs, with a regex-based fallback in `chunking.py` if the model returns malformed JSON.
- `backend/fs_tools.py` is reused unmodified from the `LLM-Powered-File-System-Assistant` project — no PDF/DOCX parsing was reimplemented. Extended in this project with `.pptx` read support and `.docx`/`.pdf`/`.pptx` writers, to give the synthetic dataset real format diversity (8 resumes each of `.txt`/`.docx`/`.pdf`/`.pptx`).
- Ground truth in `eval/ground_truth.json` is manually labeled by role/skill relevance against the fixed synthetic dataset generated by `scripts/generate_data.py`.
- ChromaDB's HNSW index metric (cosine vs. L2) is fixed at collection creation — `vector_store.py` explicitly requests cosine space since `job_matcher.py`'s similarity math assumes cosine distance in `[0, 2]`.

## Demo video

_TODO: add link after recording._
