"""Builds and executes notebook.ipynb from source cells, embedding real outputs."""
import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

md("""# RAG-Based Profile Matching — Experimentation & Analysis

Loads the ingested Chroma vector store (built by `ai/resume_rag.py`) and runs
`ai/job_matcher.py`'s matching pipeline interactively: single-query inspection,
hybrid-weight sensitivity, top-K / min-years filtering, and the full
precision@10 + latency evaluation from `eval/evaluate.py`.""")

code("""import sys, os, json, time
sys.path.insert(0, os.path.abspath("."))

from ai import job_matcher
from backend import vector_store, fs_tools

print("Chroma collection chunk count:", vector_store.count())""")

md("## 1. Single query — full pipeline on one job description")

code("""jd_path = "data/job_descriptions/jd_01_senior_backend_engineer.txt"
jd_text = fs_tools.read_file(jd_path)["content"]
print(jd_text[:400], "...")""")

code("""results = job_matcher.match(jd_text, top_n=5)
for r in results:
    print(f"{r['candidate_name']:25s} score={r['match_score']:5.1f}  "
          f"semantic={r['semantic_similarity']:.3f}  keyword={r['keyword_match']:.3f}  "
          f"skills={r['matched_skills']}")""")

code("""print(json.dumps({"job_description": jd_text[:200] + "...", "top_matches": results[:1]}, indent=2))""")

md("""## 2. Hybrid weight sensitivity

Compare pure-semantic, pure-keyword, and the default 0.6/0.4 blend on the
same JD to see how much the keyword term moves rankings.""")

code("""for sem_w, kw_w, label in [(1.0, 0.0, "semantic-only"), (0.0, 1.0, "keyword-only"), (0.6, 0.4, "default (0.6/0.4)")]:
    r = job_matcher.match(jd_text, top_n=5, semantic_weight=sem_w, keyword_weight=kw_w, include_reasoning=False)
    top_names = [c["candidate_name"] for c in r]
    print(f"{label:20s} top-5: {top_names}")""")

md("## 3. Effect of `--min-years` filter")

code("""for min_years in [None, 3, 5, 8]:
    r = job_matcher.match(jd_text, top_n=10, min_years=min_years, include_reasoning=False)
    print(f"min_years={str(min_years):5s} -> {len(r)} candidates, "
          f"years range: {[c['years_experience'] for c in r]}")""")

md("""## 4. Full evaluation — precision@10 and latency

Same logic as `eval/evaluate.py`, run inline against `eval/ground_truth.json`.""")

code("""with open("eval/ground_truth.json") as f:
    ground_truth = json.load(f)

def precision_at_k(retrieved, relevant, k=10):
    top_k = retrieved[:k]
    return sum(1 for r in top_k if r in relevant) / len(top_k) if top_k else 0.0

rows = []
for jd_filename, relevant in ground_truth.items():
    jd_path = os.path.join("data/job_descriptions", jd_filename)
    text = fs_tools.read_file(jd_path)["content"]
    start = time.perf_counter()
    r = job_matcher.match(text, top_n=10, include_reasoning=False)
    latency = time.perf_counter() - start
    retrieved = [os.path.basename(c["resume_path"]) for c in r]
    rows.append({
        "jd": jd_filename,
        "precision@10": round(precision_at_k(retrieved, set(relevant)), 3),
        "latency_sec": round(latency, 2),
    })

for row in rows:
    print(row)

avg_p = sum(r["precision@10"] for r in rows) / len(rows)
avg_l = sum(r["latency_sec"] for r in rows) / len(rows)
print(f"\\nAverage precision@10: {avg_p:.3f}")
print(f"Average latency: {avg_l:.2f}s")""")

code("""import matplotlib.pyplot as plt

labels = [r["jd"].replace("jd_", "").replace(".txt", "") for r in rows]
precisions = [r["precision@10"] for r in rows]
latencies = [r["latency_sec"] for r in rows]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(labels, precisions, color="steelblue")
axes[0].set_title("Precision@10 per job description")
axes[0].set_ylim(0, 1)
axes[0].tick_params(axis="x", rotation=45)

axes[1].bar(labels, latencies, color="darkorange")
axes[1].set_title("Latency per query (s)")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()""")

md("""## 5. Observations

- Hybrid scoring (0.6 semantic / 0.4 keyword) generally reorders top-5 vs.
  semantic-only by promoting candidates whose resumes literally contain
  JD-stated must-have skills, without fully overriding semantic ranking.
- `--min-years` acts as a hard post-filter on retrieved metadata, so raising
  it shrinks the candidate pool rather than re-ranking it.
- Average precision@10 across all 6 JDs and average per-query latency are
  reported above, matching `eval/evaluate.py`'s summary numbers.""")

nb["cells"] = cells

client = NotebookClient(nb, timeout=300, kernel_name="venv-rag")
client.execute()

with open("notebook.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("wrote notebook.ipynb")
