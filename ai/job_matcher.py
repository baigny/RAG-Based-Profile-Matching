"""JD embed -> hybrid retrieval -> score -> LLM reasoning."""
import argparse
import hashlib
import json
import os
import re
import sys
import threading

import ollama

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend import embeddings, fs_tools, vector_store

MODEL = "llama3.2:3b"
RESUME_DIR = "data/resumes"
CANDIDATE_POOL = 30  # chunks pulled from Chroma before aggregating per-resume
TOP_N = 10
KEYWORD_CACHE_PATH = "output/.keyword_cache.json"
_keyword_cache_lock = threading.Lock()

KEYWORD_PROMPT = """Extract the 5-10 most important must-have technical skills/tools/technologies from this job description.
Each item must be a short atomic term as it would literally appear on a resume (e.g. "python", "django", "aws", "docker") -
1-2 words max, NOT a descriptive phrase or sentence (do NOT write things like "python programming skills" or "strong problem-solving skills").
Return ONLY a JSON array of lowercase strings, no preamble, no markdown fences.

JOB DESCRIPTION:
{jd_text}"""

REASONING_PROMPT = """Job description:
{jd_text}

Candidate's matched resume excerpts:
{chunks_text}

In 1-2 sentences, explain why this candidate is a good match for the job (or note key gaps if relevant). Be specific and concise. Output only the explanation, no preamble."""


def _strip_fences(text):
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _load_keyword_cache():
    if os.path.exists(KEYWORD_CACHE_PATH):
        try:
            with open(KEYWORD_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_keyword_cache(cache):
    os.makedirs(os.path.dirname(KEYWORD_CACHE_PATH) or ".", exist_ok=True)
    with open(KEYWORD_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def extract_keywords(jd_text):
    cache_key = hashlib.sha256(jd_text.encode("utf-8")).hexdigest()

    with _keyword_cache_lock:
        cache = _load_keyword_cache()
        if cache_key in cache:
            return cache[cache_key]

    prompt = KEYWORD_PROMPT.format(jd_text=jd_text)
    response = ollama.generate(model=MODEL, prompt=prompt, format="json")
    raw = _strip_fences(response["response"])
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
        keywords = [str(k).lower().strip() for k in parsed if str(k).strip()]
    except (json.JSONDecodeError, TypeError):
        keywords = []

    with _keyword_cache_lock:
        cache = _load_keyword_cache()
        cache[cache_key] = keywords
        _save_keyword_cache(cache)
    return keywords


def _normalize(text):
    """Lowercase and strip everything but letters/digits, so punctuation/spacing
    variants (CI/CD vs ci-cd vs CICD) collapse to the same token stream."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def keyword_score(resume_text, keywords):
    if not keywords:
        return 0.0
    normalized_text = _normalize(resume_text)
    hits = sum(1 for kw in keywords if _normalize(kw) in normalized_text)
    return hits / len(keywords)


def generate_reasoning(jd_text, chunks_text):
    prompt = REASONING_PROMPT.format(jd_text=jd_text, chunks_text=chunks_text)
    response = ollama.generate(model=MODEL, prompt=prompt)
    return response["response"].strip()


def match(jd_text, top_n=TOP_N, min_years=None, semantic_weight=0.6, keyword_weight=0.4, include_reasoning=True):
    jd_embedding = embeddings.embed_text(jd_text)

    where = {"years_experience": {"$gte": min_years}} if min_years else None
    results = vector_store.query(jd_embedding, top_k=CANDIDATE_POOL, where=where)

    if not results["ids"][0]:
        return []

    # aggregate best (lowest distance) chunk per resume
    per_resume = {}
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        source = meta["source_file"]
        if source not in per_resume or dist < per_resume[source]["best_distance"]:
            per_resume[source] = {"best_distance": dist, "meta": meta}
        per_resume[source].setdefault("chunks", []).append(doc)

    keywords = extract_keywords(jd_text)

    candidates = []
    for source, info in per_resume.items():
        meta = info["meta"]
        resume_path = os.path.join(RESUME_DIR, source)
        read_result = fs_tools.read_file(resume_path)
        resume_text = read_result["content"] if read_result["success"] else "\n".join(info["chunks"])

        # distance -> similarity (nomic-embed-text uses cosine distance in [0, 2])
        semantic_sim = max(0.0, 1 - info["best_distance"] / 2)
        kw_score = keyword_score(resume_text, keywords)
        final_score = round((semantic_weight * semantic_sim + keyword_weight * kw_score) * 100, 1)

        candidates.append({
            "source_file": source,
            "name": meta.get("name", "Unknown"),
            "years_experience": meta.get("years_experience", 0),
            "education": meta.get("education", "Unknown"),
            "skills": meta.get("skills", ""),
            "score": final_score,
            "semantic_similarity": round(semantic_sim, 3),
            "keyword_match": round(kw_score, 3),
            "matched_chunks": info["chunks"],
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    top_candidates = candidates[:top_n]

    for c in top_candidates:
        if include_reasoning:
            chunks_text = "\n---\n".join(c["matched_chunks"][:3])
            c["reasoning"] = generate_reasoning(jd_text, chunks_text)
        del c["matched_chunks"]

    return top_candidates


def main():
    parser = argparse.ArgumentParser(description="Match a job description against ingested resumes.")
    parser.add_argument("jd_path", help="Path to job description text file")
    parser.add_argument("--top", type=int, default=TOP_N)
    parser.add_argument("--min-years", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print results as JSON instead of text")
    args = parser.parse_args()

    read_result = fs_tools.read_file(args.jd_path)
    if not read_result["success"]:
        print(f"Error reading {args.jd_path}: {read_result['error']}")
        sys.exit(1)

    jd_text = read_result["content"]
    results = match(jd_text, top_n=args.top, min_years=args.min_years)

    if args.json:
        print(json.dumps({"jd_path": args.jd_path, "top_matches": results}, indent=2))
        return

    print(f"\nTop {len(results)} matches for {args.jd_path}:\n")
    for i, c in enumerate(results, start=1):
        print(f"{i}. {c['name']} ({c['source_file']}) — score {c['score']}/100")
        print(f"   years: {c['years_experience']}, education: {c['education']}")
        print(f"   semantic: {c['semantic_similarity']}, keyword: {c['keyword_match']}")
        print(f"   reasoning: {c['reasoning']}")
        print()


if __name__ == "__main__":
    main()
