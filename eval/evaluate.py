"""Computes precision@10 and latency per query against ground truth.

Runs with include_reasoning=False: reasoning text isn't part of precision/latency
and would add ~10 extra LLM calls per JD for no metric benefit.

JDs are evaluated in parallel (thread pool) since each is an independent
match() call - mostly waiting on local Ollama HTTP calls, so threads help
despite the GIL. extract_keywords() also caches by JD text hash (see
ai/job_matcher.py) so repeated eval runs skip the LLM call entirely."""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend import fs_tools
from ai import job_matcher

JD_DIR = "data/job_descriptions"
GROUND_TRUTH_PATH = "eval/ground_truth.json"
K = 10


def precision_at_k(retrieved, relevant, k):
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for r in top_k if r in relevant)
    return hits / len(top_k)


def evaluate_jd(jd_filename, relevant):
    jd_path = os.path.join(JD_DIR, jd_filename)
    read_result = fs_tools.read_file(jd_path)
    if not read_result["success"]:
        print(f"skip {jd_filename}: {read_result['error']}")
        return None

    relevant_set = set(relevant)
    start = time.perf_counter()
    results = job_matcher.match(read_result["content"], top_n=K, include_reasoning=False)
    latency = time.perf_counter() - start

    retrieved = [c["source_file"] for c in results]
    p_at_k = precision_at_k(retrieved, relevant_set, K)

    return {
        "jd": jd_filename,
        "precision@10": round(p_at_k, 3),
        "latency_sec": round(latency, 2),
        "retrieved": len(retrieved),
        "relevant": len(relevant_set),
    }


def main():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    jd_filenames = list(ground_truth.keys())
    with ThreadPoolExecutor(max_workers=len(jd_filenames)) as pool:
        futures = [pool.submit(evaluate_jd, jd, ground_truth[jd]) for jd in jd_filenames]
        rows = [f.result() for f in futures]
    rows = [r for r in rows if r is not None]

    print(f"\n{'JD':<40} {'P@10':>8} {'Latency(s)':>12} {'Retrieved':>10} {'Relevant':>9}")
    for row in rows:
        print(f"{row['jd']:<40} {row['precision@10']:>8} {row['latency_sec']:>12} {row['retrieved']:>10} {row['relevant']:>9}")

    if rows:
        avg_precision = sum(r["precision@10"] for r in rows) / len(rows)
        avg_latency = sum(r["latency_sec"] for r in rows) / len(rows)
        print(f"\nAverage precision@10: {round(avg_precision, 3)}")
        print(f"Average latency: {round(avg_latency, 2)}s")


if __name__ == "__main__":
    main()
