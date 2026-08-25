"""Computes precision@10 and latency per query against ground truth.

Runs with include_reasoning=False: reasoning text isn't part of precision/latency
and would add ~10 extra LLM calls per JD for no metric benefit."""
import json
import os
import sys
import time

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


def main():
    with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    rows = []
    for jd_filename, relevant in ground_truth.items():
        jd_path = os.path.join(JD_DIR, jd_filename)
        read_result = fs_tools.read_file(jd_path)
        if not read_result["success"]:
            print(f"skip {jd_filename}: {read_result['error']}")
            continue

        relevant_set = set(relevant)
        start = time.perf_counter()
        results = job_matcher.match(read_result["content"], top_n=K, include_reasoning=False)
        latency = time.perf_counter() - start

        retrieved = [c["source_file"] for c in results]
        p_at_k = precision_at_k(retrieved, relevant_set, K)

        rows.append({
            "jd": jd_filename,
            "precision@10": round(p_at_k, 3),
            "latency_sec": round(latency, 2),
            "retrieved": len(retrieved),
            "relevant": len(relevant_set),
        })

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
