"""Sanity checks for job_matcher.py: experience filtering and keyword coverage.

Adapted from a reference verify_matching.py that shells out to job_matcher.py
and parses JSON. That reference project uses OpenAI; this one calls our
Ollama-only ai/job_matcher.py (--json flag) instead.
"""
import json
import subprocess
import sys

PYTHON = sys.executable


def run_job_matcher(jd_path, min_years=None, top=10):
    cmd = [PYTHON, "ai/job_matcher.py", jd_path, "--top", str(top), "--json"]
    if min_years is not None:
        cmd.extend(["--min-years", str(min_years)])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {result.stderr}")
        return None
    return json.loads(result.stdout)


def test_experience_filtering(jd_path, min_years):
    print(f"--- Test 1: Experience filtering (min_years={min_years}) ---")
    results = run_job_matcher(jd_path, min_years=min_years)
    if not results:
        print("FAIL: no results returned.")
        return False

    matches = results["top_matches"]
    print(f"Query returned {len(matches)} matches:")
    all_passed = True
    for m in matches:
        ok = m["years_experience"] >= min_years
        all_passed = all_passed and ok
        tag = "OK" if ok else "FAIL"
        print(f" - {tag}: {m['name']} ({m['source_file']}), years={m['years_experience']}, score={m['score']}")

    print(f"Experience filtering check: {'PASS' if all_passed else 'FAIL'}\n")
    return all_passed


def test_keyword_coverage(jd_path, min_keyword_match=0.0):
    print(f"--- Test 2: Keyword coverage (jd={jd_path}) ---")
    results = run_job_matcher(jd_path)
    if not results:
        print("FAIL: no results returned.")
        return False

    matches = results["top_matches"]
    print(f"Query returned {len(matches)} matches:")
    all_passed = True
    for m in matches:
        ok = m["keyword_match"] >= min_keyword_match
        all_passed = all_passed and ok
        tag = "OK" if ok else "FAIL"
        print(f" - {tag}: {m['name']}, score={m['score']}, keyword_match={m['keyword_match']}, semantic={m['semantic_similarity']}")

    print(f"Keyword coverage check: {'PASS' if all_passed else 'FAIL'}\n")
    return all_passed


if __name__ == "__main__":
    jd_path = sys.argv[1] if len(sys.argv) > 1 else "data/job_descriptions/jd_01_senior_backend_engineer.txt"
    test_experience_filtering(jd_path, min_years=5)
    test_keyword_coverage(jd_path)
