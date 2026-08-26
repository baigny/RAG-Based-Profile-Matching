"""LLM call -> structured JSON metadata (name, skills, years_experience, education) per resume."""
import json
import re

import ollama

MODEL = "llama3.2:3b"

EXTRACT_PROMPT = """Extract structured metadata from the resume below.
Return ONLY a JSON object, no preamble, no markdown fences, with exactly these keys:
{{
  "name": "<candidate full name>",
  "skills": ["<skill1>", "<skill2>", ...],
  "years_experience": <integer, best estimate of total years of professional experience>,
  "education": "<highest degree and field, e.g. 'BS Computer Science'>"
}}

RESUME:
{resume_text}"""


def _strip_fences(text):
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def extract_metadata(resume_text):
    prompt = EXTRACT_PROMPT.format(resume_text=resume_text)
    response = ollama.generate(model=MODEL, prompt=prompt, format="json")
    raw = _strip_fences(response["response"])

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    skills = data.get("skills", [])
    if not isinstance(skills, list):
        skills = []

    years = data.get("years_experience", 0)
    try:
        years = int(years)
    except (TypeError, ValueError):
        years = 0

    return {
        "name": str(data.get("name", "") or "Unknown"),
        "skills": [str(s) for s in skills],
        "years_experience": years,
        "education": str(data.get("education", "") or "Unknown"),
    }
