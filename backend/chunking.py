"""LLM-based section chunking for resumes."""
import json
import re

import ollama

MODEL = "llama3.2:3b"

CHUNK_PROMPT = """Split the following resume into its sections (e.g. Summary, Skills, Experience, Education, Certifications, Projects).
Return ONLY a JSON array, no preamble, no markdown fences. Each element must be an object:
{{"section": "<section name>", "text": "<full text of that section, verbatim>"}}
Preserve the original wording. Do not summarize or omit content. Every line of the resume must belong to exactly one section.

RESUME:
{resume_text}"""


def _strip_fences(text):
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _fallback_chunks(resume_text):
    """Regex-based split on ALL-CAPS header lines, used if the LLM output isn't parseable."""
    lines = resume_text.splitlines()
    chunks = []
    current_section = "SUMMARY"
    current_lines = []
    header_re = re.compile(r"^[A-Z][A-Z \-/&]{2,}$")
    for line in lines:
        if header_re.match(line.strip()) and len(line.strip()) < 40:
            if current_lines:
                chunks.append({"section": current_section, "text": "\n".join(current_lines).strip()})
            current_section = line.strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        chunks.append({"section": current_section, "text": "\n".join(current_lines).strip()})
    return [c for c in chunks if c["text"]]


def chunk_resume(resume_text):
    """Segments resume text into sections via LLM. Returns list of {"section", "text"}."""
    prompt = CHUNK_PROMPT.format(resume_text=resume_text)
    response = ollama.generate(model=MODEL, prompt=prompt, format="json")
    raw = _strip_fences(response["response"])

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            for value in parsed.values():
                if isinstance(value, list):
                    parsed = value
                    break
        chunks = [
            {"section": str(c["section"]).strip(), "text": str(c["text"]).strip()}
            for c in parsed
            if isinstance(c, dict) and c.get("section") and c.get("text")
        ]
        if chunks:
            return chunks
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    return _fallback_chunks(resume_text)
