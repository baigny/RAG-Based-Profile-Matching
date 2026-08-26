"""Generate synthetic resumes + job descriptions via Ollama (llama3.1)."""
import os
import re
import sys

import ollama

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend import fs_tools

MODEL = "llama3.2:3b"
RESUME_DIR = "data/resumes"
JD_DIR = "data/job_descriptions"

ROLES = [
    ("Junior", "Frontend Developer", "React, JavaScript, CSS, HTML, Git"),
    ("Senior", "Backend Engineer", "Python, Django, PostgreSQL, Docker, AWS"),
    ("Mid-level", "Data Scientist", "Python, pandas, scikit-learn, SQL, statistics"),
    ("Senior", "DevOps Engineer", "Kubernetes, Terraform, AWS, CI/CD, Linux"),
    ("Junior", "QA Engineer", "Selenium, pytest, manual testing, JIRA"),
    ("Senior", "Full Stack Developer", "React, Node.js, TypeScript, MongoDB, GraphQL"),
    ("Mid-level", "Mobile Developer", "Swift, iOS, Kotlin, Android, REST APIs"),
    ("Senior", "Machine Learning Engineer", "PyTorch, TensorFlow, MLOps, Python, NLP"),
    ("Junior", "Data Analyst", "SQL, Excel, Tableau, Python, data visualization"),
    ("Senior", "Cloud Architect", "AWS, Azure, Kubernetes, microservices, security"),
    ("Mid-level", "Product Manager", "roadmapping, Agile, stakeholder management, analytics"),
    ("Senior", "Site Reliability Engineer", "Kubernetes, monitoring, Prometheus, Go, Linux"),
    ("Junior", "Backend Developer", "Node.js, Express, MongoDB, REST APIs"),
    ("Mid-level", "Data Engineer", "Spark, Airflow, SQL, Python, ETL pipelines"),
    ("Senior", "Security Engineer", "penetration testing, SIEM, network security, Python"),
    ("Junior", "UI/UX Designer", "Figma, wireframing, user research, prototyping"),
    ("Mid-level", "Systems Administrator", "Linux, networking, Bash, Ansible, monitoring"),
    ("Senior", "Engineering Manager", "team leadership, Agile, hiring, technical strategy"),
    ("Junior", "Applied Data Scientist", "Python, pandas, numpy, machine learning basics"),
    ("Mid-level", "UI Engineer", "Vue.js, JavaScript, SCSS, webpack, testing"),
    ("Senior", "API Engineer", "Java, Spring Boot, Kafka, microservices, SQL"),
    ("Junior", "Cloud Support Engineer", "AWS, troubleshooting, Linux, customer support"),
    ("Mid-level", "ML Ops Engineer", "scikit-learn, XGBoost, Python, model deployment"),
    ("Senior", "Research Scientist", "deep learning, NLP, Python, PyTorch, research"),
    ("Junior", "Web Developer", "HTML, CSS, JavaScript, Flask, SQLite"),
    ("Mid-level", "Release Engineer", "Jenkins, Docker, GitLab CI, AWS, scripting"),
    ("Senior", "iOS Developer", "Swift, SwiftUI, Xcode, Core Data, APIs"),
    ("Mid-level", "Platform Engineer", "Ruby on Rails, PostgreSQL, Redis, Sidekiq"),
    ("Senior", "Analytics Engineer", "Snowflake, dbt, Airflow, Python, SQL"),
    ("Junior", "Android Developer", "Kotlin, Java, Android SDK, REST APIs"),
    ("Mid-level", "Technical Writer", "API documentation, Markdown, developer tools, editing"),
    ("Senior", "Solutions Architect", "AWS, system design, client consulting, microservices"),
]

JD_ROLES = [
    ("Senior Backend Engineer", "Python, Django, PostgreSQL, Docker, AWS", 5),
    ("Frontend Developer", "React, JavaScript, TypeScript, CSS", 2),
    ("Data Scientist", "Python, machine learning, SQL, statistics", 3),
    ("DevOps Engineer", "Kubernetes, Terraform, AWS, CI/CD", 4),
    ("Machine Learning Engineer", "PyTorch, NLP, MLOps, Python", 4),
    ("Full Stack Developer", "React, Node.js, MongoDB, GraphQL", 3),
]

RESUME_PROMPT = """Write a realistic but fully fictional plain-text resume for a {level} {role}.
Use these skills/technologies naturally throughout: {skills}.
Use a fictional name (not a real person). Include these sections, each on its own line as an ALL-CAPS header:
NAME, SUMMARY, SKILLS, EXPERIENCE, EDUCATION.
Under EXPERIENCE, list 1-3 jobs with company name (fictional), title, dates, and 2-3 bullet points each.
Under EDUCATION, list one degree with fictional school name and year.
Keep it under 400 words. Output ONLY the resume text, no preamble, no markdown formatting, no code fences."""

JD_PROMPT = """Write a realistic but fully fictional plain-text job description for a {role} position requiring at least {years} years of experience.
Use these must-have skills naturally: {skills}.
Include these sections, each on its own line as an ALL-CAPS header:
TITLE, ABOUT, RESPONSIBILITIES, REQUIREMENTS, NICE TO HAVE.
Keep it under 350 words. Output ONLY the job description text, no preamble, no markdown formatting, no code fences."""


def clean(text):
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    return text.strip()


def generate(prompt):
    response = ollama.generate(model=MODEL, prompt=prompt)
    return clean(response["response"])


def main():
    os.makedirs(RESUME_DIR, exist_ok=True)
    os.makedirs(JD_DIR, exist_ok=True)

    for i, (level, role, skills) in enumerate(ROLES, start=1):
        fname = f"resume_{i:02d}_{role.lower().replace(' ', '_').replace('/', '_')}.txt"
        fpath = os.path.join(RESUME_DIR, fname)
        if os.path.exists(fpath):
            print(f"[resume {i}/{len(ROLES)}] skip {fname} (exists)")
            continue
        prompt = RESUME_PROMPT.format(level=level, role=role, skills=skills)
        content = generate(prompt)
        fs_tools.write_file(fpath, content)
        print(f"[resume {i}/{len(ROLES)}] wrote {fname} ({len(content)} chars)")

    for i, (role, skills, years) in enumerate(JD_ROLES, start=1):
        fname = f"jd_{i:02d}_{role.lower().replace(' ', '_').replace('/', '_')}.txt"
        fpath = os.path.join(JD_DIR, fname)
        if os.path.exists(fpath):
            print(f"[jd {i}/{len(JD_ROLES)}] skip {fname} (exists)")
            continue
        prompt = JD_PROMPT.format(role=role, skills=skills, years=years)
        content = generate(prompt)
        fs_tools.write_file(fpath, content)
        print(f"[jd {i}/{len(JD_ROLES)}] wrote {fname} ({len(content)} chars)")


if __name__ == "__main__":
    main()
