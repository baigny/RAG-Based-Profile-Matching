"""Ingestion pipeline: fs_tools -> chunk -> extract metadata -> embed -> upsert into Chroma."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend import chunking, embeddings, fs_tools, metadata_extractor, vector_store

RESUME_DIR = "data/resumes"


def ingest_resume(filepath):
    read_result = fs_tools.read_file(filepath)
    if not read_result["success"]:
        print(f"  skip {filepath}: {read_result['error']}")
        return 0

    resume_text = read_result["content"]
    filename = os.path.basename(filepath)

    meta = metadata_extractor.extract_metadata(resume_text)
    chunks = chunking.chunk_resume(resume_text)

    to_upsert = []
    for i, chunk in enumerate(chunks):
        embedding = embeddings.embed_text(chunk["text"])
        to_upsert.append({
            "id": f"{filename}::{i}::{chunk['section']}",
            "text": chunk["text"],
            "embedding": embedding,
            "metadata": {
                "source_file": filename,
                "section": chunk["section"],
                "name": meta["name"],
                "skills": ", ".join(meta["skills"]),
                "years_experience": meta["years_experience"],
                "education": meta["education"],
            },
        })

    vector_store.upsert_chunks(to_upsert)
    return len(to_upsert)


def ingest_all(resume_dir=RESUME_DIR):
    files = fs_tools.list_files(resume_dir)
    total_chunks = 0
    for i, entry in enumerate(files, start=1):
        filepath = os.path.join(resume_dir, entry["name"])
        n = ingest_resume(filepath)
        total_chunks += n
        print(f"[{i}/{len(files)}] {entry['name']}: {n} chunks")
    print(f"Done. {len(files)} resumes, {total_chunks} chunks. Collection count: {vector_store.count()}")


if __name__ == "__main__":
    ingest_all()
