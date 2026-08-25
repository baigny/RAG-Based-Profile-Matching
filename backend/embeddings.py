"""Ollama embedding wrapper."""
import ollama

MODEL = "nomic-embed-text"


def embed_text(text):
    """Returns embedding vector (list[float]) for a single string."""
    response = ollama.embed(model=MODEL, input=text)
    return response["embeddings"][0]


def embed_texts(texts):
    """Returns list of embedding vectors for a list of strings, in one batched call."""
    if not texts:
        return []
    response = ollama.embed(model=MODEL, input=texts)
    return response["embeddings"]
