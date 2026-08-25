"""Ollama embedding wrapper."""
import ollama

MODEL = "nomic-embed-text"


def embed_text(text):
    """Returns embedding vector (list[float]) for a single string."""
    response = ollama.embeddings(model=MODEL, prompt=text)
    return response["embedding"]


def embed_texts(texts):
    """Returns list of embedding vectors for a list of strings."""
    return [embed_text(t) for t in texts]
