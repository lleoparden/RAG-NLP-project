# app/services/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_NAME = "BAAI/bge-base-en-v1.5"

_model = None 

#checking if model already there
def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

#basically turning words into numbers / vectors (smaller better or more related)
def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()

#comparing the query with the stored chunks to find relevant chunks
def embed_query(query: str) -> list[float]:
   
    return embed_texts([query])[0]