import os
import requests
from app.services.retriever import retrieve


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = "gemma3"

def run_rag(query: str, top_k: int = 8) -> dict:
    #finding most important chunks related to the query (top k) and also their source and page
    chunks = retrieve(query, top_k=top_k)

    #make them understandable / readable for model
    context = "\n\n---\n\n".join(
        f"[Source: {c['metadata']['source']}, Page {c['metadata'].get('page', '?')}]\n{c['text']}"
        for c in chunks
    )

    #actual prompt
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say "I could not find this in the provided documents."

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    #request sent to ollama
    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    })

    answer = response.json()["response"]

    #return the answer in the schema of QueryResponses
    return {
        "query": query,
        "answer": answer,
        "chunks": [
            {
                "text": c["text"],
                "source": c["metadata"]["source"],
                "page": c["metadata"].get("page", 0),
                "score": round(c["score"], 4)       
            }
            for c in chunks
        ]
    }