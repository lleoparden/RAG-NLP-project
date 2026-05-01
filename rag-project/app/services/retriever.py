# app/services/retriever.py
import chromadb
from app.services.embedder import embed_texts, embed_query

COLLECTION_NAME = "rag_documents"

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path="./chroma_db")
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def store_chunks(chunks: list[dict]):
    """Embed and upsert chunks into ChromaDB."""
    collection = get_collection()
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)
    ids = [f"{c['metadata']['source']}_p{c['metadata'].get('page',0)}_c{c['metadata']['chunk_index']}" for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    print(f"Stored {len(chunks)} chunks.")


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    """Embed query and return top-k most similar chunks."""
    collection = get_collection()
    query_embedding = embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({"text": doc, "metadata": meta, "score": 1 - dist})  # cosine → similarity
    return chunks