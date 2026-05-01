# ingest.py
import os
from app.ingestion.parser import parse_file
from app.ingestion.chunker import chunk_documents
from app.services.retriever import store_chunks

DATA_DIR = "./data/raw"

def ingest_all():
    all_chunks = []
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        print(f"Parsing: {filepath}")
        try:
            pages = parse_file(filepath)
            chunks = chunk_documents(pages)
            all_chunks.extend(chunks)
            print(f"  → {len(pages)} pages, {len(chunks)} chunks")
        except ValueError as e:
            print(f"  Skipping: {e}")

    store_chunks(all_chunks)
    print(f"\nDone. Total chunks stored: {len(all_chunks)}")

if __name__ == "__main__":
    ingest_all()