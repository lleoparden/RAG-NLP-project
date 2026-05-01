# app/ingestion/chunker.py
from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(pages: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    """
    Split parsed pages into overlapping chunks.

    Justification:
    - 500 tokens ≈ ~375 words, enough context for a complete thought
      without exceeding typical embedding model limits (512 tokens).
    - 50-token overlap prevents answers from being split across
      chunk boundaries (e.g., a sentence starting at token 490).
    - RecursiveCharacterTextSplitter splits on paragraph → sentence →
      word boundaries in that order, so it respects semantic units
      rather than hard character limits.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for j, split in enumerate(splits):
            chunks.append({
                "text": split,
                "metadata": {
                    **page["metadata"],
                    "chunk_index": j
                }
            })
    return chunks