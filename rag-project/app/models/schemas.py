from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class ChunkResult(BaseModel):
    text: str
    source: str
    page: int
    score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    chunks: list[ChunkResult]