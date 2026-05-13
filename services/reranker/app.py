"""Cross-encoder reranker as a standalone HTTP service."""
import os
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers.cross_encoder import CrossEncoder

MODEL = os.getenv("MODEL", "BAAI/bge-reranker-v2-m3")
DEFAULT_TOP_K = int(os.getenv("TOP_K", "5"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))

api = FastAPI()
model = CrossEncoder(MODEL, max_length=MAX_LENGTH)


class RerankRequest(BaseModel):
    query: str
    docs: List[str]
    top_k: Optional[int] = None


@api.post("/rerank")
def rerank(req: RerankRequest):
    if not req.docs:
        return {"results": []}
    pairs = [(req.query, d) for d in req.docs]
    scores = model.predict(pairs, batch_size=BATCH_SIZE).tolist()
    order = sorted(range(len(req.docs)), key=lambda i: scores[i], reverse=True)
    k = req.top_k or DEFAULT_TOP_K
    return {"results": [{"index": i, "score": float(scores[i])} for i in order[:k]]}


@api.get("/health")
def health():
    return {"status": "ok", "model": MODEL}
