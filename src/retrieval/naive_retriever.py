import json
import math
from typing import List, Optional
from src.config import settings
from src.retrieval.base import BaseRetriever, RetrievedChunk
from src.ingestion.indexer import get_dense_embeddings

class NaiveRetriever(BaseRetriever):
    """
    Baseline Naive RAG: Single-pass Dense Vector Search using Cosine Similarity.
    """
    def __init__(self):
        super().__init__(name="naive_dense")
        self.index_file = settings.data_dir / "dense_index.json"
        with open(self.index_file, "r", encoding="utf-8") as f:
            self.docs = json.load(f)

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a)) or 1.0
        norm_b = math.sqrt(sum(b * b for b in vec_b)) or 1.0
        return dot / (norm_a * norm_b)

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        ticker_filter: Optional[str] = None
    ) -> List[RetrievedChunk]:
        # Embed query
        query_vec = get_dense_embeddings([query])[0]
        
        scored_docs = []
        for doc in self.docs:
            if ticker_filter and doc.get("ticker") != ticker_filter:
                continue
            sim = self._cosine_similarity(query_vec, doc["embedding"])
            scored_docs.append((sim, doc))

        # Sort descending by cosine similarity
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, doc in scored_docs[:top_k]:
            results.append(RetrievedChunk(
                chunk_id=doc["chunk_id"],
                content=doc["content"],
                score=round(float(score), 4),
                retriever_origin=self.name,
                metadata={
                    "ticker": doc.get("ticker"),
                    "section": doc.get("section"),
                    "is_table": doc.get("is_table")
                }
            ))
            
        return results
