from typing import List, Optional
from src.config import settings
from src.retrieval.base import BaseRetriever, RetrievedChunk
from src.ingestion.indexer import load_bm25_index_safe, tokenize_for_bm25

class SparseBM25Retriever(BaseRetriever):
    """
    Pure Sparse Keyword Search using BM25Okapi over tokenized 10-K filings.
    """
    def __init__(self):
        super().__init__(name="sparse_bm25")
        self.bm25, self.metadata = load_bm25_index_safe(settings.data_dir / "bm25_index.json")

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        ticker_filter: Optional[str] = None
    ) -> List[RetrievedChunk]:
        query_tokens = tokenize_for_bm25(query)
        scores = self.bm25.get_scores(query_tokens)
        
        scored_docs = []
        for idx, score in enumerate(scores):
            meta = self.metadata[idx]
            if ticker_filter and meta.get("ticker") != ticker_filter:
                continue
            scored_docs.append((score, meta))
            
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Max score normalization for readability
        max_score = scored_docs[0][0] if scored_docs and scored_docs[0][0] > 0 else 1.0
        
        results = []
        for score, doc in scored_docs[:top_k]:
            normalized_score = round(float(score / max_score), 4) if max_score > 0 else 0.0
            results.append(RetrievedChunk(
                chunk_id=doc["chunk_id"],
                content=doc["content"],
                score=normalized_score,
                retriever_origin=self.name,
                metadata={
                    "ticker": doc.get("ticker"),
                    "section": doc.get("section"),
                    "is_table": doc.get("is_table"),
                    "raw_bm25_score": round(float(score), 2)
                }
            ))
            
        return results
