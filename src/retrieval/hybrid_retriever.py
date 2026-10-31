from typing import List, Optional, Dict
from src.config import settings
from src.retrieval.base import BaseRetriever, RetrievedChunk
from src.retrieval.naive_retriever import NaiveRetriever
from src.retrieval.sparse_retriever import SparseBM25Retriever

class HybridRRFRetriever(BaseRetriever):
    """
    Advanced Hybrid Search: Fuses Dense Cosine Search and Sparse BM25 using
    Reciprocal Rank Fusion (RRF) with constant k=60.
    """
    def __init__(self):
        super().__init__(name="hybrid_rrf")
        self.dense_retriever = NaiveRetriever()
        self.sparse_retriever = SparseBM25Retriever()
        self.k = settings.rrf_k

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        ticker_filter: Optional[str] = None
    ) -> List[RetrievedChunk]:
        # Broad recall candidate pool
        broad_pool = max(top_k * 3, settings.retrieval_top_k)
        
        dense_results = self.dense_retriever.retrieve(query, top_k=broad_pool, ticker_filter=ticker_filter)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=broad_pool, ticker_filter=ticker_filter)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, RetrievedChunk] = {}

        # 1. Score dense ranks
        for rank, item in enumerate(dense_results, start=1):
            cid = item.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.k + rank))
            chunk_map[cid] = item

        # 2. Score sparse ranks
        for rank, item in enumerate(sparse_results, start=1):
            cid = item.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.k + rank))
            if cid not in chunk_map:
                chunk_map[cid] = item

        # 3. Sort by combined RRF score descending
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        max_rrf = sorted_chunks[0][1] if sorted_chunks else 1.0
        
        final_results = []
        for cid, score in sorted_chunks[:top_k]:
            orig = chunk_map[cid]
            norm_score = round(float(score / max_rrf), 4) if max_rrf > 0 else 0.0
            
            final_results.append(RetrievedChunk(
                chunk_id=orig.chunk_id,
                content=orig.content,
                score=norm_score,
                retriever_origin=self.name,
                metadata={
                    **orig.metadata,
                    "rrf_raw_score": round(float(score), 5)
                }
            ))
            
        return final_results
