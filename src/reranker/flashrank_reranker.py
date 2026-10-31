import logging
from typing import List
from flashrank import Ranker, RerankRequest
from src.config import settings
from src.retrieval.base import RetrievedChunk

logger = logging.getLogger("FlashRankReranker")

class FlashRankReranker:
    """
    Ultra-lightweight Cross-Encoder Re-Ranker utilizing FlashRank.
    Computes inter-token cross-attention between Query and Document Chunks.
    Runs on CPU with ~50MB RAM and 12-25ms latency.
    """
    def __init__(self):
        try:
            self.ranker = Ranker(model_name=settings.flashrank_model_name, cache_dir="./.flashrank_cache")
            self.is_active = True
            logger.info(f"✅ FlashRank initialized with model: {settings.flashrank_model_name}")
        except Exception as e:
            logger.warning(f"FlashRank init failed: {e}. Running in passthrough mode.")
            self.is_active = False

    def rerank(
        self, 
        query: str, 
        candidates: List[RetrievedChunk], 
        top_n: int = 5,
        threshold: float = 0.35
    ) -> List[RetrievedChunk]:
        if not candidates or not self.is_active:
            return candidates[:top_n]

        # Format candidates for FlashRank
        passages = [
            {
                "id": chunk.chunk_id,
                "text": chunk.content,
                "meta": {**chunk.metadata, "orig_score": chunk.score, "orig_origin": chunk.retriever_origin}
            }
            for chunk in candidates
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        ranked_results = self.ranker.rerank(rerank_req)

        reranked_chunks: List[RetrievedChunk] = []
        for item in ranked_results:
            score = round(float(item["score"]), 4)
            
            # Apply relevance cutoff threshold to drop noise
            if score < threshold and len(reranked_chunks) >= 2:
                continue
                
            reranked_chunks.append(RetrievedChunk(
                chunk_id=item["id"],
                content=item["text"],
                score=score,
                retriever_origin=f"{item['meta'].get('orig_origin', 'unknown')} -> flashrank_reranked",
                metadata={
                    **item["meta"],
                    "cross_encoder_score": score,
                    "rank_improvement": True
                }
            ))

        return reranked_chunks[:top_n]
