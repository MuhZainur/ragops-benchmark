import json
import math
from typing import List, Optional, Dict
from src.config import settings
from src.retrieval.base import BaseRetriever, RetrievedChunk
from src.ingestion.indexer import get_dense_embeddings

class HierarchicalRetriever(BaseRetriever):
    """
    Hierarchical (Small-to-Big) Retrieval:
    Indexes small, focused child chunks (128 words) for high-precision semantic search,
    but retrieves and returns the complete Parent Document (1024 words) to preserve
    full contextual continuity and tabular rows.
    """
    def __init__(self):
        super().__init__(name="hierarchical_parent_doc")
        
        # Load hierarchical children
        h_file = settings.data_dir / "hierarchical_chunks.json"
        with open(h_file, "r", encoding="utf-8") as f:
            self.child_chunks = json.load(f)
            
        # Load parent lookup map
        p_file = settings.data_dir / "parent_lookup.json"
        with open(p_file, "r", encoding="utf-8") as f:
            self.parent_lookup = json.load(f)
            
        self.child_embeddings = None

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
        query_vec = get_dense_embeddings([query])[0]
        
        # Cache child embeddings once
        if self.child_embeddings is None:
            child_texts = [c["content"] for c in self.child_chunks]
            self.child_embeddings = get_dense_embeddings(child_texts)

        scored_children = []
        for child, emb in zip(self.child_chunks, self.child_embeddings):
            if ticker_filter and child.get("ticker") != ticker_filter:
                continue
            sim = self._cosine_similarity(query_vec, emb)
            scored_children.append((sim, child))

        scored_children.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate parents while preserving rank order
        seen_parents = set()
        results = []
        
        for sim, child in scored_children:
            parent_id = child["parent_id"]
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            
            parent_text = self.parent_lookup.get(parent_id, child["content"])
            
            results.append(RetrievedChunk(
                chunk_id=parent_id,
                content=parent_text,
                score=round(float(sim), 4),
                retriever_origin=self.name,
                metadata={
                    "ticker": child.get("ticker"),
                    "section": child.get("section"),
                    "matched_child_id": child["chunk_id"],
                    "is_parent_expanded": True
                }
            ))
            
            if len(results) >= top_k:
                break
                
        return results
