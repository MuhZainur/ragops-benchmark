from typing import List, Set

def calculate_hit_rate_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 5) -> float:
    """
    Hit Rate@K: Returns 1.0 if at least one ground truth chunk ID is in the top-K retrieved IDs, else 0.0.
    """
    if not ground_truth_ids:
        return 0.0
    
    top_k_retrieved = set(retrieved_ids[:k])
    gt_set = set(ground_truth_ids)
    
    return 1.0 if len(top_k_retrieved.intersection(gt_set)) > 0 else 0.0

def calculate_mrr_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 5) -> float:
    """
    Mean Reciprocal Rank (MRR@K): Calculates 1 / rank of the first relevant document found within top-K.
    If no relevant document is found, returns 0.0. Protected against division-by-zero.
    """
    if not ground_truth_ids:
        return 0.0
    
    gt_set = set(ground_truth_ids)
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gt_set:
            return 1.0 / float(rank)
            
    return 0.0

def calculate_context_precision_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 5) -> float:
    """
    Context Precision@K: Evaluates whether relevant chunks are prioritized at the top of the retrieval list.
    Formula: Sum(Precision@i * relevance_indicator_i) / Total relevant items in top-K.
    Protected against division by zero.
    """
    if not ground_truth_ids or not retrieved_ids:
        return 0.0
        
    gt_set = set(ground_truth_ids)
    relevant_hits = 0
    precision_sum = 0.0
    
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in gt_set:
            relevant_hits += 1
            precision_at_i = relevant_hits / float(i)
            precision_sum += precision_at_i
            
    if relevant_hits == 0:
        return 0.0
        
    return precision_sum / float(min(len(gt_set), k))

def estimate_query_cost(prompt_tokens: int, completion_tokens: int, prompt_rate_per_m: float = 0.14, completion_rate_per_m: float = 0.28) -> float:
    """
    Calculates the USD cost of an LLM query given token counts (default DeepSeek rates: $0.14/1M prompt, $0.28/1M completion).
    """
    prompt_cost = (prompt_tokens / 1_000_000.0) * prompt_rate_per_m
    completion_cost = (completion_tokens / 1_000_000.0) * completion_rate_per_m
    return round(prompt_cost + completion_cost, 6)
