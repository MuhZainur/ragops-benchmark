from typing import List, Set

def _normalize_id(chunk_id: str) -> str:
    """
    Normalizes chunk ID by extracting the filing ticker and section identifier.
    E.g. 'NVDA_ITEM_7._MANAGEM_CHUNK_004' -> 'NVDA_ITEM_7'
         'NVDA_ITEM_7._MANAGEM_PARENT_004' -> 'NVDA_ITEM_7'
    """
    parts = chunk_id.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}_{parts[2]}"
    return chunk_id

def _is_match(retrieved_id: str, gt_id: str) -> bool:
    if retrieved_id == gt_id:
        return True
    # Section-level equivalence (e.g. child vs parent vs chunk numbering)
    return _normalize_id(retrieved_id) == _normalize_id(gt_id)

def calculate_hit_rate_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 5) -> float:
    """
    Hit Rate@K: Returns 1.0 if at least one ground truth chunk ID is in the top-K retrieved IDs, else 0.0.
    """
    if not ground_truth_ids:
        return 1.0  # Adversarial / unanswerable correctly handled
    
    top_k_retrieved = retrieved_ids[:k]
    for r_id in top_k_retrieved:
        for gt_id in ground_truth_ids:
            if _is_match(r_id, gt_id):
                return 1.0
                
    return 0.0

def calculate_mrr_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 5) -> float:
    """
    Mean Reciprocal Rank (MRR@K): Calculates 1 / rank of the first relevant document found within top-K.
    """
    if not ground_truth_ids:
        return 1.0  # Correct refusal
    
    for rank, doc_id in enumerate(retrieved_ids[:k], start=1):
        for gt_id in ground_truth_ids:
            if _is_match(doc_id, gt_id):
                return 1.0 / float(rank)
            
    return 0.0

def calculate_context_precision_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int = 5) -> float:
    """
    Context Precision@K: Evaluates whether relevant chunks are prioritized at the top of the retrieval list.
    Formula: Sum(Precision@i * relevance_indicator_i) / Total relevant items in top-K.
    """
    if not ground_truth_ids:
        return 1.0
        
    top_k = retrieved_ids[:k]
    cum_precision = 0.0
    relevant_count = 0
    
    for i, doc_id in enumerate(top_k, start=1):
        is_rel = any(_is_match(doc_id, gt_id) for gt_id in ground_truth_ids)
        if is_rel:
            relevant_count += 1
            cum_precision += (relevant_count / float(i))
            
    if relevant_count == 0:
        return 0.0
        
    return cum_precision / float(relevant_count)

def estimate_query_cost(prompt_tokens: int, completion_tokens: int, input_rate_per_m: float = 0.14, output_rate_per_m: float = 0.28) -> float:
    """
    Estimates query cost in USD based on OpenRouter / DeepSeek token usage.
    """
    input_cost = (prompt_tokens / 1_000_000.0) * input_rate_per_m
    output_cost = (completion_tokens / 1_000_000.0) * output_rate_per_m
    return round(input_cost + output_cost, 6)
