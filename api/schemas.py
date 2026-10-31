from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    query: str = Field(..., description="Financial question regarding SEC 10-K filings")
    strategy: str = Field("hybrid_rerank", description="Retrieval strategy: 'naive', 'bm25', 'hybrid_rrf', 'hyde', 'hierarchical', or 'hybrid_rerank'")
    top_k: int = Field(5, description="Final number of chunks to feed into generator")
    ticker_filter: Optional[str] = Field(None, description="Filter by ticker: 'NVDA', 'AAPL', 'MSFT', or None for all")

class RetrievedChunkSchema(BaseModel):
    chunk_id: str
    content: str
    score: float
    retriever_origin: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    query: str
    strategy: str
    answer: str
    citations: List[str]
    retrieved_chunks: List[RetrievedChunkSchema]
    latency_breakdown: Dict[str, float]  # retrieval_ms, rerank_ms, generation_ms, total_ms
    estimated_cost_usd: float

class MetricDetail(BaseModel):
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    mrr_at_5: float
    hit_rate_at_5: float
    p95_latency_ms: float
    cost_per_1k_queries_usd: float

class BenchmarkSummaryResponse(BaseModel):
    timestamp: str
    dataset_size: int
    strategies: Dict[str, MetricDetail]
