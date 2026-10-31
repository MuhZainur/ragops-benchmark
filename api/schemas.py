from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

# Validated Strategy Types
RAGStrategy = Literal["naive", "bm25", "hybrid_rrf", "hyde", "hierarchical", "hybrid_rerank"]
TickerSymbol = Literal["NVDA", "AAPL", "MSFT"]

class QueryRequest(BaseModel):
    query: str = Field(
        ..., 
        min_length=3, 
        max_length=1000, 
        description="Financial question regarding SEC 10-K filings (sanitized length)"
    )
    strategy: RAGStrategy = Field(
        "hybrid_rerank", 
        description="Retrieval strategy: 'naive', 'bm25', 'hybrid_rrf', 'hyde', 'hierarchical', or 'hybrid_rerank'"
    )
    top_k: int = Field(
        5, 
        ge=1, 
        le=20, 
        description="Final number of chunks to feed into generator (bounded to prevent token exhaustion)"
    )
    ticker_filter: Optional[TickerSymbol] = Field(
        None, 
        description="Filter by ticker: 'NVDA', 'AAPL', 'MSFT', or None for all filings"
    )

class RetrievedChunkSchema(BaseModel):
    chunk_id: str
    content: str
    score: float = Field(..., ge=0.0)
    retriever_origin: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    query: str
    strategy: RAGStrategy
    answer: str
    citations: List[str]
    retrieved_chunks: List[RetrievedChunkSchema]
    latency_breakdown: Dict[str, float]  # retrieval_ms, rerank_ms, generation_ms, total_ms
    estimated_cost_usd: float

class MetricDetail(BaseModel):
    faithfulness: float = Field(..., ge=0.0, le=1.0)
    answer_relevance: float = Field(..., ge=0.0, le=1.0)
    context_precision: float = Field(..., ge=0.0, le=1.0)
    context_recall: float = Field(..., ge=0.0, le=1.0)
    mrr_at_5: float = Field(..., ge=0.0, le=1.0)
    hit_rate_at_5: float = Field(..., ge=0.0, le=1.0)
    p95_latency_ms: float = Field(..., ge=0.0)
    cost_per_1k_queries_usd: float = Field(..., ge=0.0)

class BenchmarkSummaryResponse(BaseModel):
    timestamp: str
    dataset_size: int
    strategies: Dict[str, MetricDetail]
