from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    score: float = Field(default=0.0)
    retriever_origin: str  # 'dense', 'bm25', 'rrf_hybrid', 'hyde', 'parent_doc'
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BaseRetriever(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        ticker_filter: Optional[str] = None
    ) -> List[RetrievedChunk]:
        """
        Abstract method to retrieve candidate chunks based on a query.
        Must be implemented by all retrieval strategies.
        """
        pass
