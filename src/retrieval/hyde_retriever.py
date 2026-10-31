import logging
import httpx
from typing import List, Optional
from src.config import settings
from src.retrieval.base import BaseRetriever, RetrievedChunk
from src.retrieval.naive_retriever import NaiveRetriever

logger = logging.getLogger("HyDERetriever")

class HyDERetriever(BaseRetriever):
    """
    Hypothetical Document Embeddings (HyDE):
    Uses an LLM to generate a hypothetical SEC 10-K answer paragraph,
    then embeds and searches using the hypothetical text rather than the user's raw query.
    """
    def __init__(self):
        super().__init__(name="hyde_query_transform")
        self.dense_retriever = NaiveRetriever()

    def _generate_hypothetical_document(self, query: str) -> str:
        prompt = (
            f"Please write a hypothetical paragraph from a corporate SEC Form 10-K annual report "
            f"that provides a formal, comprehensive answer to this question: '{query}'. "
            f"Use formal financial terminology, realistic balance sheet / revenue figures, and item section headers. "
            f"Do not write conversational filler; output only the document text."
        )
        api_key = settings.openrouter_api_key
        if api_key and not api_key.startswith("sk-or-v1-your"):
            try:
                url = f"{settings.openrouter_base_url}/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": settings.openrouter_model,
                    "messages": [
                        {"role": "system", "content": "You are an SEC financial reporting specialist."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 250
                }
                with httpx.Client(timeout=20.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        hypo_doc = resp.json()["choices"][0]["message"]["content"].strip()
                        return hypo_doc
            except Exception as e:
                logger.warning(f"HyDE LLM generation failed: {e}. Falling back to augmented query.")

        return f"Item 7 Management Discussion and Analysis SEC 10-K Report. {query} Total revenues, segment performance, operating metrics, and capital expenditures."

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        ticker_filter: Optional[str] = None
    ) -> List[RetrievedChunk]:
        # Generate hypothetical document
        hypothetical_doc = self._generate_hypothetical_document(query)
        logger.info(f"Generated HyDE Doc: {hypothetical_doc[:120]}...")
        
        # Search vector space using the hypothetical document
        results = self.dense_retriever.retrieve(
            hypothetical_doc, 
            top_k=top_k, 
            ticker_filter=ticker_filter
        )
        
        # Mark origin
        for r in results:
            r.retriever_origin = self.name
            r.metadata["hyde_sample"] = hypothetical_doc[:100]
            
        return results
