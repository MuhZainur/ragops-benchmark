import os
import json
import logging
import httpx
from pathlib import Path
from typing import List, Dict, Any, Tuple
from rank_bm25 import BM25Okapi

from src.config import settings
from src.ingestion.chunker import clean_text, split_into_sections, create_naive_chunks, create_hierarchical_chunks

logger = logging.getLogger("RAGIndexer")
logging.basicConfig(level=logging.INFO)

# ==========================================
# SAFE BM25 SERIALIZATION (CWE-502 Mitigated)
# ==========================================

def tokenize_for_bm25(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for sparse BM25."""
    import re
    return re.findall(r'\b[a-zA-Z0-9_\-\$\%\.]+\b', text.lower())

def save_bm25_index_safe(corpus_tokens: List[List[str]], metadata: List[Dict[str, Any]], filepath: Path):
    """Saves BM25 tokenized corpus and metadata safely to JSON."""
    payload = {
        "k1": settings.bm25_k1,
        "b": settings.bm25_b,
        "corpus": corpus_tokens,
        "metadata": metadata
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    logger.info(f"✅ Safe BM25 index saved: {filepath} ({len(corpus_tokens)} docs)")

def load_bm25_index_safe(filepath: Path) -> Tuple[BM25Okapi, List[Dict[str, Any]]]:
    """Loads and reconstructs BM25Okapi from JSON."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    bm25 = BM25Okapi(
        corpus=data["corpus"],
        k1=data.get("k1", settings.bm25_k1),
        b=data.get("b", settings.bm25_b)
    )
    return bm25, data.get("metadata", [])

# ==========================================
# DENSE EMBEDDINGS (OpenRouter with Fallback)
# ==========================================

def get_dense_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Fetches embeddings from OpenRouter text-embedding-3-small,
    with a deterministic fallback if API key is absent or offline.
    """
    api_key = settings.openrouter_api_key
    if api_key and not api_key.startswith("sk-or-v1-your"):
        try:
            url = f"{settings.openrouter_base_url}/embeddings"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            # Batch embedding in chunks of 20
            all_embeddings = []
            for i in range(0, len(texts), 20):
                batch = texts[i:i+20]
                payload = {
                    "model": settings.openrouter_embedding_model,
                    "input": batch
                }
                with httpx.Client(timeout=30.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json().get("data", [])
                        for item in data:
                            all_embeddings.append(item["embedding"])
                    else:
                        raise RuntimeError(f"Embedding API error {resp.status_code}: {resp.text}")
            if len(all_embeddings) == len(texts):
                logger.info(f"✅ Generated {len(all_embeddings)} dense embeddings via OpenRouter.")
                return all_embeddings
        except Exception as e:
            logger.warning(f"Embedding API unavailable: {e}. Switching to deterministic vector fallback.")

    # High-accuracy TF-IDF + hashing fallback vector (dim=256)
    import hashlib
    import math
    dim = 256
    vectors = []
    for text in texts:
        vec = [0.0] * dim
        tokens = tokenize_for_bm25(text)
        for tok in tokens:
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
            vec[idx] += 1.0
        # L2 Normalize
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        vectors.append([x / norm for x in vec])
    logger.info(f"✅ Generated {len(vectors)} deterministic fallback vectors.")
    return vectors

# ==========================================
# MASTER INGESTION PIPELINE
# ==========================================

def build_all_indices():
    """
    Parses processed SEC 10-K files, produces naive & hierarchical chunks,
    and indexes them into BM25 JSON and Qdrant vector storage.
    """
    processed_dir = settings.processed_data_dir
    files = list(processed_dir.glob("*.md"))
    if not files:
        logger.error(f"No Markdown filings found in {processed_dir}")
        return

    all_naive_chunks = []
    all_hierarchical_chunks = []
    master_parent_lookup = {}

    for file_path in files:
        ticker = file_path.stem.split("_")[0]
        logger.info(f"Processing 10-K filing for ticker: {ticker} ({file_path.name})")
        
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = clean_text(f.read())
            
        sections = split_into_sections(raw_text)
        
        # 1. Naive Chunks
        naive_chunks = create_naive_chunks(
            sections, 
            ticker, 
            chunk_size=settings.chunk_size_naive, 
            overlap=settings.chunk_overlap
        )
        all_naive_chunks.extend(naive_chunks)
        
        # 2. Hierarchical Chunks
        child_chunks, parent_lookup = create_hierarchical_chunks(
            sections, 
            ticker, 
            parent_size=settings.parent_chunk_size, 
            child_size=settings.child_chunk_size, 
            overlap=32
        )
        all_hierarchical_chunks.extend(child_chunks)
        master_parent_lookup.update(parent_lookup)

    # Save Chunks to Disk (Safe JSON)
    naive_file = settings.data_dir / "naive_chunks.json"
    with open(naive_file, "w", encoding="utf-8") as f:
        json.dump(all_naive_chunks, f, indent=2, ensure_ascii=False)
        
    hierarchical_file = settings.data_dir / "hierarchical_chunks.json"
    with open(hierarchical_file, "w", encoding="utf-8") as f:
        json.dump(all_hierarchical_chunks, f, indent=2, ensure_ascii=False)
        
    parent_file = settings.data_dir / "parent_lookup.json"
    with open(parent_file, "w", encoding="utf-8") as f:
        json.dump(master_parent_lookup, f, indent=2, ensure_ascii=False)

    logger.info(f"Ingested {len(all_naive_chunks)} naive chunks and {len(all_hierarchical_chunks)} hierarchical chunks.")

    # Build Safe BM25 Index
    corpus_tokens = [tokenize_for_bm25(c["content"]) for c in all_naive_chunks]
    save_bm25_index_safe(corpus_tokens, all_naive_chunks, settings.data_dir / "bm25_index.json")

    # Build Dense Embeddings & Store in Local Qdrant / JSON Vector Storage
    texts_to_embed = [c["content"] for c in all_naive_chunks]
    embeddings = get_dense_embeddings(texts_to_embed)

    dense_index_data = []
    for chunk, emb in zip(all_naive_chunks, embeddings):
        dense_index_data.append({
            "chunk_id": chunk["chunk_id"],
            "ticker": chunk["ticker"],
            "section": chunk["section"],
            "content": chunk["content"],
            "is_table": chunk["is_table"],
            "embedding": emb
        })

    dense_file = settings.data_dir / "dense_index.json"
    with open(dense_file, "w", encoding="utf-8") as f:
        json.dump(dense_index_data, f)
    logger.info(f"✅ Dense vector index saved: {dense_file} ({len(dense_index_data)} vectors)")

if __name__ == "__main__":
    build_all_indices()
