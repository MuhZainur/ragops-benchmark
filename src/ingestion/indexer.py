import json
from pathlib import Path
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.config import settings

def save_bm25_index_safe(corpus: List[List[str]], metadata: List[Dict[str, Any]], filepath: Path):
    """
    Saves BM25 tokenized corpus and metadata safely to JSON (Preventing CWE-502 Pickle RCE).
    """
    payload = {
        "k1": settings.bm25_k1,
        "b": settings.bm25_b,
        "corpus": corpus,
        "metadata": metadata
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

def load_bm25_index_safe(filepath: Path) -> tuple[BM25Okapi, List[Dict[str, Any]]]:
    """
    Safely loads and reconstructs BM25Okapi from JSON without arbitrary code execution risk.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    bm25 = BM25Okapi(
        corpus=data["corpus"],
        k1=data.get("k1", settings.bm25_k1),
        b=data.get("b", settings.bm25_b)
    )
    return bm25, data.get("metadata", [])
