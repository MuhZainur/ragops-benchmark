import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

# Base Directory Resolution (Cross-Platform)
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # LLM & Embedding API
    openrouter_api_key: str = Field(default="", env="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="deepseek/deepseek-chat", env="OPENROUTER_MODEL")
    openrouter_embedding_model: str = Field(default="openai/text-embedding-3-small", env="OPENROUTER_EMBEDDING_MODEL")
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Vector DB (Qdrant)
    qdrant_host: str = Field(default="localhost", env="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, env="QDRANT_PORT")
    qdrant_api_key: str = Field(default="", env="QDRANT_API_KEY")
    qdrant_collection_name: str = "sec_10k_filings"
    qdrant_storage_path: Path = BASE_DIR / "data" / "qdrant_storage"

    # FlashRank Re-ranker
    flashrank_model_name: str = Field(default="ms-marco-TinyBERT-L-2-v2", env="FLASHRANK_MODEL")

    # Hyperparameters
    chunk_size_naive: int = 512
    chunk_overlap: int = 64
    child_chunk_size: int = 128
    parent_chunk_size: int = 1024
    
    bm25_k1: float = 1.6
    bm25_b: float = 0.75
    rrf_k: int = 60
    
    retrieval_top_k: int = 20
    final_top_k: int = 5
    rerank_score_threshold: float = 0.35

    # Directory Paths
    data_dir: Path = BASE_DIR / "data"
    raw_data_dir: Path = BASE_DIR / "data" / "raw"
    processed_data_dir: Path = BASE_DIR / "data" / "processed"
    golden_eval_dataset_path: Path = BASE_DIR / "data" / "golden_eval_dataset.json"

    # API Server Settings
    port: int = Field(default=8000, env="PORT")
    host: str = Field(default="0.0.0.0", env="HOST")
    environment: str = Field(default="development", env="ENVIRONMENT")

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
