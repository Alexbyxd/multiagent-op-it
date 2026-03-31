"""Configuración centralizada del proyecto MASO."""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Configuración del proyecto."""
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DB_PATH: Path = DATA_DIR / "tickets.db"
    QDRANT_PATH: str = str(DATA_DIR / "qdrant")
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    STATUS_FILE: Path = DATA_DIR / "status.json"
    
    # Google Gemini
    google_api_key: str
    
    # Embeddings
    embedding_model: str = "gemini-embedding-001"
    chunk_size: int = 80
    chunk_overlap: int = 10
    
    # Modelos
    router_model: str = "gemini-2.5-flash"
    synthesizer_model: str = "gemini-2.5-pro"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
