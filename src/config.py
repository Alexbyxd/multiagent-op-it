"""Configuración centralizada del proyecto MASO."""
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuración del proyecto."""
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DB_PATH: Path = DATA_DIR / "tickets.db"
    QDRANT_PATH: str = str(DATA_DIR / "qdrant")
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    STATUS_FILE: Path = DATA_DIR / "status.json"
    
    # Google Gemini (deprecated - usar OpenRouter)
    google_api_key: str = ""
    
    # OpenRouter (proveedor LLM principal)
    openrouter_api_key: str
    openrouter_model: str = "z-ai/glm-4.5-air:free"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_grpc_port: int = 6334
    
    # Database
    database_path: str = "./data/tickets.db"
    
    # Logging
    log_level: str = "INFO"
    
    # Embeddings
    embedding_model: str = "gemini-embedding-001"
    chunk_size: int = 80
    chunk_overlap: int = 10
    
    # Modelos (alias para compatibilidad)
    router_model: str = "z-ai/glm-4.5-air:free"
    synthesizer_model: str = "z-ai/glm-4.5-air:free"

    # Timeouts (segundos)
    router_llm_timeout: int = 15
    synthesizer_llm_timeout: int = 30
    tool_execution_timeout: int = 20
    embedding_timeout: int = 15
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()
