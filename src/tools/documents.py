"""Tool para búsqueda en documentos técnicos."""
import logging
from typing import Optional

from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import settings
from src.vectorstore.qdrant_client import QdrantManager
from src.exceptions import VectorStoreError, LLMError
from src.utils.sanitize import sanitize_text

logger = logging.getLogger(__name__)


@tool
def search_documents(query: str, limit: int = 5) -> str:
    """Busca en la documentación técnica.
    
    Úsala cuando el usuario pregunte por:
    - Cómo configurar algo
    - Qué significa un error
    - Dónde está un manual o guía
    - Información sobre servidores, redes, seguridad
    
    Args:
        query: Texto de búsqueda o pregunta.
        limit: Número máximo de resultados (default 5).
    
    Returns:
        String con los documentos encontrados.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
            request_timeout=settings.embedding_timeout,
        )
        
        query_vector = embeddings.embed_query(query)
        
        qdrant = QdrantManager()
        results = qdrant.search(query_vector, limit=limit)
        
        if not results:
            return "No se encontró documentación relevante para la consulta."
        
        formatted = []
        for i, result in enumerate(results, 1):
            # Sanitizar texto del documento
            text = sanitize_text(result.get("text", ""))
            source = sanitize_text(result.get("source", ""))
            page = result.get("page", 0)
            score = result.get("score", 0.0)
            
            formatted.append(
                f"--- Resultado {i} ---\n"
                f"Fuente: {source} (pág. {page})\n"
                f"Relevancia: {score:.2%}\n"
                f"\n{text}\n"
            )
        
        return "\n\n".join(formatted)
        
    except Exception as e:
        logger.error(f"Error en search_documents: {e}")
        return f"Error buscando documentos: {str(e)}"