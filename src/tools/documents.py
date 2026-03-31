"""Tool para búsqueda en documentos técnicos."""
import logging
from typing import Optional

from langchain_core.tools import tool
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import settings
from src.vectorstore.qdrant_client import QdrantManager
from src.exceptions import VectorStoreError, LLMError

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
            google_api_key=settings.google_api_key
        )
        
        query_vector = embeddings.embed_query(query)
        
        qdrant = QdrantManager()
        results = qdrant.search(query_vector, limit=limit)
        
        if not results:
            return "No se encontró documentación relevante para la consulta."
        
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"--- Resultado {i} ---\n"
                f"Fuente: {result['source']} (pág. {result['page']})\n"
                f"Relevancia: {result['score']:.2%}\n"
                f"\n{result['text']}\n"
            )
        
        return "\n\n".join(formatted)
        
    except Exception as e:
        logger.error(f"Error en search_documents: {e}")
        return f"Error buscando documentos: {str(e)}"