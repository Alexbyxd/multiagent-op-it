"""Cliente para Qdrant vector store."""
from pathlib import Path
from typing import Optional
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

from src.config import settings
from src.exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class QdrantManager:
    """Gestor del vector store Qdrant."""
    
    COLLECTION_NAME = "technical_docs"
    
    def __init__(self, path: Optional[str] = None):
        """Inicializa el cliente Qdrant.
        
        Args:
            path: Path al directorio de Qdrant. Si es None, usa settings.
        """
        self.path = path or settings.QDRANT_PATH
        self.client = QdrantClient(path=self.path)
    
    def create_collection(self, vector_size: int = 768) -> None:
        """Crea la colección de documentos.
        
        Args:
            vector_size: Dimensión del vector de embedding.
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.COLLECTION_NAME not in collection_names:
                self.client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Colección {self.COLLECTION_NAME} creada")
            else:
                logger.info(f"Colección {self.COLLECTION_NAME} ya existe")
                
        except Exception as e:
            raise VectorStoreError(f"Error creando colección: {e}") from e
    
    def add_documents(self, documents: list[dict], vectors: list[list[float]]) -> None:
        """Añade documentos al vector store.
        
        Args:
            documents: Lista de documentos con id, text, source, page.
            vectors: Lista de vectores de embedding.
        """
        try:
            points = [
                PointStruct(
                    id=doc["id"],
                    vector=vector,
                    payload={
                        "text": doc["text"],
                        "source": doc.get("source", ""),
                        "page": doc.get("page", 0)
                    }
                )
                for doc, vector in zip(documents, vectors)
            ]
            
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points
            )
            logger.info(f"Añadidos {len(points)} documentos")
            
        except Exception as e:
            raise VectorStoreError(f"Error añadiendo documentos: {e}") from e
    
    def search(self, query_vector: list[float], limit: int = 5) -> list[dict]:
        """Busca documentos similares.
        
        Args:
            query_vector: Vector de embedding de la query.
            limit: Número máximo de resultados.
        
        Returns:
            Lista de documentos similares.
        """
        try:
            results = self.client.search(
                collection_name=self.COLLECTION_NAME,
                query_vector=query_vector,
                limit=limit
            )
            
            return [
                {
                    "id": r.id,
                    "text": r.payload["text"],
                    "source": r.payload.get("source", ""),
                    "page": r.payload.get("page", 0),
                    "score": r.score
                }
                for r in results
            ]
            
        except Exception as e:
            raise VectorStoreError(f"Error en búsqueda: {e}") from e
