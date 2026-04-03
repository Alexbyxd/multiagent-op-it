"""Ingesta de documentos al vector store."""
import json
import logging
from pathlib import Path
from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.vectorstore.qdrant_client import QdrantManager
from src.exceptions import VectorStoreError, LLMError

logger = logging.getLogger(__name__)


def get_embedding_model():
    """Obtiene el modelo de embedding."""
    try:
        return GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key
        )
    except Exception as e:
        raise LLMError(f"Error inicializando embeddings: {e}") from e


def load_document(file_path: Path) -> list[dict]:
    """Carga un documento y lo convierte en chunks.
    
    Args:
        file_path: Path al documento.
    
    Returns:
        Lista de chunks con metadatos.
    """
    if file_path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
    elif file_path.suffix.lower() in [".md", ".txt"]:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
    else:
        logger.warning(f"Tipo de archivo no soportado: {file_path.suffix}")
        return []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=lambda x: len(x.split())
    )
    
    chunks = splitter.split_documents(docs)
    
    return [
        {
            "text": chunk.page_content,
            "source": file_path.name,
            "page": chunk.metadata.get("page", 0)
        }
        for chunk in chunks
    ]


def ingest_documents(documents_dir: Optional[Path] = None) -> None:
    """Ingiere todos los documentos al vector store.
    
    Args:
        documents_dir: Directorio con documentos. Si es None, usa settings.
    """
    dir_path = documents_dir or settings.DOCUMENTS_DIR
    
    if not dir_path.exists():
        logger.warning(f"Directorio de documentos no existe: {dir_path}")
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Creado directorio: {dir_path}")
        return
    
    qdrant = QdrantManager()
    qdrant.create_collection()
    
    embeddings = get_embedding_model()
    
    all_chunks = []
    for file_path in dir_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in [".pdf", ".md", ".txt"]:
            logger.info(f"Procesando: {file_path.name}")
            chunks = load_document(file_path)
            all_chunks.extend(chunks)
    
    if not all_chunks:
        logger.warning("No hay documentos para ingestarr")
        return
    
    logger.info(f"Generando embeddings para {len(all_chunks)} chunks...")
    texts = [chunk["text"] for chunk in all_chunks]
    vectors = embeddings.embed_documents(texts)
    
    documents_with_id = [
        {**chunk, "id": i}
        for i, chunk in enumerate(all_chunks)
    ]
    
    qdrant.add_documents(documents_with_id, vectors)
    
    logger.info(f"Ingesta completada: {len(all_chunks)} chunks")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_documents()