"""Punto de entrada del sistema MASO."""
import argparse
import logging
import sys

from src.config import settings
from src.db.setup_db import setup_database
from src.vectorstore.ingest import ingest_documents
from src.graph.maso_graph import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup():
    """Inicializa el sistema."""
    logger.info("Inicializando MASO...")
    
    if not settings.DB_PATH.exists():
        logger.info("Creando base de datos...")
        setup_database()
    else:
        logger.info("Base de datos ya existe")
    
    if settings.DOCUMENTS_DIR.exists():
        logger.info("Ingiiriendo documentos...")
        ingest_documents()
    else:
        logger.warning(f"Directorio de documentos no existe: {settings.DOCUMENTS_DIR}")
    
    logger.info("Setup completado")


def chat(user_level: str = "basic"):
    """Modo interactivo."""
    print("=" * 50)
    print("MASO - Asistente de Soporte Técnico")
    print("Escribe 'exit' para salir")
    print("=" * 50)
    
    while True:
        try:
            query = input("\n> ")
            
            if query.lower() in ["exit", "quit", "salir"]:
                print("¡Hasta luego!")
                break
            
            if not query.strip():
                continue
            
            response = run_agent(query, user_level)
            print(f"\n{response}\n")
            
        except KeyboardInterrupt:
            print("\n¡Hasta luego!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}")


def main():
    """Main entrypoint."""
    parser = argparse.ArgumentParser(description="MASO - Asistente de Soporte Técnico")
    parser.add_argument("--setup", action="store_true", help="Ejecutar setup inicial")
    parser.add_argument("--level", choices=["basic", "advanced", "admin"], default="basic", help="Nivel de usuario")
    
    args = parser.parse_args()
    
    if args.setup:
        setup()
    else:
        chat(args.level)


if __name__ == "__main__":
    main()