"""Punto de entrada del sistema MASO."""
import argparse
import logging
import sys

from src.config import settings
from src.db.setup_db import setup_database
from src.vectorstore.ingest import ingest_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup() -> None:
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


def chat(user_level: str = "basic") -> None:
    """Modo interactivo con REPL mejorada.

    Detecta si la entrada estándar es un TTY y usa la REPL con
    Rich/Prompt Toolkit o el modo simple según corresponda.

    Args:
        user_level: Nivel de usuario (basic, advanced, admin).
    """
    if not sys.stdin.isatty():
        # Non-TTY: fallback a modo simple (pipes, redirección)
        from src.cli.repl import run_simple_mode

        run_simple_mode(user_level)
    else:
        # TTY: REPL con Rich y Prompt Toolkit
        from src.cli.repl import MASORePL

        repl = MASORePL(user_level=user_level)
        repl.run()


def main() -> None:
    """Main entrypoint."""
    parser = argparse.ArgumentParser(
        description="MASO - Asistente de Soporte Técnico",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Ejecutar setup inicial",
    )
    parser.add_argument(
        "--level",
        choices=["basic", "advanced", "admin"],
        default="basic",
        help="Nivel de usuario",
    )

    args = parser.parse_args()

    if args.setup:
        setup()
    else:
        chat(args.level)


if __name__ == "__main__":
    main()
