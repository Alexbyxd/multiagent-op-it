"""Consultas a la base de datos de tickets."""
import sqlite3
from typing import Optional
from datetime import datetime

from src.config import settings
from src.exceptions import DatabaseError


def get_connection() -> sqlite3.Connection:
    """Obtiene conexión a la base de datos."""
    try:
        conn = sqlite3.connect(settings.DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        raise DatabaseError(f"Error conectando a la base de datos: {e}") from e


def search_tickets(query: str, limit: int = 5, db_path: Optional[str] = None) -> list[dict]:
    """Busca tickets por query.
    
    Args:
        query: Texto de búsqueda.
        limit: Número máximo de resultados.
        db_path: Path opcional a la DB (para testing).
    
    Returns:
        Lista de diccionarios con los tickets encontrados.
    """
    path = db_path or settings.DB_PATH
    
    try:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        search_term = f"%{query}%"
        cursor.execute("""
            SELECT id, title, description, error_code, solution, severity, created_at, resolved_at
            FROM tickets
            WHERE title LIKE ? OR description LIKE ? OR solution LIKE ?
            ORDER BY 
                CASE severity
                    WHEN 'critical' THEN 1
                    WHEN 'high' THEN 2
                    WHEN 'medium' THEN 3
                    WHEN 'low' THEN 4
                END,
                resolved_at DESC
            LIMIT ?
        """, (search_term, search_term, search_term, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
        
    except sqlite3.Error as e:
        raise DatabaseError(f"Error buscando tickets: {e}") from e
