"""Tool para búsqueda de tickets."""
import logging
from typing import Optional

from langchain_core.tools import tool

from src.db.queries import search_tickets as db_search_tickets
from src.exceptions import DatabaseError

logger = logging.getLogger(__name__)


@tool
def search_tickets(query: str, limit: int = 5) -> str:
    """Busca en el historial de tickets de soporte técnico.
    
    Úsala cuando el usuario pregunte por:
    - Problemas pasados similares
    - Cómo se resolvió un error antes
    - Historial de incidentes
    - Errores recurrentes
    
    Args:
        query: Descripción del problema o error.
        limit: Número máximo de resultados (default 5).
    
    Returns:
        String con los tickets encontrados.
    """
    try:
        results = db_search_tickets(query, limit=limit)
        
        if not results:
            return "No se encontró historial de tickets relacionado."
        
        formatted = []
        for ticket in results:
            severity_emoji = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢"
            }.get(ticket["severity"], "⚪")
            
            formatted.append(
                f"--- Ticket #{ticket['id']} {severity_emoji} {ticket['severity'].upper()} ---\n"
                f"Título: {ticket['title']}\n"
                f"Código: {ticket['error_code'] or 'N/A'}\n"
                f"Descripción: {ticket['description']}\n"
                f"Solución: {ticket['solution']}\n"
            )
        
        return "\n\n".join(formatted)
        
    except DatabaseError as e:
        logger.error(f"Error en search_tickets: {e}")
        return f"Error consultando tickets: {str(e)}"