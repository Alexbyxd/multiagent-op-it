"""Tool para búsqueda de tickets."""
import logging
from typing import Optional

from langchain_core.tools import tool

from src.db.queries import search_tickets as db_search_tickets
from src.exceptions import DatabaseError
from src.utils.sanitize import sanitize_text

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
            # Sanitizar datos del ticket
            title = sanitize_text(ticket.get("title", ""))
            description = sanitize_text(ticket.get("description", ""))
            error_code = sanitize_text(ticket.get("error_code", ""))
            solution = sanitize_text(ticket.get("solution", ""))
            severity = ticket.get("severity", "unknown")
            
            severity_emoji = {
                "critical": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢"
            }.get(severity, "⚪")
            
            formatted.append(
                f"--- Ticket #{ticket['id']} {severity_emoji} {severity.upper()} ---\n"
                f"Título: {title}\n"
                f"Código: {error_code or 'N/A'}\n"
                f"Descripción: {description}\n"
                f"Solución: {solution}\n"
            )
        
        return "\n\n".join(formatted)
        
    except DatabaseError as e:
        logger.error(f"Error en search_tickets: {e}")
        return f"Error consultando tickets: {str(e)}"