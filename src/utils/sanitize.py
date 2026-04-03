"""Utilidades de sanitización para outputs."""
import html
import re
from typing import Any


def sanitize_text(text: str, max_length: int = 10000) -> str:
    """Sanitiza texto para prevenir XSS y otros ataques.
    
    Args:
        text: Texto a sanitizar.
        max_length: Longitud máxima del texto.
    
    Returns:
        Texto sanitizado.
    """
    if not text:
        return ""
    
    # Escapar HTML entities
    sanitized = html.escape(text)
    
    # Remover patrones potencialmente peligrosos
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                  # JavaScript protocol
        r'on\w+\s*=',                    # Event handlers
        r'<iframe[^>]*>.*?</iframe>',   # Iframes
    ]
    
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    # Truncar si es muy largo
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."
    
    return sanitized


def sanitize_ticket_data(ticket: dict[str, Any]) -> dict[str, Any]:
    """Sanitiza datos de un ticket.
    
    Args:
        ticket: Diccionario con datos del ticket.
    
    Returns:
        Ticket con datos sanitizados.
    """
    sanitized = {}
    fields_to_sanitize = ['title', 'description', 'error_code', 'solution']
    
    for key, value in ticket.items():
        if key in fields_to_sanitize and isinstance(value, str):
            sanitized[key] = sanitize_text(value)
        else:
            sanitized[key] = value
    
    return sanitized


def sanitize_document_result(result: dict[str, Any]) -> dict[str, Any]:
    """Sanitiza resultado de búsqueda de documentos.
    
    Args:
        result: Diccionario con datos del documento.
    
    Returns:
        Documento con datos sanitizados.
    """
    sanitized = {}
    fields_to_sanitize = ['text', 'source']
    
    for key, value in result.items():
        if key in fields_to_sanitize and isinstance(value, str):
            sanitized[key] = sanitize_text(value)
        else:
            sanitized[key] = value
    
    return sanitized