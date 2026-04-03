"""Tool para consultar estado de servicios."""
import json
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from src.config import settings
from src.exceptions import ToolError


@tool
def check_service_status(service_name: str) -> str:
    """Consulta el estado de un servicio específico.
    
    Args:
        service_name: Nombre del servicio a consultar.
    
    Returns:
        String con el estado del servicio.
    """
    try:
        status_file = settings.STATUS_FILE
        
        if not status_file.exists():
            raise ToolError(f"Archivo de estado no encontrado: {status_file}")
        
        with open(status_file, "r") as f:
            data = json.load(f)
        
        for service in data.get("services", []):
            if service["name"].lower() == service_name.lower():
                return _format_service_status(service)
        
        available = [s["name"] for s in data.get("services", [])]
        return f"Servicio '{service_name}' no encontrado. Servicios disponibles: {', '.join(available)}"
    
    except json.JSONDecodeError as e:
        raise ToolError(f"Error parseando archivo de estado: {e}") from e
    except Exception as e:
        raise ToolError(f"Error consultando estado: {e}") from e


def _format_service_status(service: dict) -> str:
    """Formatea el estado de un servicio."""
    status_emoji = "✅" if service["status"] == "UP" else "❌"
    
    result = f"""
{status_emoji} **{service['display_name']}** ({service['name']})
- Estado: {service['status']}
- Uptime: {service['uptime']}
- Última verificación: {service['last_check']}
"""
    
    if "response_time_ms" in service:
        result += f"- Tiempo de respuesta: {service['response_time_ms']}ms\n"
    
    if "last_failure" in service:
        result += f"- Última falla: {service['last_failure']}\n"
        if "error" in service:
            result += f"- Error: {service['error']}\n"
    
    return result.strip()