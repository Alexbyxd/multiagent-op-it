"""Tool para sugerir acciones basadas en resultados."""
import json
import logging
from typing import Optional

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def suggest_action(tool_results: str) -> str:
    """Sugiere acciones basadas en los resultados de otras herramientas.
    
    Úsala después de ejecutar search_documents o search_tickets
    para proporcionar recomendaciones prácticas al usuario.
    
    Args:
        tool_results: Resultados de otras herramientas en formato JSON o texto.
    
    Returns:
        String con acciones sugeridas.
    """
    try:
        suggestions = []
        
        error_keywords = ["error", "503", "502", "500", "timeout", "failed", "down", "caído"]
        has_error = any(keyword in tool_results.lower() for keyword in error_keywords)
        
        if has_error:
            suggestions.append("📋 Revisa los logs del servicio para más detalles")
            suggestions.append("🔧 Considera reiniciar el servicio afectado")
            suggestions.append("📞 Escala al equipo de infraestructura si el problema persiste")
        
        solution_keywords = ["solución", "solved", "fix", "resolver"]
        has_solution = any(keyword in tool_results.lower() for keyword in solution_keywords)
        
        if has_solution:
            suggestions.append("✅ Se encontró una solución en el historial")
            suggestions.append("📝 Documenta los pasos seguidos para referencia futura")
        
        if not suggestions:
            suggestions.append("📚 Consulta la documentación completa para más información")
            suggestions.append("🔍 Refina tu búsqueda si no encontraste lo que necesitabas")
        
        return "**Acciones sugeridas:**\n" + "\n".join(f"- {s}" for s in suggestions)
        
    except Exception as e:
        logger.error(f"Error en suggest_action: {e}")
        return "No fue posible generar sugerencias en este momento."