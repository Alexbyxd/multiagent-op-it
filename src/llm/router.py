"""Router con Gemini 2.5 Flash."""
import logging
from typing import Optional
from pathlib import Path

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings

logger = logging.getLogger(__name__)


def get_router_llm():
    """Obtiene el modelo del router."""
    return ChatGoogleGenerativeAI(
        model=settings.router_model,
        google_api_key=settings.google_api_key,
        temperature=0.1
    )


def load_prompt(prompt_name: str) -> str:
    """Carga un prompt desde archivo."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


SYSTEM_PROMPT = load_prompt("router_prompt") or """Eres el orquestador..."""


@tool
def router(query: str) -> str:
    """Determina qué herramienta usar para responder la consulta.
    
    Args:
        query: Consulta del usuario.
    
    Returns:
        JSON con la herramienta seleccionada y razón.
    """
    llm = get_router_llm()
    
    doc_keywords = ["documentación", "manual", "configurar", "guía", "qué es", "cómo"]
    ticket_keywords = ["error", "problema", "antes", "historial", "solución", "resuelto"]
    status_keywords = ["estado", "servicio", "servidor", "caído", "uptime", "disponible"]
    
    query_lower = query.lower()
    
    if any(kw in query_lower for kw in status_keywords):
        return '{"tool": "check_service_status", "reason": "Consulta sobre estado de servicio"}'
    elif any(kw in query_lower for kw in ticket_keywords):
        return '{"tool": "search_tickets", "reason": "Consulta sobre problemas o errores pasados"}'
    elif any(kw in query_lower for kw in doc_keywords):
        return '{"tool": "search_documents", "reason": "Consulta sobre documentación técnica"}'
    else:
        return '{"tool": "search_documents", "reason": "Consulta general, se busca en documentación"}'