"""Router con OpenRouter usando Tool Calling real."""
import json
import logging
import time
from typing import Any, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import settings
from src.exceptions import LLMTimeoutError
from src.llm.timeout_wrapper import call_llm_with_timeout
from src.utils.circuit_breaker import openrouter_circuit_breaker
from src.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL DEFINITIONS (para binding)
# =============================================================================

@tool
def search_documents(query: str, limit: int = 5) -> str:
    """Busca en la documentación técnica.
    
    Úsala cuando el usuario pregunte por:
    - Cómo configurar algo
    - Qué significa un error
    - Dónde está un manual o guía
    - Información sobre servidores, redes, seguridad
    - Preguntas generales de tecnología
    
    Args:
        query: Texto de búsqueda o pregunta.
        limit: Número máximo de resultados (default 5).
    
    Returns:
        String con los documentos encontrados.
    """
    from src.tools.documents import search_documents as _search
    return _search.invoke({"query": query, "limit": limit})


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
    from src.tools.tickets import search_tickets as _search
    return _search.invoke({"query": query, "limit": limit})


@tool
def check_service_status(service_name: str) -> str:
    """Consulta el estado de un servicio específico.
    
    Úsala cuando el usuario pregunte por:
    - Estado de un servidor o servicio
    - Si un servicio está caído o disponible
    - Uptime de un sistema
    
    Args:
        service_name: Nombre del servicio a consultar.
    
    Returns:
        String con el estado del servicio.
    """
    from src.tools.status import check_service_status as _check
    return _check.invoke({"service_name": service_name})


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
    from src.tools.suggestion import suggest_action as _suggest
    return _suggest.invoke({"tool_results": tool_results})


# Lista de herramientas disponibles para bind_tools
AVAILABLE_TOOLS = [
    search_documents,
    search_tickets,
    check_service_status,
    suggest_action
]


def get_openrouter_llm():
    """Obtiene el modelo de OpenRouter."""
    return ChatOpenAI(
        model=settings.openrouter_model,
        openai_api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0.1,
        request_timeout=15,  # Timeout HTTP en segundos (debe ser <= LLM_CALL_TIMEOUT)
        max_retries=0  # Retry manejado por nosotros con circuit breaker
    )


def load_prompt(prompt_name: str) -> str:
    """Carga un prompt desde archivo."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


SYSTEM_PROMPT = load_prompt("router_prompt") or """Eres el orquestador de un sistema de soporte técnico.
Tu única tarea es decidir qué herramienta usar para responder la consulta del usuario.

Dispones de estas herramientas:
- search_documents: Busca en documentación técnica
- search_tickets: Busca en historial de tickets resueltos  
- check_service_status: Consulta el estado de un servicio/servidor
- suggest_action: Sugiere acciones basadas en resultados

Instrucciones:
1. Analiza la consulta del usuario
2. Selecciona la herramienta más apropiada (puede ser más de una)
3. Si no necesitas herramientas, responde directamente con "direct"

Responde en formato JSON:
{"tools": ["nombre_de_herramienta"], "reason": "razón de la selección"}

Ejemplos:
- {"tools": ["search_documents"], "reason": "El usuario pregunta sobre configuración de nginx"}
- {"tools": ["check_service_status"], "reason": "El usuario quiere saber si el servidor web está disponible"}
- {"tools": ["search_tickets", "suggest_action"], "reason": "El usuario reporta un error y quiere soluciones previas"}
- {"direct": true, "reason": "El usuario solo wants saludar"}"""


def router(query: str) -> dict:
    """Determina qué herramienta(s) usar para responder la consulta.
    
    Args:
        query: Consulta del usuario.
    
    Returns:
        Dict con herramientas seleccionadas y razón.
    """
    start_time = time.time()
    
    # Intentar usar OpenRouter
    try:
        result = _try_llm_router(query)
        if result:
            elapsed = time.time() - start_time
            logger.info(f"Router completado en {elapsed:.2f}s: {result.get('tools', [])}")
            return result
    except Exception as e:
        logger.warning(f"OpenRouter falló, usando fallback: {e}")
    
    # Fallback: keyword matching
    elapsed = time.time() - start_time
    result = _keyword_fallback(query)
    logger.info(f"Router fallback en {elapsed:.2f}s: {result.get('tools', [])}")
    return result


def _try_llm_router(query: str) -> Optional[dict]:
    """Intenta usar OpenRouter para routing con circuit breaker y timeout real."""
    cb = openrouter_circuit_breaker()

    def _call_llm() -> Any:
        logger.info("Llamando a OpenRouter con modelo: %s", settings.openrouter_model)
        llm = get_openrouter_llm()
        llm = llm.bind_tools(AVAILABLE_TOOLS, tool_choice="auto")

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=query)
        ]

        logger.info("Enviando mensajes al LLM...")
        response = llm.invoke(messages)
        logger.info(
            "Respuesta del LLM recibida. tool_calls: %s",
            hasattr(response, 'tool_calls') and response.tool_calls,
        )
        return response

    def _call_with_timeout() -> Any:
        """Ejecuta la llamada al LLM con timeout REAL a nivel de thread."""
        return call_llm_with_timeout(
            llm_callable=_call_llm,
            timeout_seconds=settings.router_llm_timeout,
            model_name=settings.openrouter_model,
        )

    try:
        logger.info("Intentando router con OpenRouter...")
        response = cb.call(_call_with_timeout)

        # Verificar tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            tools = [tc["name"] for tc in response.tool_calls]
            args: dict[str, Any] = {}
            for tc in response.tool_calls:
                if tc.get("args"):
                    args.update(tc["args"])

            logger.info("Tool calling via OpenRouter: %s", tools)
            return {
                "tools": tools,
                "args": args,
                "reason": "Llamada a herramienta(s) seleccionada(s) por LLM"
            }

        # Verificar respuesta directa
        content = response.content if hasattr(response, "content") else str(response)
        if "direct" in content.lower():
            return {
                "tools": [],
                "direct": True,
                "reason": "No se requiere herramienta"
            }

        # Intentar parsear JSON
        try:
            parsed = json.loads(content)
            tools = parsed.get("tools", [])
            return {
                "tools": tools if isinstance(tools, list) else [tools],
                "reason": parsed.get("reason", "Selección por JSON")
            }
        except json.JSONDecodeError:
            return None

    except LLMTimeoutError as exc:
        logger.warning("Router LLM timed out: %s", exc)
        return None
    except Exception as exc:
        logger.warning("OpenRouter LLM call failed: %s", exc)
        return None


def _keyword_fallback(query: str) -> dict:
    """Fallback con keyword matching si el LLM falla."""
    query_lower = query.lower()
    
    status_keywords = ["estado", "servicio", "servidor", "caído", "uptime", "disponible", "status"]
    ticket_keywords = ["error", "problema", "antes", "historial", "solución", "resuelto", "ticket"]
    doc_keywords = ["documentación", "manual", "configurar", "guía", "qué es", "que es", "cómo", "como", "docs"]
    
    if any(kw in query_lower for kw in status_keywords):
        return {"tools": ["check_service_status"], "reason": "Fallback: keywords de estado"}
    elif any(kw in query_lower for kw in ticket_keywords):
        return {"tools": ["search_tickets"], "reason": "Fallback: keywords de tickets"}
    elif any(kw in query_lower for kw in doc_keywords):
        return {"tools": ["search_documents"], "reason": "Fallback: keywords de docs"}
    else:
        return {"tools": ["search_documents"], "reason": "Fallback: consulta general"}
