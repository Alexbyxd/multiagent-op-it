"""Synthesizer con OpenRouter."""
import logging
from typing import Any, Optional
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import settings
from src.exceptions import LLMTimeoutError
from src.llm.timeout_wrapper import call_llm_with_timeout
from src.utils.circuit_breaker import openrouter_circuit_breaker
from src.utils.retry import retry_with_backoff

logger = logging.getLogger(__name__)


def get_openrouter_llm():
    """Obtiene el modelo de OpenRouter para síntesis."""
    return ChatOpenAI(
        model=settings.openrouter_model,
        openai_api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=0.3,
        request_timeout=30,  # Timeout HTTP en segundos para síntesis (puede ser más largo)
        max_retries=0  # Retry manejado por nosotros con circuit breaker
    )


def load_prompt(prompt_name: str) -> str:
    """Carga un prompt desde archivo."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


SYSTEM_PROMPT = load_prompt("synthesizer_prompt") or """Eres un asistente de soporte técnico corporativo.
Tu tarea es generar una respuesta final clara y útil para el usuario."""


def synthesize(query: str, tool_results: str, user_level: str = "basic") -> str:
    """Genera respuesta final usando OpenRouter.
    
    Args:
        query: Consulta original del usuario.
        tool_results: Resultados de las herramientas ejecutadas.
        user_level: Nivel del usuario (basic, advanced, admin).
    
    Returns:
        Respuesta final sintetizada.
    """
    # Si no hay resultados, usar fallback
    if not tool_results:
        return _fallback_response(query, tool_results, user_level)
    
    # Intentar usar OpenRouter
    try:
        result = _try_llm_synthesize(query, tool_results, user_level)
        if result:
            return result
    except Exception as e:
        logger.warning(f"OpenRouter synthesize falló: {e}")
    
    # Fallback sin LLM
    return _fallback_response(query, tool_results, user_level)


def _try_llm_synthesize(query: str, tool_results: str, user_level: str) -> Optional[str]:
    """Intenta usar OpenRouter para síntesis con circuit breaker y timeout."""
    cb = openrouter_circuit_breaker()

    def _call_llm() -> Any:
        llm = get_openrouter_llm()

        level_context = {
            "basic": "Explica de forma sencilla, evita jerga técnica innecesaria.",
            "advanced": "Puedes usar términos técnicos, sé detallado.",
            "admin": "Incluye información de infraestructura y métricas relevantes."
        }

        messages = [
            SystemMessage(content=f"{SYSTEM_PROMPT}\n\n{level_context.get(user_level, '')}"),
            HumanMessage(content=f"""
Consulta del usuario: {query}

Resultados de herramientas:
{tool_results}

Genera una respuesta final clara y útil.
""")
        ]

        return llm.invoke(messages)

    def _call_with_timeout() -> Any:
        """Wrap LLM call with thread-level timeout enforcement."""
        return call_llm_with_timeout(
            llm_callable=_call_llm,
            timeout_seconds=settings.synthesizer_llm_timeout,
            model_name=settings.openrouter_model,
        )

    try:
        response = cb.call(_call_with_timeout)
        return response.content if hasattr(response, "content") else str(response)
    except LLMTimeoutError as exc:
        logger.warning("Synthesizer LLM timed out: %s", exc)
        return None
    except Exception as exc:
        logger.warning("OpenRouter synthesize failed: %s", exc)
        return None


def _fallback_response(query: str, tool_results: str, user_level: str) -> str:
    """Formatea respuesta sin usar LLM.

    Args:
        query: Consulta original del usuario.
        tool_results: Resultados de las herramientas ejecutadas.
        user_level: Nivel del usuario (basic, advanced, admin).

    Returns:
        Respuesta formateada con nota de síntesis no disponible.
    """
    if not tool_results:
        return (
            f"**AI Synthesis Unavailable**\n\n"
            f"No tengo información específica para responder a: {query}"
        )

    level_note = {
        "basic": "",
        "advanced": "[Nivel técnico: avanzado]",
        "admin": "[Nivel técnico: administrador]"
    }

    response = (
        "**AI Synthesis Unavailable**\n\n"
        "La generación automática de respuestas no está disponible en este momento. "
        "A continuación se muestran los resultados obtenidos:\n\n"
        f"**Consulta:** {query}\n\n"
        f"**Resultados:**\n\n{tool_results}\n\n"
    )

    if level_note.get(user_level):
        response += f"{level_note[user_level]}\n"

    response += "\n¿Te sirve esta información?"

    return response
