"""Synthesizer con Gemini 2.5 Pro."""
import logging
from typing import Optional
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings

logger = logging.getLogger(__name__)


def get_synthesizer_llm():
    """Obtiene el modelo del synthesizer."""
    return ChatGoogleGenerativeAI(
        model=settings.synthesizer_model,
        google_api_key=settings.google_api_key,
        temperature=0.3
    )


def load_prompt(prompt_name: str) -> str:
    """Carga un prompt desde archivo."""
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.txt"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return ""


SYSTEM_PROMPT = load_prompt("synthesizer_prompt") or """Eres un asistente de soporte técnico corporativo..."""


def synthesize(query: str, tool_results: str, user_level: str = "basic") -> str:
    """Genera respuesta final usando Gemini Pro.
    
    Args:
        query: Consulta original del usuario.
        tool_results: Resultados de las herramientas ejecutadas.
        user_level: Nivel del usuario (basic, advanced, admin).
    
    Returns:
        Respuesta final sintetizada.
    """
    llm = get_synthesizer_llm()
    
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

Genera una respuesta final.
""")
    ]
    
    response = llm.invoke(messages)
    return response.content
