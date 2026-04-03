"""Tests para fallback de timeout en router y synthesizer."""
import pytest
from unittest.mock import patch, MagicMock

from src.exceptions import LLMTimeoutError
from src.utils.circuit_breaker import _circuit_breakers


@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Reset shared circuit breakers before each test."""
    _circuit_breakers.clear()
    yield
    _circuit_breakers.clear()


# =============================================================================
# TEST 3.4: Synthesizer fallback on LLMTimeoutError
# =============================================================================

def test_synthesizer_fallback_on_llm_timeout_error():
    """Mock wrapper to raise LLMTimeoutError; assert fallback string contains tool results and notice."""
    from src.llm.synthesizer import synthesize

    with patch(
        "src.llm.synthesizer.call_llm_with_timeout",
        side_effect=LLMTimeoutError(timeout_seconds=30.0, model="test-model"),
    ):
        query = "¿Cómo resuelvo el error 500?"
        tool_results = "El error 500 indica un problema interno del servidor."

        result = synthesize(query, tool_results, "basic")

        assert "AI Synthesis Unavailable" in result
        assert query in result
        assert tool_results in result


def test_synthesizer_fallback_on_empty_results_with_timeout():
    """Fallback when tool_results is empty and LLM times out."""
    from src.llm.synthesizer import synthesize

    with patch(
        "src.llm.synthesizer.call_llm_with_timeout",
        side_effect=LLMTimeoutError(timeout_seconds=30.0),
    ):
        result = synthesize("¿Qué es esto?", "", "basic")

        assert "AI Synthesis Unavailable" in result
        assert "No tengo información" in result


# =============================================================================
# TEST 3.5: Router fallback to keyword matching on timeout
# =============================================================================

def test_router_fallback_on_llm_timeout_error():
    """Mock wrapper to raise LLMTimeoutError; assert returned dict selects tool via keyword analysis."""
    from src.llm.router import router

    with patch(
        "src.llm.router.call_llm_with_timeout",
        side_effect=LLMTimeoutError(timeout_seconds=15.0, model="test-model"),
    ):
        # Query with status-related keywords
        result = router("¿Está caído el servidor web?")

        assert isinstance(result, dict)
        assert "check_service_status" in result["tools"]
        assert "Fallback" in result["reason"]


def test_router_fallback_on_llm_timeout_ticket_keywords():
    """Router falls back to ticket search on timeout with ticket keywords."""
    from src.llm.router import router

    with patch(
        "src.llm.router.call_llm_with_timeout",
        side_effect=LLMTimeoutError(timeout_seconds=15.0),
    ):
        result = router("¿Cómo se resolvió el error antes?")

        assert "search_tickets" in result["tools"]
        assert "Fallback" in result["reason"]


def test_router_fallback_on_llm_timeout_doc_keywords():
    """Router falls back to documents on timeout with doc keywords."""
    from src.llm.router import router

    with patch(
        "src.llm.router.call_llm_with_timeout",
        side_effect=LLMTimeoutError(timeout_seconds=15.0),
    ):
        result = router("¿Dónde está la documentación de nginx?")

        assert "search_documents" in result["tools"]
        assert "Fallback" in result["reason"]
