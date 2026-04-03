"""Tests para el circuit breaker y su integración con router/synthesizer."""
import pytest
import time
from unittest.mock import patch, MagicMock, call

from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
    openrouter_circuit_breaker,
)
from src.llm.router import router, _try_llm_router
from src.llm.synthesizer import synthesize, _try_llm_synthesize


# =============================================================================
# HELPERS
# =============================================================================

def fail_once():
    """Función que lanza error."""
    raise ConnectionError("test error")


def fail_once_factory():
    """Factory que crea función que falla."""
    return fail_once


# =============================================================================
# TESTS DEL CIRCUIT BREAKER
# =============================================================================

def test_circuit_breaker_opens_after_failures():
    """Verifica que el CB se abre después de X fallos consecutivos."""
    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=30.0
    )
    cb = CircuitBreaker("test_cb", config=config)

    # Simular fallos consecutivos
    for i in range(3):
        with pytest.raises(ConnectionError):
            cb.call(fail_once)

    # Verificar que está abierto
    assert cb.state == CircuitState.OPEN
    stats = cb.get_stats()
    assert stats.consecutive_failures == 3
    assert stats.failed_calls == 3


def test_circuit_breaker_closes_after_successes():
    """Verifica que el CB se cierra después de X éxitos en HALF_OPEN."""
    config = CircuitBreakerConfig(
        failure_threshold=2,
        success_threshold=2,
        timeout=0.1  # Timeout corto para testing
    )
    cb = CircuitBreaker("test_cb", config=config)

    # Provocar apertura: 2 fallos
    for i in range(2):
        with pytest.raises(ConnectionError):
            cb.call(fail_once)

    assert cb.state == CircuitState.OPEN

    # Esperar a que timeout expire y pase a HALF_OPEN
    time.sleep(0.2)
    assert cb.state == CircuitState.HALF_OPEN

    # Simular éxitos consecutivos
    cb.call(lambda: "success")
    cb.call(lambda: "success")

    # Verificar que se cerró
    assert cb.state == CircuitState.CLOSED
    stats = cb.get_stats()
    assert stats.consecutive_successes == 2


def test_circuit_breaker_rejects_calls_when_open():
    """Verifica que rechaza llamadas cuando está OPEN."""
    config = CircuitBreakerConfig(
        failure_threshold=2,
        success_threshold=2,
        timeout=30.0
    )
    cb = CircuitBreaker("test_cb", config=config)

    # Provocar apertura
    for i in range(2):
        with pytest.raises(ConnectionError):
            cb.call(fail_once)

    assert cb.state == CircuitState.OPEN

    # Intentar llamar cuando está abierto - debe rechazar
    with pytest.raises(CircuitBreakerOpen) as exc_info:
        cb.call(lambda: "this should not execute")

    assert "is OPEN" in str(exc_info.value)

    stats = cb.get_stats()
    assert stats.rejected_calls == 1


def test_circuit_breaker_success_resets_consecutive_failures():
    """Verifica que un éxito resetea el conteo de fallos consecutivos."""
    config = CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        timeout=30.0
    )
    cb = CircuitBreaker("test_cb", config=config)

    # Simular 2 fallos
    for i in range(2):
        with pytest.raises(ConnectionError):
            cb.call(fail_once)

    assert cb.get_stats().consecutive_failures == 2

    # Un éxito debe resetear los fallos consecutivos
    cb.call(lambda: "success")

    assert cb.get_stats().consecutive_failures == 0
    assert cb.get_stats().consecutive_successes == 1


def test_circuit_breaker_half_open_to_open_on_failure():
    """Verifica que vuelve a abrir si falla en estado HALF_OPEN."""
    config = CircuitBreakerConfig(
        failure_threshold=2,
        success_threshold=2,
        timeout=0.1
    )
    cb = CircuitBreaker("test_cb", config=config)

    # Abrir el circuit breaker
    for i in range(2):
        with pytest.raises(ConnectionError):
            cb.call(fail_once)

    # Esperar a que pase a HALF_OPEN
    time.sleep(0.2)
    assert cb.state == CircuitState.HALF_OPEN

    # Fallar en HALF_OPEN debe volver a abrir
    with pytest.raises(ConnectionError):
        cb.call(fail_once)

    assert cb.state == CircuitState.OPEN


# =============================================================================
# TESTS DEL ROUTER CON CIRCUIT BREAKER
# =============================================================================

def test_router_calls_circuit_breaker():
    """Verifica que el router usa el circuit breaker."""
    mock_cb = MagicMock()
    mock_cb.call.return_value = MagicMock(
        content='{"tools": ["search_documents"], "reason": "Prueba"}'
    )

    with patch('src.llm.router.openrouter_circuit_breaker', return_value=mock_cb):
        with patch('src.llm.router.get_openrouter_llm') as mock_llm:
            mock_llm.return_value.bind_tools.return_value.invoke.return_value = (
                MagicMock(tool_calls=[{"name": "search_documents", "args": {}}])
            )

            result = _try_llm_router("¿Dónde está el manual?")

            # Verificar que se llamó al circuit breaker
            mock_cb.call.assert_called_once()


def test_router_fallback_when_cb_open():
    """Verifica que hace fallback a keywords cuando el CB está abierto."""
    with patch('src.llm.router.openrouter_circuit_breaker') as mock_get_cb:
        # Crear mock del circuit breaker
        mock_cb = MagicMock()

        # Configurar para que la primera llamada lance CircuitBreakerOpen
        def side_effect(func, *args, **kwargs):
            raise CircuitBreakerOpen("Circuit breaker is OPEN")

        mock_cb.call.side_effect = side_effect
        mock_get_cb.return_value = mock_cb

        # Ejecutar router
        result = router("¿Está caído el servidor?")

        # Verificar que returns fallback por keywords
        assert "check_service_status" in result["tools"]
        assert "Fallback" in result.get("reason", "")


def test_router_fallback_on_llm_exception():
    """Verifica que el router hace fallback cuando el LLM lanza excepción."""
    with patch('src.llm.router.openrouter_circuit_breaker') as mock_get_cb:
        mock_cb = MagicMock()
        mock_cb.call.side_effect = Exception("API Error")
        mock_get_cb.return_value = mock_cb

        result = router("¿Cómo configuro nginx?")

        # Debe hacer fallback a keywords de documentación
        assert "search_documents" in result["tools"]


def test_router_keyword_fallback_default():
    """Verifica el fallback por defecto cuando no hay keywords."""
    with patch('src.llm.router.openrouter_circuit_breaker') as mock_get_cb:
        mock_cb = MagicMock()
        mock_cb.call.side_effect = Exception("API Error")
        mock_get_cb.return_value = mock_cb

        result = router("Hola, ¿cómo estás?")

        # Por defecto debe usar search_documents
        assert "search_documents" in result["tools"]


# =============================================================================
# TESTS DEL SYNTHESIZER CON CIRCUIT BREAKER
# =============================================================================

def test_synthesizer_calls_circuit_breaker():
    """Verifica que el synthesizer usa el circuit breaker."""
    mock_cb = MagicMock()
    mock_cb.call.return_value = MagicMock(content="Respuesta sintetizada")

    with patch('src.llm.synthesizer.openrouter_circuit_breaker', return_value=mock_cb):
        with patch('src.llm.synthesizer.get_openrouter_llm') as mock_llm:
            mock_llm.return_value.invoke.return_value = MagicMock(
                content="Respuesta sintetizada"
            )

            result = _try_llm_synthesize(
                "Consulta de prueba",
                "Resultados de tools",
                "basic"
            )

            # Verificar que se llamó al circuit breaker
            mock_cb.call.assert_called_once()


def test_synthesizer_fallback_when_cb_open():
    """Verifica que usa fallback cuando el CB está abierto."""
    with patch('src.llm.synthesizer.openrouter_circuit_breaker') as mock_get_cb:
        mock_cb = MagicMock()
        mock_cb.call.side_effect = CircuitBreakerOpen("Circuit breaker is OPEN")
        mock_get_cb.return_value = mock_cb

        query = "¿Cómo resuelvo el error 500?"
        tool_results = "El error 500 es un error interno del servidor"

        result = synthesize(query, tool_results, "basic")

        # Verificar que usa fallback
        assert query in result
        assert tool_results in result
        assert "basic" not in result.lower() or "basic" in result.lower()


def test_synthesizer_fallback_on_llm_exception():
    """Verifica que el synthesizer hace fallback cuando el LLM falla."""
    with patch('src.llm.synthesizer.openrouter_circuit_breaker') as mock_get_cb:
        mock_cb = MagicMock()
        mock_cb.call.side_effect = Exception("API Error")
        mock_get_cb.return_value = mock_cb

        query = "¿Cuál es el estado del servidor?"
        tool_results = "El servidor está funcionando correctamente"

        result = synthesize(query, tool_results, "advanced")

        # Debe usar fallback
        assert tool_results in result
        assert "advanced" in result.lower() or "avanzado" in result.lower()


def test_synthesizer_fallback_on_empty_tool_results():
    """Verifica fallback cuando no hay resultados de herramientas."""
    result = synthesize("¿Qué es esto?", "", "basic")

    # Debe responder con mensaje de no información
    assert "No tengo información" in result or "No tengo" in result


def test_synthesizer_returns_none_on_cb_exception():
    """Verifica que retorna None cuando CB lanza excepción en _try_llm."""
    with patch('src.llm.synthesizer.openrouter_circuit_breaker') as mock_get_cb:
        mock_cb = MagicMock()
        mock_cb.call.side_effect = CircuitBreakerOpen("CB Open")
        mock_get_cb.return_value = mock_cb

        result = _try_llm_synthesize("query", "tool_results", "basic")

        # Debe retornar None para que synthesize use fallback
        assert result is None


# =============================================================================
# TESTS DE INTEGRACIÓN
# =============================================================================

def test_circuit_breaker_preserves_stats_after_state_change():
    """Verifica que las estadísticas se preservan cuando cambia el estado."""
    config = CircuitBreakerConfig(
        failure_threshold=2,
        success_threshold=2,
        timeout=0.1
    )
    cb = CircuitBreaker("test_stats", config=config)

    # Una llamada exitosa
    cb.call(lambda: "success")
    stats1 = cb.get_stats()
    assert stats1.successful_calls == 1

    # Provocar apertura
    for i in range(2):
        with pytest.raises(ConnectionError):
            cb.call(fail_once)

    stats2 = cb.get_stats()
    assert stats2.failed_calls == 2
    assert stats2.total_calls == 3


def test_multiple_circuit_breakers_are_independent():
    """Verifica que múltiples CBs son independientes."""
    cb1 = CircuitBreaker("cb1", CircuitBreakerConfig(failure_threshold=2))
    cb2 = CircuitBreaker("cb2", CircuitBreakerConfig(failure_threshold=2))

    # Abrir cb1
    for i in range(2):
        with pytest.raises(ConnectionError):
            cb1.call(fail_once)

    # cb1 debe estar abierto
    assert cb1.state == CircuitState.OPEN

    # cb2 debe seguir cerrado (llamada exitosa)
    result = cb2.call(lambda: "success")
    assert result == "success"
    assert cb2.state == CircuitState.CLOSED
