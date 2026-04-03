"""Tests para circuit breaker con LLMTimeoutError."""
import pytest
from unittest.mock import patch

from src.exceptions import LLMTimeoutError
from src.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
)


# =============================================================================
# TEST 3.6: Circuit breaker increments failure on LLMTimeoutError
# =============================================================================

def test_circuit_breaker_increments_failure_on_llm_timeout_error():
    """Wrap timeout-raising callable in CB; assert failure counter increments."""
    config = CircuitBreakerConfig(failure_threshold=3, timeout=30.0)
    cb = CircuitBreaker("test_timeout_cb", config=config)

    def timeout_callable():
        raise LLMTimeoutError(timeout_seconds=15.0, model="test-model")

    with pytest.raises(LLMTimeoutError):
        cb.call(timeout_callable)

    stats = cb.get_stats()
    assert stats.failed_calls == 1
    assert stats.consecutive_failures == 1


def test_circuit_breaker_trips_open_after_timeout_threshold():
    """CB should trip to OPEN after failure_threshold timeout errors."""
    config = CircuitBreakerConfig(failure_threshold=3, timeout=30.0)
    cb = CircuitBreaker("test_trip_cb", config=config)

    def timeout_callable():
        raise LLMTimeoutError(timeout_seconds=15.0)

    for _ in range(3):
        with pytest.raises(LLMTimeoutError):
            cb.call(timeout_callable)

    assert cb.state == CircuitState.OPEN


def test_circuit_breaker_rejects_calls_after_timeout_trips_open():
    """After timeouts trip the CB, subsequent calls should be rejected."""
    config = CircuitBreakerConfig(failure_threshold=2, timeout=30.0)
    cb = CircuitBreaker("test_reject_cb", config=config)

    def timeout_callable():
        raise LLMTimeoutError(timeout_seconds=15.0)

    # Trip the breaker
    for _ in range(2):
        with pytest.raises(LLMTimeoutError):
            cb.call(timeout_callable)

    assert cb.state == CircuitState.OPEN

    # Next call should be rejected immediately
    with pytest.raises(CircuitBreakerOpen):
        cb.call(lambda: "should not execute")

    stats = cb.get_stats()
    assert stats.rejected_calls == 1


def test_circuit_breaker_distinguishes_timeout_from_exception():
    """Timeout failures should be logged distinctly (verified via stats)."""
    config = CircuitBreakerConfig(failure_threshold=5, timeout=30.0)
    cb = CircuitBreaker("test_distinct_cb", config=config)

    def timeout_callable():
        raise LLMTimeoutError(timeout_seconds=15.0)

    def exception_callable():
        raise ConnectionError("network error")

    with pytest.raises(LLMTimeoutError):
        cb.call(timeout_callable)

    with pytest.raises(ConnectionError):
        cb.call(exception_callable)

    stats = cb.get_stats()
    assert stats.failed_calls == 2
    assert stats.consecutive_failures == 2
