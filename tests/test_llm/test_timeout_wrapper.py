"""Tests para el timeout wrapper de LLM calls."""
import time
import pytest
from unittest.mock import MagicMock, patch

from src.exceptions import LLMTimeoutError
from src.llm.timeout_wrapper import call_llm_with_timeout


# =============================================================================
# TEST 3.1: call_llm_with_timeout success
# =============================================================================

def test_call_llm_with_timeout_success():
    """Mock callable returning instantly; assert result matches and no exception raised."""
    expected = {"content": "test response"}

    def fast_callable():
        return expected

    result = call_llm_with_timeout(
        llm_callable=fast_callable,
        timeout_seconds=5.0,
        model_name="test-model",
    )

    assert result == expected


# =============================================================================
# TEST 3.2: call_llm_with_timeout timeout
# =============================================================================

def test_call_llm_with_timeout_raises_llm_timeout_error():
    """Mock callable sleeping > timeout; assert LLMTimeoutError raised with correct attributes."""
    def slow_callable():
        time.sleep(10)
        return "should never reach"

    with pytest.raises(LLMTimeoutError) as exc_info:
        call_llm_with_timeout(
            llm_callable=slow_callable,
            timeout_seconds=0.1,
            model_name="slow-model",
        )

    assert exc_info.value.timeout_seconds == 0.1
    assert exc_info.value.model == "slow-model"
    assert "0.1s" in str(exc_info.value)
    assert "slow-model" in str(exc_info.value)


# =============================================================================
# TEST 3.3: LLMTimeoutError attributes and message
# =============================================================================

def test_llm_timeout_error_with_model():
    """Instantiate with model; assert message format and attribute values."""
    exc = LLMTimeoutError(timeout_seconds=15.0, model="gemini-2.5-flash")

    assert exc.timeout_seconds == 15.0
    assert exc.model == "gemini-2.5-flash"
    assert "15.0s" in str(exc)
    assert "gemini-2.5-flash" in str(exc)
    assert isinstance(exc, Exception)


def test_llm_timeout_error_without_model():
    """Instantiate without model; assert message format."""
    exc = LLMTimeoutError(timeout_seconds=30.0)

    assert exc.timeout_seconds == 30.0
    assert exc.model == ""
    assert "30.0s" in str(exc)
    assert "(model:" not in str(exc)


def test_llm_timeout_error_is_llm_error_subclass():
    """Verify LLMTimeoutError inherits from LLMError."""
    from src.exceptions import LLMError

    exc = LLMTimeoutError(timeout_seconds=10.0)
    assert isinstance(exc, LLMError)


# =============================================================================
# Additional: callable that raises an exception propagates correctly
# =============================================================================

def test_call_llm_with_timeout_propagates_exception():
    """Callable raising an exception should propagate it (not LLMTimeoutError)."""
    def failing_callable():
        raise ValueError("simulated LLM error")

    with pytest.raises(ValueError, match="simulated LLM error"):
        call_llm_with_timeout(
            llm_callable=failing_callable,
            timeout_seconds=5.0,
        )
