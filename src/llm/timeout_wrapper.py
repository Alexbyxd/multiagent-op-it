"""Timeout wrapper for LLM calls using daemon thread pattern.

Provides a shared `call_llm_with_timeout()` function that enforces
thread-level timeouts on any callable, preventing indefinite hangs
even when the underlying HTTP client blocks forever.
"""
import logging
import threading
from typing import Any, Callable

from src.exceptions import LLMTimeoutError

logger = logging.getLogger(__name__)


def call_llm_with_timeout(
    llm_callable: Callable[[], Any],
    timeout_seconds: float,
    model_name: str = "",
) -> Any:
    """Execute an LLM call with thread-level timeout enforcement.

    Uses a daemon thread + join() pattern to guarantee termination
    even when the underlying HTTP client hangs indefinitely.

    Args:
        llm_callable: Zero-arg callable that performs the LLM invocation.
        timeout_seconds: Maximum seconds to wait.
        model_name: Optional model identifier for error messages.

    Returns:
        The LLM response object.

    Raises:
        LLMTimeoutError: If the call exceeds timeout_seconds.
    """
    result_container: dict[str, Any] = {"response": None, "error": None}

    def _target() -> None:
        try:
            result_container["response"] = llm_callable()
        except Exception as exc:
            result_container["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        logger.error(
            "LLM call timed out after %.1fs — thread still alive, abandoning "
            "(model: %s)",
            timeout_seconds,
            model_name or "unknown",
        )
        raise LLMTimeoutError(timeout_seconds=timeout_seconds, model=model_name)

    if result_container["error"] is not None:
        raise result_container["error"]

    return result_container["response"]
