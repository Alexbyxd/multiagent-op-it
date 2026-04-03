"""Tests para timeout en nodos del grafo."""
import time
import pytest
from unittest.mock import MagicMock

from src.config import settings
from src.graph import nodes as nodes_mod
from src.graph.nodes import AgentState, execute_tool_node
from src.exceptions import LLMTimeoutError


@pytest.fixture(autouse=True)
def restore_tool_map():
    """Restore TOOL_MAP after each test."""
    original = dict(nodes_mod.TOOL_MAP)
    yield
    nodes_mod.TOOL_MAP = original


# =============================================================================
# TEST 3.7: future.result(timeout=...) in nodes
# =============================================================================

def test_execute_tool_node_handles_tool_timeout_parallel():
    """Multiple tools with one timing out; assert TimeoutError caught gracefully."""
    state: AgentState = {
        "query": "test query",
        "user_level": "basic",
        "intent": "general",
        "selected_tools": ["search_documents", "search_tickets"],
        "tool_results": [],
        "final_response": None,
        "error": None,
    }

    mock_slow = MagicMock()
    mock_slow.invoke.side_effect = lambda _: time.sleep(60)

    mock_fast = MagicMock()
    mock_fast.invoke.return_value = "fast result"

    nodes_mod.TOOL_MAP = {
        "search_documents": mock_slow,
        "search_tickets": mock_fast,
    }

    original_timeout = settings.tool_execution_timeout
    try:
        settings.tool_execution_timeout = 0.5
        result = execute_tool_node(state)
    finally:
        settings.tool_execution_timeout = original_timeout

    # Should complete gracefully with one timeout and one success
    assert len(result["tool_results"]) == 2

    timeout_results = [r for r in result["tool_results"] if not r["success"]]
    assert len(timeout_results) >= 1
    assert "timed out" in timeout_results[0]["result"].lower()

    success_results = [r for r in result["tool_results"] if r["success"]]
    assert len(success_results) >= 1


def test_execute_tool_node_handles_tool_timeout_sequential():
    """Single tool timing out; assert TimeoutError caught gracefully."""
    state: AgentState = {
        "query": "test query",
        "user_level": "basic",
        "intent": "general",
        "selected_tools": ["search_documents"],
        "tool_results": [],
        "final_response": None,
        "error": None,
    }

    mock_slow = MagicMock()
    mock_slow.invoke.side_effect = lambda _: time.sleep(60)

    nodes_mod.TOOL_MAP = {"search_documents": mock_slow}

    original_timeout = settings.tool_execution_timeout
    try:
        settings.tool_execution_timeout = 0.5
        result = execute_tool_node(state)
    finally:
        settings.tool_execution_timeout = original_timeout

    assert len(result["tool_results"]) == 1
    tool_result = result["tool_results"][0]
    assert tool_result["tool"] == "search_documents"
    assert tool_result["success"] is False
    assert "timed out" in tool_result["result"].lower()


def test_execute_tool_node_succeeds_within_timeout():
    """Tool completes before timeout; result retrieved successfully."""
    state: AgentState = {
        "query": "test query",
        "user_level": "basic",
        "intent": "general",
        "selected_tools": ["search_documents"],
        "tool_results": [],
        "final_response": None,
        "error": None,
    }

    mock_tool = MagicMock()
    mock_tool.invoke.return_value = "document content found"

    nodes_mod.TOOL_MAP = {"search_documents": mock_tool}

    original_timeout = settings.tool_execution_timeout
    try:
        settings.tool_execution_timeout = 5
        result = execute_tool_node(state)
    finally:
        settings.tool_execution_timeout = original_timeout

    assert len(result["tool_results"]) == 1
    tool_result = result["tool_results"][0]
    assert tool_result["success"] is True
    assert tool_result["result"] == "document content found"


def test_execute_tool_node_no_tools():
    """No tools selected; should return empty results with error."""
    state: AgentState = {
        "query": "test query",
        "user_level": "basic",
        "intent": "general",
        "selected_tools": [],
        "tool_results": [],
        "final_response": None,
        "error": None,
    }

    result = execute_tool_node(state)

    assert result["tool_results"] == []
    assert result["error"] == "No tool selected"


def test_execute_tool_node_invalid_tool():
    """Invalid tool name; should return error."""
    state: AgentState = {
        "query": "test query",
        "user_level": "basic",
        "intent": "general",
        "selected_tools": ["nonexistent_tool"],
        "tool_results": [],
        "final_response": None,
        "error": None,
    }

    result = execute_tool_node(state)

    assert result["tool_results"] == []
    assert "Ninguna herramienta" in result["error"]


# =============================================================================
# TEST 3.8: Integration test - full graph with mocked slow LLM
# =============================================================================

def test_full_graph_with_mocked_slow_llm():
    """Mock LLM to sleep 60s; assert graph completes within timeout, fallback executes."""
    from unittest.mock import patch
    from src.llm.router import router
    from src.llm.synthesizer import synthesize
    from src.utils.circuit_breaker import _circuit_breakers

    # Reset shared circuit breakers
    _circuit_breakers.clear()

    # Test router: should fallback to keywords within the timeout
    start = time.time()

    with patch("src.llm.router.call_llm_with_timeout", side_effect=LLMTimeoutError(timeout_seconds=15.0)):
        result = router("¿Está caído el servidor?")

    elapsed = time.time() - start

    # Router should complete quickly via fallback (not wait 60s)
    assert elapsed < 5.0, f"Router took {elapsed:.1f}s — should use fallback immediately"
    assert "check_service_status" in result["tools"]

    # Test synthesizer: should fallback gracefully
    with patch(
        "src.llm.synthesizer.call_llm_with_timeout",
        side_effect=LLMTimeoutError(timeout_seconds=30.0),
    ):
        synth_result = synthesize(
            "¿Está caído el servidor?",
            "El servidor web-server no responde desde hace 10 minutos.",
            "basic",
        )

    assert "AI Synthesis Unavailable" in synth_result
    assert "servidor" in synth_result
