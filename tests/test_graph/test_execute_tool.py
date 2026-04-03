"""Tests para el nodo de ejecución de herramientas."""
import pytest
from unittest.mock import MagicMock, patch
from typing import TypedDict

from src.graph.nodes import execute_tool_node, TOOL_MAP


class MockTool:
    """Mock de StructuredTool de LangChain con método invoke."""

    def __init__(self, mock_func):
        self.func = mock_func
        self._mock = mock_func

    def invoke(self, input_dict):
        """Simula el invoke de LangChain tool."""
        # Extraer el primer valor del dict (query, service_name, tool_results, etc.)
        if isinstance(input_dict, dict):
            # Obtener el primer valor que no sea None
            for key in input_dict:
                if input_dict[key] is not None:
                    return self._mock(input_dict[key])
        return self._mock(input_dict)


def create_mock_tool(mock_func):
    """Crea un mock de tool compatible con LangChain."""
    mock = MagicMock()
    mock.invoke = lambda inp: mock_func(list(inp.values())[0] if inp else "")
    mock.func = mock_func
    return mock


class TestSuggestActionRunsLast:
    """Tests para verificar que suggest_action se ejecuta al final."""

    def test_suggest_action_runs_last(self):
        """Verifica que suggest_action se ejecuta después de las otras tools."""
        # Arrange: crear estado inicial
        state = {
            "query": "test query",
            "user_level": "basic",
            "intent": "test",
            "selected_tools": ["search_documents", "search_tickets", "suggest_action"],
            "tool_results": [],
            "final_response": None,
            "error": None
        }

        # Track de orden de ejecución
        execution_order = []

        def mock_documents(query: str) -> MagicMock:
            execution_order.append("search_documents")
            mock_result = MagicMock()
            mock_result.content = "documents result"
            return mock_result

        def mock_tickets(query: str) -> MagicMock:
            execution_order.append("search_tickets")
            mock_result = MagicMock()
            mock_result.content = "tickets result"
            return mock_result

        def mock_suggestion(results: str) -> MagicMock:
            execution_order.append("suggest_action")
            mock_result = MagicMock()
            mock_result.content = "suggestion result"
            return mock_result

        # Crear mocks de tools
        mock_tool_map = {
            "search_documents": create_mock_tool(mock_documents),
            "search_tickets": create_mock_tool(mock_tickets),
            "check_service_status": create_mock_tool(lambda x: MagicMock(content="status")),
            "suggest_action": create_mock_tool(mock_suggestion)
        }

        # Act
        with patch.dict(TOOL_MAP, mock_tool_map, clear=False):
            result = execute_tool_node(state)

        # Assert: suggest_action debe ser la última en ejecutarse
        assert "suggest_action" in execution_order
        assert execution_order[-1] == "suggest_action"
        assert len(result["tool_results"]) == 3

    def test_suggest_action_receives_all_results(self):
        """Verifica que suggest_action recibe todos los resultados de otras tools."""
        # Arrange
        state = {
            "query": "test query",
            "user_level": "basic",
            "intent": "test",
            "selected_tools": ["search_documents", "search_tickets", "suggest_action"],
            "tool_results": [],
            "final_response": None,
            "error": None
        }

        # Capturar los argumentos pasados a suggest_action
        captured_args = []

        def mock_documents(query: str) -> MagicMock:
            mock_result = MagicMock()
            mock_result.content = "documents content"
            return mock_result

        def mock_tickets(query: str) -> MagicMock:
            mock_result = MagicMock()
            mock_result.content = "tickets content"
            return mock_result

        def mock_suggestion(results: str) -> MagicMock:
            captured_args.append(results)
            mock_result = MagicMock()
            mock_result.content = "suggestion"
            return mock_result

        mock_tool_map = {
            "search_documents": create_mock_tool(mock_documents),
            "search_tickets": create_mock_tool(mock_tickets),
            "check_service_status": create_mock_tool(lambda x: MagicMock(content="status")),
            "suggest_action": create_mock_tool(mock_suggestion)
        }

        # Act
        with patch.dict(TOOL_MAP, mock_tool_map, clear=False):
            result = execute_tool_node(state)

        # Assert: suggest_action debe recibir los resultados combinados
        assert len(captured_args) == 1
        combined = captured_args[0]
        assert "search_documents" in combined
        assert "search_tickets" in combined

    def test_suggest_action_not_executed_when_alone(self):
        """Verifica que si solo se pide suggest_action, igualmente funciona."""
        # Arrange
        state = {
            "query": "test query",
            "user_level": "basic",
            "intent": "test",
            "selected_tools": ["suggest_action"],
            "tool_results": [],
            "final_response": None,
            "error": None
        }

        suggestion_called = []

        def mock_suggestion(results: str) -> MagicMock:
            suggestion_called.append(True)
            mock_result = MagicMock()
            mock_result.content = "suggestion only"
            return mock_result

        mock_tool_map = {
            "search_documents": create_mock_tool(lambda x: MagicMock(content="docs")),
            "search_tickets": create_mock_tool(lambda x: MagicMock(content="tickets")),
            "check_service_status": create_mock_tool(lambda x: MagicMock(content="status")),
            "suggest_action": create_mock_tool(mock_suggestion)
        }

        # Act
        with patch.dict(TOOL_MAP, mock_tool_map, clear=False):
            result = execute_tool_node(state)

        # Assert: debe ejecutarse suggest_action aunque no haya otros resultados
        assert len(suggestion_called) == 1
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["tool"] == "suggest_action"


class TestParallelExecution:
    """Tests para verificar ejecución paralela de tools."""

    def test_multiple_tools_run_in_parallel(self):
        """Verifica que múltiples tools (sin suggest_action) se ejecutan en paralelo."""
        # Arrange
        state = {
            "query": "test query",
            "user_level": "basic",
            "intent": "test",
            "selected_tools": ["search_documents", "search_tickets", "check_service_status"],
            "tool_results": [],
            "final_response": None,
            "error": None
        }

        def mock_documents(query: str) -> MagicMock:
            import time
            time.sleep(0.05)
            mock_result = MagicMock()
            mock_result.content = "documents"
            return mock_result

        def mock_tickets(query: str) -> MagicMock:
            import time
            time.sleep(0.05)
            mock_result = MagicMock()
            mock_result.content = "tickets"
            return mock_result

        def mock_status(service: str) -> MagicMock:
            import time
            time.sleep(0.05)
            mock_result = MagicMock()
            mock_result.content = "status"
            return mock_result

        mock_tool_map = {
            "search_documents": create_mock_tool(mock_documents),
            "search_tickets": create_mock_tool(mock_tickets),
            "check_service_status": create_mock_tool(mock_status),
            "suggest_action": create_mock_tool(lambda x: MagicMock(content="suggestion"))
        }

        # Act
        with patch.dict(TOOL_MAP, mock_tool_map, clear=False):
            result = execute_tool_node(state)

        # Assert: deben ejecutarse las 3 tools (sin suggest_action)
        assert len(result["tool_results"]) == 3

    def test_parallel_execution_collects_all_results(self):
        """Verifica que todos los resultados se recolectan correctamente."""
        # Arrange
        state = {
            "query": "test query",
            "user_level": "basic",
            "intent": "test",
            "selected_tools": ["search_documents", "search_tickets"],
            "tool_results": [],
            "final_response": None,
            "error": None
        }

        def mock_documents(query: str) -> MagicMock:
            mock_result = MagicMock()
            mock_result.content = "documents result"
            return mock_result

        def mock_tickets(query: str) -> MagicMock:
            mock_result = MagicMock()
            mock_result.content = "tickets result"
            return mock_result

        mock_tool_map = {
            "search_documents": create_mock_tool(mock_documents),
            "search_tickets": create_mock_tool(mock_tickets),
            "check_service_status": create_mock_tool(lambda x: MagicMock(content="status")),
            "suggest_action": create_mock_tool(lambda x: MagicMock(content="suggestion"))
        }

        # Act
        with patch.dict(TOOL_MAP, mock_tool_map, clear=False):
            result = execute_tool_node(state)

        # Assert: ambos resultados deben estar presentes
        tools_executed = [r["tool"] for r in result["tool_results"]]
        assert "search_documents" in tools_executed
        assert "search_tickets" in tools_executed

        # Verificar que ambos fueron exitosos
        for r in result["tool_results"]:
            assert r["success"] is True


class TestErrorHandling:
    """Tests para manejo de errores."""

    def test_partial_failure_doesnt_crash_suggestion(self):
        """Verifica que si algunas tools fallan, suggest_action igualmente intenta ejecutarse."""
        # Arrange
        state = {
            "query": "test query",
            "user_level": "basic",
            "intent": "test",
            "selected_tools": ["search_documents", "search_tickets", "suggest_action"],
            "tool_results": [],
            "final_response": None,
            "error": None
        }

        suggestion_called = []

        def mock_documents_fail(query: str) -> MagicMock:
            raise Exception("Document search failed")

        def mock_tickets(query: str) -> MagicMock:
            mock_result = MagicMock()
            mock_result.content = "tickets result"
            return mock_result

        def mock_suggestion(results: str) -> MagicMock:
            suggestion_called.append(True)
            mock_result = MagicMock()
            mock_result.content = "suggestion"
            return mock_result

        mock_tool_map = {
            "search_documents": create_mock_tool(mock_documents_fail),
            "search_tickets": create_mock_tool(mock_tickets),
            "check_service_status": create_mock_tool(lambda x: MagicMock(content="status")),
            "suggest_action": create_mock_tool(mock_suggestion)
        }

        # Act
        with patch.dict(TOOL_MAP, mock_tool_map, clear=False):
            result = execute_tool_node(state)

        # Assert: suggest_action debe ser llamada aunque search_documents falle
        assert len(suggestion_called) == 1
        assert len(result["tool_results"]) == 3

        # El resultado de search_documents debe tener success=False
        doc_result = next(r for r in result["tool_results"] if r["tool"] == "search_documents")
        assert doc_result["success"] is False

    def test_suggestion_fallback_on_error(self):
        """Verifica que si suggest_action falla, no rompe el flujo."""
        # Arrange
        state = {
            "query": "test query",
            "user_level": "basic",
            "intent": "test",
            "selected_tools": ["search_documents", "suggest_action"],
            "tool_results": [],
            "final_response": None,
            "error": None
        }

        def mock_documents(query: str) -> MagicMock:
            mock_result = MagicMock()
            mock_result.content = "documents result"
            return mock_result

        def mock_suggestion_fail(results: str) -> MagicMock:
            raise Exception("Suggestion service unavailable")

        mock_tool_map = {
            "search_documents": create_mock_tool(mock_documents),
            "search_tickets": create_mock_tool(lambda x: MagicMock(content="tickets")),
            "check_service_status": create_mock_tool(lambda x: MagicMock(content="status")),
            "suggest_action": create_mock_tool(mock_suggestion_fail)
        }

        # Act
        with patch.dict(TOOL_MAP, mock_tool_map, clear=False):
            result = execute_tool_node(state)

        # Assert: el flujo debe continuar, no romper
        assert result["error"] is None  # No debe haber error en el estado
        assert len(result["tool_results"]) == 2

        # El resultado de suggest_action debe tener success=False
        suggestion_result = next(r for r in result["tool_results"] if r["tool"] == "suggest_action")
        assert suggestion_result["success"] is False
        assert "Error" in suggestion_result["result"]


class TestSingleTool:
    """Tests para casos con una sola tool."""

    def test_single_tool_execution(self):
        """Verifica que una sola tool se ejecuta correctamente."""
        # Arrange
        state = {
            "query": "test query",
            "user_level": "basic",
            "intent": "test",
            "selected_tools": ["search_documents"],
            "tool_results": [],
            "final_response": None,
            "error": None
        }

        def mock_documents(query: str) -> MagicMock:
            mock_result = MagicMock()
            mock_result.content = "single document result"
            return mock_result

        mock_tool_map = {
            "search_documents": create_mock_tool(mock_documents),
            "search_tickets": create_mock_tool(lambda x: MagicMock(content="tickets")),
            "check_service_status": create_mock_tool(lambda x: MagicMock(content="status")),
            "suggest_action": create_mock_tool(lambda x: MagicMock(content="suggestion"))
        }

        # Act
        with patch.dict(TOOL_MAP, mock_tool_map, clear=False):
            result = execute_tool_node(state)

        # Assert
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["tool"] == "search_documents"
        assert result["tool_results"][0]["success"] is True


class TestEdgeCases:
    """Tests para casos edge."""

    def test_empty_selected_tools(self):
        """Verifica comportamiento con lista vacía de tools."""
        # Arrange
        state = {
            "query": "test query",
            "user_level": "basic",
            "intent": "test",
            "selected_tools": [],
            "tool_results": [],
            "final_response": None,
            "error": None
        }

        # Act
        result = execute_tool_node(state)

        # Assert
        assert result["error"] == "No tool selected"
        assert result["tool_results"] == []

    def test_invalid_tools_ignored(self):
        """Verifica que tools inválidas son ignoradas."""
        # Arrange
        state = {
            "query": "test query",
            "user_level": "basic",
            "intent": "test",
            "selected_tools": ["invalid_tool", "search_documents"],
            "tool_results": [],
            "final_response": None,
            "error": None
        }

        def mock_documents(query: str) -> MagicMock:
            mock_result = MagicMock()
            mock_result.content = "documents"
            return mock_result

        mock_tool_map = {
            "search_documents": create_mock_tool(mock_documents),
            "search_tickets": create_mock_tool(lambda x: MagicMock(content="tickets")),
            "check_service_status": create_mock_tool(lambda x: MagicMock(content="status")),
            "suggest_action": create_mock_tool(lambda x: MagicMock(content="suggestion"))
        }

        # Act
        with patch.dict(TOOL_MAP, mock_tool_map, clear=False):
            result = execute_tool_node(state)

        # Assert: solo search_documents debe ejecutarse
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["tool"] == "search_documents"


class TestExecutionOrder:
    """Tests específicos para verificar orden de ejecución."""

    def test_order_with_three_tools_and_suggestion(self):
        """Verifica el orden exacto: doc -> tickets -> suggestion."""
        # Arrange
        state = {
            "query": "test query",
            "user_level": "basic",
            "intent": "test",
            "selected_tools": ["search_documents", "search_tickets", "suggest_action"],
            "tool_results": [],
            "final_response": None,
            "error": None
        }

        execution_log = []

        def log_and_return(name, content):
            def wrapper(arg):
                execution_log.append(name)
                mock_result = MagicMock()
                mock_result.content = content
                return mock_result
            return wrapper

        mock_tool_map = {
            "search_documents": create_mock_tool(log_and_return("doc", "documents")),
            "search_tickets": create_mock_tool(log_and_return("tickets", "tickets")),
            "check_service_status": create_mock_tool(lambda x: MagicMock(content="status")),
            "suggest_action": create_mock_tool(log_and_return("suggestion", "suggestion"))
        }

        # Act
        with patch.dict(TOOL_MAP, mock_tool_map, clear=False):
            result = execute_tool_node(state)

        # Assert: el log debe contener las 3 ejecuciones en orden
        assert len(execution_log) == 3
        assert execution_log[0] == "doc"
        assert execution_log[1] == "tickets"
        assert execution_log[2] == "suggestion"

    def test_parallel_tools_complete_before_suggestion(self):
        """Verifica que las tools paralelas terminan antes de suggest_action."""
        # Arrange
        state = {
            "query": "test query",
            "user_level": "basic",
            "intent": "test",
            "selected_tools": ["search_documents", "search_tickets", "check_service_status", "suggest_action"],
            "tool_results": [],
            "final_response": None,
            "error": None
        }

        completion_log = []

        def log_completion(name):
            def wrapper(arg):
                import time
                time.sleep(0.02)  # Simula trabajo
                completion_log.append(f"{name}_done")
                mock_result = MagicMock()
                mock_result.content = f"{name} result"
                return mock_result
            return wrapper

        mock_tool_map = {
            "search_documents": create_mock_tool(log_completion("doc")),
            "search_tickets": create_mock_tool(log_completion("tickets")),
            "check_service_status": create_mock_tool(log_completion("status")),
            "suggest_action": create_mock_tool(log_completion("suggestion"))
        }

        # Act
        with patch.dict(TOOL_MAP, mock_tool_map, clear=False):
            result = execute_tool_node(state)

        # Assert: suggestion debe ser la última
        assert "suggestion_done" in completion_log
        assert completion_log[-1] == "suggestion_done"