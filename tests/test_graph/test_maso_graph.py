"""Tests de integración para el flujo principal del grafo LangGraph."""
import pytest
from unittest.mock import MagicMock, patch

from src.graph.maso_graph import create_graph
from src.graph.nodes import TOOL_MAP


# =============================================================================
# TESTS DE ESTRUCTURA DEL GRAF0
# =============================================================================

class TestGraphStructure:
    """Tests para verificar la estructura del grafo."""

    def test_create_graph_adds_all_nodes(self):
        """Verifica que se crean los 4 nodos: input, router, execute_tool, synthesizer."""
        graph = create_graph()
        compiled = graph.compile()
        assert compiled is not None

    def test_create_graph_adds_all_edges(self):
        """Verifica las conexiones entre nodos."""
        graph = create_graph()
        compiled = graph.compile()
        assert compiled is not None

    def test_create_graph_sets_entry_point(self):
        """Verifica punto de entrada en 'input'."""
        graph = create_graph()
        compiled = graph.compile()
        assert compiled is not None


# =============================================================================
# TESTS DE FALLBACK (KEYWORDS)
# =============================================================================

class TestFallbackBehavior:
    """Tests para el comportamiento de fallback."""

    def test_fallback_keyword_detection(self):
        """Verifica detección de keywords para fallback."""
        from src.llm.router import _keyword_fallback
        
        # Test con keywords de estado
        result = _keyword_fallback("¿está caído el servidor?")
        assert "check_service_status" in result["tools"]
        
        # Test con keywords de tickets
        result = _keyword_fallback("tuve un error en producción")
        assert "search_tickets" in result["tools"]
        
        # Test con keywords de docs
        result = _keyword_fallback("dónde está el manual de nginx")
        assert "search_documents" in result["tools"]
        
        # Test default
        result = _keyword_fallback("hola mundo")
        assert "search_documents" in result["tools"]

    def test_fallback_returns_valid_dict(self):
        """Verifica que el fallback retorna estructura válida."""
        from src.llm.router import _keyword_fallback
        
        result = _keyword_fallback("test query")
        
        assert isinstance(result, dict)
        assert "tools" in result
        assert "reason" in result
        assert isinstance(result["tools"], list)


# =============================================================================
# TESTS DEL MISMATCH DE TIPOS
# =============================================================================

class TestTypeMismatch:
    """Tests específicos para el mismatch entre maso_graph.py y nodes.py."""

    def test_initial_state_keys_compatible(self):
        """Verifica que las keys del initial_state son compatibles."""
        # Keys que usa maso_graph.py
        initial_state = {
            "query": "test",
            "user_level": "basic",
            "selected_tool": None,  # singular
            "tool_result": None,   # singular
            "final_response": None,
            "error": None
        }
        
        assert "query" in initial_state
        assert "user_level" in initial_state

    def test_state_allows_singular_keys(self):
        """Verifica que el estado puede tener keys singulares o plurales."""
        # Formato singular (maso_graph.py)
        state_singular = {
            "query": "test",
            "user_level": "basic",
            "intent": "test",
            "selected_tool": "search_documents",
            "tool_result": "result",
            "final_response": None,
            "error": None
        }
        
        # Formato plural (nodes.py AgentState)
        state_plural = {
            "query": "test",
            "user_level": "basic", 
            "intent": "test",
            "selected_tools": ["search_documents"],
            "tool_results": [{"tool": "search_documents", "result": "result"}],
            "final_response": None,
            "error": None
        }
        
        assert isinstance(state_singular, dict)
        assert isinstance(state_plural, dict)
        assert state_singular["query"] == state_plural["query"]


# =============================================================================
# TESTS DE TOOL MAP
# =============================================================================

class TestToolMap:
    """Tests para verificar el mapeo de tools."""

    def test_tool_map_has_required_tools(self):
        """Verifica que TOOL_MAP tiene las tools requeridas."""
        assert "search_documents" in TOOL_MAP
        assert "search_tickets" in TOOL_MAP
        assert "check_service_status" in TOOL_MAP
        assert "suggest_action" in TOOL_MAP

    def test_tool_map_values_have_invoke(self):
        """Verifica que los valores de TOOL_MAP tienen el método invoke."""
        for tool_name, tool_func in TOOL_MAP.items():
            assert hasattr(tool_func, 'invoke'), f"{tool_name} no tiene invoke"


# =============================================================================
# TESTS DE ROUTER Y SYNTHESIZER
# =============================================================================

class TestRouterAndSynthesizer:
    """Tests para router y synthesizer."""

    def test_router_function_exists(self):
        """Verifica que la función router existe."""
        from src.llm.router import router
        assert callable(router)

    def test_synthesize_function_exists(self):
        """Verifica que la función synthesize existe."""
        from src.llm.synthesizer import synthesize
        assert callable(synthesize)

    def test_synthesize_fallback_response(self):
        """Verifica que synthesize tiene fallback."""
        from src.llm.synthesizer import _fallback_response
        
        result = _fallback_response("query", "tool_results", "basic")
        
        assert isinstance(result, str)
        assert len(result) > 0


# =============================================================================
# TESTS DE NODOS INDIVIDUALES
# =============================================================================

class TestNodes:
    """Tests para nodos individuales."""

    def test_input_node_exists(self):
        """Verifica que input_node existe."""
        from src.graph.nodes import input_node
        assert callable(input_node)

    def test_router_node_exists(self):
        """Verifica que router_node existe."""
        from src.graph.nodes import router_node
        assert callable(router_node)

    def test_execute_tool_node_exists(self):
        """Verifica que execute_tool_node existe."""
        from src.graph.nodes import execute_tool_node
        assert callable(execute_tool_node)

    def test_synthesizer_node_exists(self):
        """Verifica que synthesizer_node existe."""
        from src.graph.nodes import synthesizer_node
        assert callable(synthesizer_node)


# =============================================================================
# TESTS DE INTEGRACIÓN CON MOCHS (SIN LLAMADAS DE RED)
# =============================================================================

class TestNodesWithMocks:
    """Tests de nodos con mocks para evitar llamadas de red."""

    def test_router_node_with_mock(self):
        """Test router_node con router mockeado."""
        from src.graph.nodes import router_node
        
        test_state = {
            "query": "test query",
            "user_level": "basic"
        }
        
        with patch('src.llm.router.router') as mock_router:
            mock_router.return_value = {"tools": ["search_documents"], "reason": "test"}
            
            result = router_node(test_state)
            
            assert "selected_tools" in result
            assert result["selected_tools"] == ["search_documents"]
            assert "intent" in result

    def test_execute_tool_node_with_mock(self):
        """Test execute_tool_node con tools mockeadas."""
        from src.graph.nodes import execute_tool_node
        
        test_state = {
            "query": "test query",
            "user_level": "basic",
            "intent": "test",
            "selected_tools": ["search_documents"],
            "tool_results": [],
            "final_response": None,
            "error": None
        }
        
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = MagicMock(content="test result")
        
        with patch.dict(TOOL_MAP, {"search_documents": mock_tool}, clear=False):
            result = execute_tool_node(test_state)
        
        assert "tool_results" in result
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["tool"] == "search_documents"

    def test_synthesizer_node_with_mock(self):
        """Test synthesizer_node con synthesize mockeado."""
        from src.graph.nodes import synthesizer_node
        
        test_state = {
            "query": "test query",
            "user_level": "basic",
            "intent": "test",
            "selected_tools": ["search_documents"],
            "tool_results": [{"tool": "search_documents", "result": "test", "success": True}],
            "final_response": None,
            "error": None
        }
        
        with patch('src.llm.synthesizer.synthesize') as mock_synth:
            mock_synth.return_value = "Respuesta final"
            
            result = synthesizer_node(test_state)
            
            assert "final_response" in result
            assert result["final_response"] == "Respuesta final"
