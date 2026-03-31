"""Tests para el router."""
import pytest
import json
from unittest.mock import patch, MagicMock


def test_router_keywords_status():
    """Verifica que selecciona check_service_status para consultas de estado."""
    from src.llm.router import router
    
    with patch('src.llm.router.get_router_llm') as mock_llm:
        mock_response = MagicMock()
        mock_response.content = '{"tool": "check_service_status", "reason": "Consulta sobre estado de servicio"}'
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = router.invoke("¿Está caído el servidor web?")
        
        assert "check_service_status" in result


def test_router_keywords_documents():
    """Verifica que selecciona search_documents para consultas de docs."""
    from src.llm.router import router
    
    with patch('src.llm.router.get_router_llm') as mock_llm:
        mock_response = MagicMock()
        mock_response.content = '{"tool": "search_documents", "reason": "Consulta sobre documentación"}'
        mock_llm.return_value.invoke.return_value = mock_response
        
        result = router.invoke("¿Dónde está el manual de nginx?")
        
        assert "search_documents" in result