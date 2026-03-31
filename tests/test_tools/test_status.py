"""Tests para tool de status."""
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_check_service_status_returns_status(tmp_path):
    """Verifica que retorna el estado de un servicio."""
    from src.tools.status import check_service_status
    
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({
        "services": [
            {"name": "web-server", "display_name": "Web Server", "status": "UP", "uptime": "99.9%", "last_check": "2024-01-01"}
        ]
    }))
    
    with patch('src.tools.status.settings') as mock_settings:
        mock_settings.STATUS_FILE = status_file
        result = check_service_status("web-server")
    
    assert "UP" in result
    assert "web-server" in result.lower()


def test_check_service_status_not_found(tmp_path):
    """Verifica comportamiento cuando servicio no existe."""
    from src.tools.status import check_service_status
    
    status_file = tmp_path / "status.json"
    status_file.write_text(json.dumps({"services": [{"name": "web-server", "status": "UP"}]}))
    
    with patch('src.tools.status.settings') as mock_settings:
        mock_settings.STATUS_FILE = status_file
        result = check_service_status("nonexistent")
    
    assert "no encontrado" in result.lower()