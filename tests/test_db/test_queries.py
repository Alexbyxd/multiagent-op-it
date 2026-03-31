"""Tests para queries de tickets."""
import pytest
import sqlite3
from src.db.queries import search_tickets


def test_search_tickets_returns_results(tmp_path):
    """Verifica que search_tickets retorna resultados."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE tickets (
            id INTEGER PRIMARY KEY,
            title TEXT,
            description TEXT,
            error_code TEXT,
            solution TEXT,
            severity TEXT,
            created_at DATETIME,
            resolved_at DATETIME
        )
    """)
    conn.execute("""
        INSERT INTO tickets VALUES 
        (1, 'Error 503', 'Servidor caído', '503', 'Reiniciar nginx', 'critical', '2024-01-01', '2024-01-01'),
        (2, 'DB timeout', 'Consulta lenta', 'DB_TIMEOUT', 'Optimizar query', 'high', '2024-01-02', '2024-01-02')
    """)
    conn.commit()
    conn.close()
    
    results = search_tickets("servidor caído", db_path=db_path)
    
    assert len(results) > 0
    assert results[0]["error_code"] == "503"
