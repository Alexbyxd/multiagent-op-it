"""Tests para el context manager de base de datos."""
import pytest
import sqlite3
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

from src.db.queries import get_connection, search_tickets
from src.exceptions import DatabaseError


class TestGetConnection:
    """Tests para la función get_connection."""

    def test_get_connection_returns_connection(self):
        """Verifica que get_connection retorna una conexión válida."""
        with get_connection(":memory:") as conn:
            assert conn is not None
            assert isinstance(conn, sqlite3.Connection)

    def test_get_connection_sets_row_factory(self):
        """Verifica que get_connection configura row_factory a sqlite3.Row."""
        with get_connection(":memory:") as conn:
            assert conn.row_factory == sqlite3.Row

    def test_connection_closes_on_success(self):
        """Verifica que la conexión se cierra después de uso exitoso."""
        # Patch de sqlite3.connect para devolver un mock
        mock_conn = MagicMock(spec=sqlite3.Connection)

        with patch("sqlite3.connect", return_value=mock_conn):
            with get_connection(":memory:") as conn:
                pass  # Simular uso exitoso

        mock_conn.close.assert_called_once()

    def test_connection_closes_on_exception(self):
        """Verifica que la conexión se cierra incluso cuando hay excepción."""
        # Patch de sqlite3.connect para devolver un mock
        mock_conn = MagicMock(spec=sqlite3.Connection)

        with patch("sqlite3.connect", return_value=mock_conn):
            with pytest.raises(ValueError):
                with get_connection(":memory:") as conn:
                    raise ValueError("Error de test")

        mock_conn.close.assert_called_once()

    def test_get_connection_raises_on_error(self):
        """Verifica que lanza DatabaseError cuando no puede conectar."""
        with pytest.raises(DatabaseError) as exc_info:
            # Usar un path inválido para forzar error
            with get_connection("/nonexistent/path/to/db.sqlite"):
                pass

        assert "Error conectando a la base de datos" in str(exc_info.value)


class TestSearchTickets:
    """Tests para la función search_tickets."""

    def test_search_tickets_uses_context_manager(self, tmp_path):
        """Verifica que search_tickets usa get_connection."""
        db_path = tmp_path / "test.db"

        # Crear DB con datos de prueba
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
            (1, 'Error 503', 'Servidor caído', '503', 'Reiniciar nginx', 'critical', '2024-01-01', '2024-01-01')
        """)
        conn.commit()
        conn.close()

        # Verificar que usa context manager (la conexión se cierra correctamente)
        with patch("src.db.queries.get_connection") as mock_get_conn:
            mock_conn = MagicMock(spec=sqlite3.Connection)
            mock_cursor = MagicMock()
            mock_cursor.fetchall.return_value = []
            mock_conn.cursor.return_value = mock_cursor

            mock_get_conn.return_value.__enter__ = Mock(return_value=mock_conn)
            mock_get_conn.return_value.__exit__ = Mock(return_value=False)

            search_tickets("test", db_path=db_path)

            mock_get_conn.assert_called_once()

    def test_search_tickets_returns_results(self, tmp_path):
        """Verifica que search_tickets retorna los resultados correctos."""
        db_path = tmp_path / "test.db"

        # Crear DB con datos de prueba
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
            (1, 'Error 503', 'Servidor caído', '503', 'Reiniciar nginx', 'critical', '2024-01-01 10:00:00', '2024-01-01 11:00:00'),
            (2, 'DB timeout', 'Consulta lenta', 'DB_TIMEOUT', 'Optimizar query', 'high', '2024-01-02 10:00:00', '2024-01-02 11:00:00'),
            (3, 'Memory leak', 'Fuga de memoria', 'MEM_LEAK', 'Aumentar RAM', 'medium', '2024-01-03 10:00:00', '2024-01-03 11:00:00')
        """)
        conn.commit()
        conn.close()

        # Buscar por "Error"
        results = search_tickets("Error", limit=5, db_path=db_path)

        assert len(results) == 1
        assert results[0]["title"] == "Error 503"
        assert results[0]["error_code"] == "503"
        assert results[0]["severity"] == "critical"

    def test_search_tickets_handles_empty_results(self, tmp_path):
        """Verifica que search_tickets retorna lista vacía cuando no hay resultados."""
        db_path = tmp_path / "test.db"

        # Crear DB vacía
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
        conn.commit()
        conn.close()

        results = search_tickets("inexistente", limit=5, db_path=db_path)

        assert results == []
        assert isinstance(results, list)

    def test_search_tickets_respects_limit(self, tmp_path):
        """Verifica que search_tickets respeta el límite de resultados."""
        db_path = tmp_path / "test.db"

        # Crear DB con varios tickets
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
        # Insertar 10 tickets de severity different para tener resultados
        for i in range(10):
            conn.execute(
                "INSERT INTO tickets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (i + 1, f"Error {i}", f"Descripción {i}", f"ERR_{i}", f"Solución {i}", "low", "2024-01-01 10:00:00", "2024-01-01 11:00:00")
            )
        conn.commit()
        conn.close()

        results = search_tickets("Error", limit=3, db_path=db_path)

        assert len(results) <= 3