"""Tests para funciones de sanitización."""
import pytest
from src.utils.sanitize import sanitize_text, sanitize_ticket_data, sanitize_document_result


class TestSanitizeText:
    """Tests para sanitize_text."""

    def test_sanitize_text_escapes_html(self):
        """Verifica que <script> se convierte en &lt;script&gt;."""
        result = sanitize_text("<script>alert('xss')</script>")
        assert "&lt;script&gt;" in result

    def test_sanitize_text_removes_script_tags(self):
        """Verifica que tags <script>...</script> se eliminan."""
        result = sanitize_text('<script>malicious code</script> normal text')
        # Verifica que el tag HTML no está presente (puede estar escapado o removido)
        assert "<script" not in result.lower()
        assert "normal text" in result

    def test_sanitize_text_removes_javascript_protocol(self):
        """Verifica que javascript: se elimina."""
        result = sanitize_text('Click <a href="javascript:alert(1)">here</a>')
        assert "javascript:" not in result.lower()

    def test_sanitize_text_removes_event_handlers(self):
        """Verifica que onclick=, onerror= se eliminan."""
        result = sanitize_text('<img onclick="bad()" onerror="evil()">')
        assert "onclick" not in result.lower()
        assert "onerror" not in result.lower()

    def test_sanitize_text_truncates_long_text(self):
        """Verifica que texto muy largo se trunca."""
        long_text = "a" * 15000
        result = sanitize_text(long_text, max_length=10000)
        assert len(result) == 10003  # 10000 + "..."
        assert result.endswith("...")

    def test_sanitize_text_handles_empty_string(self):
        """Verifica que maneja string vacío."""
        assert sanitize_text("") == ""
        assert sanitize_text(None) == ""


class TestSanitizeTicketData:
    """Tests para sanitize_ticket_data."""

    def test_sanitize_ticket_data_sanitizes_all_fields(self):
        """Verifica que todos los campos de texto se sanitizan."""
        ticket = {
            "id": 1,
            "title": "<script>alert('xss')</script>",
            "description": "Click <a href='javascript:bad()'>here</a>",
            "error_code": "<img onerror='evil'>",
            "solution": "Normal solution",
            "severity": "high",
        }
        result = sanitize_ticket_data(ticket)

        assert "<script>" not in result["title"]
        assert "javascript:" not in result["description"]
        assert "onerror" not in result["error_code"]
        assert result["solution"] == "Normal solution"

    def test_sanitize_ticket_data_preserves_non_string_fields(self):
        """Verifica que campos como id, severity no se modifican."""
        ticket = {
            "id": 42,
            "title": "Normal title",
            "description": "Normal description",
            "severity": 5,
            "created_at": "2024-01-01",
        }
        result = sanitize_ticket_data(ticket)

        assert result["id"] == 42
        assert result["severity"] == 5
        assert result["created_at"] == "2024-01-01"


class TestSanitizeDocumentResult:
    """Tests para sanitize_document_result."""

    def test_sanitize_document_result_sanitizes_text_and_source(self):
        """Verifica que text y source se sanitizan."""
        result = {
            "text": "<script>alert('xss')</script> contenido",
            "source": "http://evil.com<script>",
            "score": 0.95,
            "metadata": {"key": "value"},
        }
        sanitized = sanitize_document_result(result)

        assert "<script>" not in sanitized["text"]
        assert "<script>" not in sanitized["source"]
        assert "contenido" in sanitized["text"]

    def test_sanitize_document_result_preserves_metadata(self):
        """Verifica que campos numéricos se preservan."""
        result = {
            "text": "Some text",
            "source": "doc.txt",
            "score": 0.85,
            "page": 3,
            "metadata": {"author": "test"},
        }
        sanitized = sanitize_document_result(result)

        assert sanitized["score"] == 0.85
        assert sanitized["page"] == 3
        assert sanitized["metadata"] == {"author": "test"}