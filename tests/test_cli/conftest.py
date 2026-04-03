"""Fixtures for CLI REPL tests."""
import json
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.cli.repl import MASORePL


@pytest.fixture
def temp_history_file(tmp_path: Path) -> Path:
    """Temporary file for FileHistory."""
    return tmp_path / ".maso_history"


@pytest.fixture
def temp_session_history_file(tmp_path: Path) -> Path:
    """Temporary file for session history JSON."""
    return tmp_path / ".maso_sessions.json"


@pytest.fixture
def repl(
    temp_history_file: Path, temp_session_history_file: Path
) -> MASORePL:
    """MASORePL instance with temp files."""
    return MASORePL(
        user_level="basic",
        history_file=temp_history_file,
        session_history_file=temp_session_history_file,
    )


@pytest.fixture
def repl_with_history(
    temp_history_file: Path,
    temp_session_history_file: Path,
) -> MASORePL:
    """MASORePL instance pre-loaded with history entries."""
    history_data = [
        {
            "query": "How do I reset my password?",
            "response": "To reset your password, go to settings...",
            "timestamp": "2024-01-15T10:30:00+00:00",
        },
        {
            "query": "What is the status of ticket #123?",
            "response": "Ticket #123 is currently in progress.",
            "timestamp": "2024-01-15T11:00:00+00:00",
        },
    ]
    with open(temp_session_history_file, "w", encoding="utf-8") as f:
        json.dump(history_data, f)

    return MASORePL(
        user_level="advanced",
        history_file=temp_history_file,
        session_history_file=temp_session_history_file,
    )
