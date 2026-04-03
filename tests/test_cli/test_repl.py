"""Tests for MASORePL class and REPL functionality."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rich.panel import Panel

from src.cli.repl import MAX_HISTORY_ENTRIES, MASORePL, run_simple_mode


class TestSlashCommandParsing:
    """Tests for slash command parsing in _handle_command()."""

    def test_handle_help_command(self, repl: MASORePL) -> None:
        """_handle_command('/help') should call _show_help and return True."""
        repl._show_help = MagicMock()  # type: ignore[assignment]
        result = repl._handle_command("/help")
        repl._show_help.assert_called_once()  # type: ignore[attr-defined]
        assert result is True

    def test_handle_clear_command(self, repl: MASORePL) -> None:
        """_handle_command('/clear') should call console.clear() and return True."""
        repl.console.clear = MagicMock()
        result = repl._handle_command("/clear")
        repl.console.clear.assert_called_once()
        assert result is True

    def test_handle_exit_command(self, repl: MASORePL) -> None:
        """_handle_command('/exit') should return False."""
        result = repl._handle_command("/exit")
        assert result is False

    def test_handle_normal_query(self, repl: MASORePL) -> None:
        """_handle_command('hello') should call _process_query and return True."""
        repl._process_query = MagicMock()  # type: ignore[assignment]
        result = repl._handle_command("hello")
        repl._process_query.assert_called_once_with("hello")  # type: ignore[attr-defined]
        assert result is True

    def test_handle_history_command(self, repl: MASORePL) -> None:
        """_handle_command('/history') should call _show_history and return True."""
        repl._show_history = MagicMock()  # type: ignore[assignment]
        result = repl._handle_command("/history")
        repl._show_history.assert_called_once_with(["/history"])  # type: ignore[attr-defined]
        assert result is True


class TestInvalidSlashCommand:
    """Tests for invalid slash command handling."""

    def test_invalid_command_shows_error(self, repl: MASORePL) -> None:
        """Invalid slash command should show error panel."""
        repl.console.print = MagicMock()  # type: ignore[assignment]
        result = repl._handle_command("/foobar")
        repl.console.print.assert_called_once()  # type: ignore[attr-defined]
        assert result is True

    def test_invalid_command_returns_true(self, repl: MASORePL) -> None:
        """Invalid slash command should not exit the REPL."""
        result = repl._handle_command("/unknown")
        assert result is True


class TestLevelCommand:
    """Tests for /level command validation."""

    def test_level_valid_basic(self, repl: MASORePL) -> None:
        """Valid level 'basic' should update user_level."""
        repl.console.print = MagicMock()  # type: ignore[assignment]
        result = repl._handle_command("/level basic")
        assert repl.user_level == "basic"
        assert result is True

    def test_level_valid_advanced(self, repl: MASORePL) -> None:
        """Valid level 'advanced' should update user_level."""
        repl.console.print = MagicMock()  # type: ignore[assignment]
        result = repl._handle_command("/level advanced")
        assert repl.user_level == "advanced"
        assert result is True

    def test_level_valid_admin(self, repl: MASORePL) -> None:
        """Valid level 'admin' should update user_level."""
        repl.console.print = MagicMock()  # type: ignore[assignment]
        result = repl._handle_command("/level admin")
        assert repl.user_level == "admin"
        assert result is True

    def test_level_invalid(self, repl: MASORePL) -> None:
        """Invalid level should show error and not change user_level."""
        repl.console.print = MagicMock()  # type: ignore[assignment]
        old_level = repl.user_level
        result = repl._handle_command("/level superadmin")
        assert repl.user_level == old_level
        repl.console.print.assert_called_once()  # type: ignore[attr-defined]
        assert result is True

    def test_level_no_argument(self, repl: MASORePL) -> None:
        """Missing level argument should show usage."""
        repl.console.print = MagicMock()  # type: ignore[assignment]
        result = repl._handle_command("/level")
        repl.console.print.assert_called_once()  # type: ignore[attr-defined]
        assert result is True

    def test_level_case_insensitive(self, repl: MASORePL) -> None:
        """Level should be case-insensitive."""
        repl.console.print = MagicMock()  # type: ignore[assignment]
        result = repl._handle_command("/level ADMIN")
        assert repl.user_level == "admin"
        assert result is True


class TestHistoryManagement:
    """Tests for history save/load/persist functionality."""

    def test_add_to_history(self, repl: MASORePL) -> None:
        """_add_to_history should append entry to history list."""
        repl._add_to_history("test query", "test response")
        assert len(repl.history) == 1
        assert repl.history[0]["query"] == "test query"
        assert "test response" in repl.history[0]["response"]
        assert "timestamp" in repl.history[0]

    def test_save_and_load_history(
        self, repl: MASORePL, tmp_path: Path
    ) -> None:
        """History should persist to JSON and load correctly."""
        history_file = tmp_path / "test_history.json"
        repl.session_history_file = history_file

        repl._add_to_history("query1", "response1")
        repl._add_to_history("query2", "response2")
        repl._save_history()

        # Create new REPL with same file
        new_repl = MASORePL(
            user_level="basic",
            history_file=tmp_path / ".maso_history",
            session_history_file=history_file,
        )
        assert len(new_repl.history) == 2
        assert new_repl.history[0]["query"] == "query1"
        assert new_repl.history[1]["query"] == "query2"

    def test_history_cap_at_max_entries(
        self, repl: MASORePL, tmp_path: Path
    ) -> None:
        """History should cap at MAX_HISTORY_ENTRIES."""
        history_file = tmp_path / "test_cap.json"
        repl.session_history_file = history_file

        # Add more than MAX_HISTORY_ENTRIES
        for i in range(MAX_HISTORY_ENTRIES + 100):
            repl._add_to_history(f"query {i}", f"response {i}")

        repl._save_history()

        # Reload and check cap
        new_repl = MASORePL(
            user_level="basic",
            history_file=tmp_path / ".maso_history",
            session_history_file=history_file,
        )
        assert len(new_repl.history) == MAX_HISTORY_ENTRIES
        # Should keep the most recent entries
        assert new_repl.history[0]["query"] == f"query {100}"

    def test_load_corrupted_history(
        self, repl: MASORePL, tmp_path: Path
    ) -> None:
        """Corrupted JSON file should result in empty history."""
        history_file = tmp_path / "corrupted.json"
        history_file.write_text("not valid json {{{")
        repl.session_history_file = history_file

        repl._load_history()
        assert repl.history == []

    def test_load_nonexistent_history(self, repl: MASORePL) -> None:
        """Non-existent history file should not raise error."""
        repl.session_history_file = Path("/nonexistent/path/history.json")
        repl._load_history()  # Should not raise
        assert repl.history == []

    def test_load_history_wrong_format(
        self, repl: MASORePL, tmp_path: Path
    ) -> None:
        """History file with wrong format (not a list) should reset."""
        history_file = tmp_path / "wrong_format.json"
        history_file.write_text('{"key": "value"}')
        repl.session_history_file = history_file

        repl._load_history()
        assert repl.history == []


class TestNonTTYFallback:
    """Tests for non-TTY fallback mode."""

    def test_run_simple_mode_exits_on_exit(self, capsys) -> None:
        """Simple mode should exit on 'exit' command."""
        with patch("builtins.input", side_effect=["exit"]):
            with patch("src.graph.maso_graph.run_agent"):
                run_simple_mode("basic")

        captured = capsys.readouterr()
        assert "MASO" in captured.out
        assert "Hasta luego" in captured.out

    def test_run_simple_mode_processes_query(self, capsys) -> None:
        """Simple mode should process queries and print responses."""
        with patch(
            "builtins.input",
            side_effect=["hello", "exit"],
        ):
            with patch(
                "src.graph.maso_graph.run_agent",
                return_value="Hello back!",
            ):
                run_simple_mode("basic")

        captured = capsys.readouterr()
        assert "Hello back!" in captured.out

    def test_run_simple_mode_skips_empty(self, capsys) -> None:
        """Simple mode should skip empty input."""
        with patch(
            "builtins.input",
            side_effect=["", "   ", "hello", "exit"],
        ):
            with patch(
                "src.graph.maso_graph.run_agent",
                return_value="Response",
            ):
                run_simple_mode("basic")

        captured = capsys.readouterr()
        # Should only have one "Procesando" message
        assert captured.out.count("Procesando") == 1

    def test_run_simple_mode_handles_error(self, capsys) -> None:
        """Simple mode should handle agent errors gracefully."""
        with patch(
            "builtins.input",
            side_effect=["query", "exit"],
        ):
            with patch(
                "src.graph.maso_graph.run_agent",
                side_effect=Exception("API error"),
            ):
                run_simple_mode("basic")

        captured = capsys.readouterr()
        assert "Error" in captured.out


class TestWelcomeScreen:
    """Tests for welcome screen generation."""

    def test_show_welcome_renders_panel(self, repl: MASORePL) -> None:
        """_show_welcome should call console.print with a Panel."""
        repl.console.print = MagicMock()  # type: ignore[assignment]
        repl._show_welcome()
        repl.console.print.assert_called_once()  # type: ignore[attr-defined]
        # Verify the argument is actually a Panel instance
        call_args = repl.console.print.call_args[0][0]  # type: ignore[attr-defined]
        assert isinstance(call_args, Panel)


class TestSpinnerIntegration:
    """Tests for spinner integration with run_agent."""

    def test_run_with_spinner_calls_run_agent(
        self, repl: MASORePL
    ) -> None:
        """_process_query should call run_agent with correct args."""
        repl.console.print = MagicMock()  # type: ignore[assignment]
        repl.console.status = MagicMock()
        repl.console.status.return_value.__enter__ = MagicMock(
            return_value=None,
        )
        repl.console.status.return_value.__exit__ = MagicMock(
            return_value=False,
        )

        with patch(
            "src.graph.maso_graph.run_agent",
            return_value="Agent response",
        ) as mock_agent:
            repl._process_query("test query")
            mock_agent.assert_called_once_with("test query", "basic")

    def test_run_with_spinner_handles_error(
        self, repl: MASORePL
    ) -> None:
        """_process_query should handle run_agent exceptions."""
        repl.console.print = MagicMock()  # type: ignore[assignment]
        repl.console.status = MagicMock()
        repl.console.status.return_value.__enter__ = MagicMock(
            return_value=None,
        )
        repl.console.status.return_value.__exit__ = MagicMock(
            return_value=False,
        )

        with patch(
            "src.graph.maso_graph.run_agent",
            side_effect=Exception("Connection failed"),
        ):
            repl._process_query("test query")

        # Should have called print at least twice (error panel)
        assert repl.console.print.call_count >= 1  # type: ignore[attr-defined]


class TestIntegration:
    """Integration tests for full REPL cycle."""

    def test_full_repl_cycle_with_mocked_input(
        self,
        temp_history_file: Path,
        temp_session_history_file: Path,
        capsys,
    ) -> None:
        """Simulate: welcome → /help → query → /exit."""
        from src.cli.repl import MASORePL

        repl = MASORePL(
            user_level="basic",
            history_file=temp_history_file,
            session_history_file=temp_session_history_file,
        )

        # Mock the session.prompt to simulate user input
        call_count = 0

        def mock_prompt(prompt_text: str = "> ") -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "/help"
            elif call_count == 2:
                return "What is the status?"
            elif call_count == 3:
                return "/exit"
            return "/exit"

        repl.session.prompt = mock_prompt  # type: ignore[assignment]

        with patch(
            "src.graph.maso_graph.run_agent",
            return_value="The status is **OK**.",
        ):
            repl.run()

        assert call_count == 3
        # History should have one entry from the query
        assert len(repl.history) == 1
        assert repl.history[0]["query"] == "What is the status?"

    def test_session_history_persists_after_exit(
        self,
        temp_history_file: Path,
        temp_session_history_file: Path,
    ) -> None:
        """Session history should be saved to JSON after REPL exits."""
        repl = MASORePL(
            user_level="basic",
            history_file=temp_history_file,
            session_history_file=temp_session_history_file,
        )

        call_count = 0

        def mock_prompt(prompt_text: str = "> ") -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Test query"
            return "/exit"

        repl.session.prompt = mock_prompt  # type: ignore[assignment]

        with patch(
            "src.graph.maso_graph.run_agent",
            return_value="Test response",
        ):
            repl.run()

        # Verify JSON file exists and has content
        assert temp_session_history_file.exists()
        with open(temp_session_history_file, encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["query"] == "Test query"
