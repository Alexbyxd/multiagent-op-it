"""REPL interactivo para MASO con Rich y Prompt Toolkit."""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

logger = logging.getLogger(__name__)

# Maximum entries in session history JSON
MAX_HISTORY_ENTRIES = 500

# Valid user levels
VALID_LEVELS = ("basic", "advanced", "admin")

# Slash commands
SLASH_COMMANDS = ["/help", "/clear", "/level", "/history", "/exit"]


class MASORePL:
    """REPL interactivo para MASO.

    Encapsula el ciclo interactivo de línea de comandos con soporte
    para slash commands, historial persistente, spinner de carga y
    renderizado de Markdown.

    Attributes:
        user_level: Nivel de usuario actual (basic, advanced, admin).
        history_file: Path al archivo de historial de comandos (FileHistory).
        session_history_file: Path al archivo JSON de historial de sesiones.
        console: Instancia de Rich Console para output formateado.
        history: Lista en memoria de interacciones {query, response, timestamp}.
        session: PromptSession de prompt_toolkit para input con completado.
    """

    def __init__(
        self,
        user_level: str = "basic",
        history_file: Optional[Path] = None,
        session_history_file: Optional[Path] = None,
    ) -> None:
        """Inicializa la REPL con configuración de historial y nivel de usuario.

        Args:
            user_level: Nivel de usuario (basic, advanced, admin).
            history_file: Path para FileHistory de prompt_toolkit.
                Por defecto usa ~/.maso_history.
            session_history_file: Path para historial JSON de sesiones.
                Por defecto usa ~/.maso_sessions.json.
        """
        self.user_level = user_level
        self.console = Console()

        # FileHistory path
        self.history_file = history_file or (Path.home() / ".maso_history")

        # Session history JSON path
        self.session_history_file = session_history_file or (
            Path.home() / ".maso_sessions.json"
        )

        # In-memory history list
        self.history: list[dict[str, str]] = []

        # Load persisted session history
        self._load_history()

        # Prompt Toolkit session
        completer = WordCompleter(SLASH_COMMANDS, ignore_case=True)
        self.session: PromptSession = PromptSession(
            history=FileHistory(str(self.history_file)),
            completer=completer,
            complete_while_typing=True,
        )

    def run(self) -> None:
        """Loop principal de la REPL.

        Muestra la pantalla de bienvenida y procesa input del usuario
        hasta que se ejecuta el comando /exit o se recibe KeyboardInterrupt.
        """
        self._show_welcome()

        while True:
            try:
                query = self.session.prompt("> ")
            except KeyboardInterrupt:
                self.console.print("\n¡Hasta luego!")
                self._save_history()
                break
            except EOFError:
                self.console.print("\n¡Hasta luego!")
                self._save_history()
                break

            if not query.strip():
                continue

            should_continue = self._handle_command(query.strip())
            if not should_continue:
                self._save_history()
                break

    def _handle_command(self, query: str) -> bool:
        """Procesa un comando del usuario.

        Determina si el input es un slash command o una consulta normal
        y lo despacha al handler correspondiente.

        Args:
            query: Texto ingresado por el usuario.

        Returns:
            True si el loop debe continuar, False si debe terminar.
        """
        if query.startswith("/"):
            return self._handle_slash_command(query)

        # Normal query: send to agent
        self._process_query(query)
        return True

    def _handle_slash_command(self, command: str) -> bool:
        """Procesa un slash command.

        Args:
            command: El slash command completo (ej: "/level admin").

        Returns:
            True si el loop debe continuar, False si debe terminar.
        """
        parts = command.split()
        cmd = parts[0].lower()

        if cmd == "/help":
            self._show_help()
            return True
        elif cmd == "/clear":
            self.console.clear()
            return True
        elif cmd == "/level":
            return self._handle_level(parts)
        elif cmd == "/history":
            self._show_history(parts)
            return True
        elif cmd == "/exit":
            self.console.print("¡Hasta luego!")
            return False
        else:
            self._show_invalid_command(cmd)
            return True

    def _show_welcome(self) -> None:
        """Muestra la pantalla de bienvenida con branding MASO."""
        welcome_text = (
            "Bienvenido al **MASO** - Multi-Agent Support Operator\n\n"
            f"Nivel actual: **{self.user_level}**\n\n"
            "Escribe tu consulta o usa los comandos disponibles.\n"
            "Escribe `/help` para ver la lista de comandos."
        )
        panel = Panel(
            welcome_text,
            title="[bold blue]MASO[/bold blue]",
            subtitle="Asistente de Soporte Técnico",
            border_style="blue",
        )
        self.console.print(panel)

    def _show_help(self) -> None:
        """Muestra la tabla de comandos disponibles."""
        table = Table(title="Comandos Disponibles")
        table.add_column("Comando", style="cyan", no_wrap=True)
        table.add_column("Descripción", style="white")

        table.add_row("/help", "Muestra esta ayuda")
        table.add_row("/clear", "Limpia la pantalla")
        table.add_row(
            "/level <basic|advanced|admin>",
            "Cambia el nivel de usuario",
        )
        table.add_row(
            "/history [n]",
            "Muestra las últimas n interacciones (default: 10)",
        )
        table.add_row("/exit", "Sale de la REPL")

        self.console.print(table)

    def _handle_level(self, parts: list[str]) -> bool:
        """Maneja el comando /level.

        Args:
            parts: Partes del comando splitteadas por espacio.

        Returns:
            True siempre (no termina la REPL).
        """
        if len(parts) < 2:
            self.console.print(
                Panel(
                    "Uso: /level <basic|advanced|admin>",
                    title="[yellow]Error[/yellow]",
                    border_style="yellow",
                )
            )
            return True

        new_level = parts[1].lower()
        if new_level not in VALID_LEVELS:
            self.console.print(
                Panel(
                    f"Nivel inválido: '{new_level}'. "
                    f"Opciones válidas: {', '.join(VALID_LEVELS)}",
                    title="[red]Error[/red]",
                    border_style="red",
                )
            )
            return True

        old_level = self.user_level
        self.user_level = new_level
        self.console.print(
            Panel(
                f"Nivel cambiado de **{old_level}** a **{new_level}**",
                title="[green]Nivel Actualizado[/green]",
                border_style="green",
            )
        )
        return True

    def _show_history(self, parts: list[str]) -> None:
        """Muestra el historial de interacciones en una tabla.

        Args:
            parts: Partes del comando. Si hay un segundo elemento, se usa
                como cantidad de entradas a mostrar.
        """
        n = 10
        if len(parts) >= 2:
            try:
                n = int(parts[1])
            except ValueError:
                self.console.print(
                    Panel(
                        f"Valor inválido: '{parts[1]}'. "
                        "Debe ser un número entero.",
                        title="[yellow]Error[/yellow]",
                        border_style="yellow",
                    )
                )
                return

        if not self.history:
            self.console.print(
                Panel(
                    "No hay interacciones en el historial.",
                    title="Historial",
                    border_style="blue",
                )
            )
            return

        entries = self.history[-n:]
        table = Table(title=f"Historial (últimas {len(entries)} entradas)")
        table.add_column("#", style="dim", width=4)
        table.add_column("Query", style="cyan", width=40)
        table.add_column("Timestamp", style="dim", width=20)

        for i, entry in enumerate(entries, start=1):
            query_preview = entry.get("query", "")[:50]
            if len(entry.get("query", "")) > 50:
                query_preview += "..."
            timestamp = entry.get("timestamp", "")
            table.add_row(str(i), query_preview, timestamp)

        self.console.print(table)

    def _process_query(self, query: str) -> None:
        """Procesa una consulta normal enviándola al agente.

        Envuelve la llamada a run_agent() con un spinner y renderiza
        la respuesta como Markdown.

        Args:
            query: La consulta del usuario.
        """
        # Import here to avoid circular imports
        from src.graph.maso_graph import run_agent

        try:
            with self.console.status("[bold blue]Procesando consulta...[/bold blue]"):
                response = run_agent(query, self.user_level)

            # Render response as Markdown
            self.console.print()
            self.console.print(Markdown(response))
            self.console.print()

            # Save to history
            self._add_to_history(query, response)

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            self.console.print(
                Panel(
                    f"Error al procesar la consulta:\n\n**{e}**\n\n"
                    "Intentá de nuevo o usá `/help` para ver comandos.",
                    title="[red]Error[/red]",
                    border_style="red",
                )
            )

    def _add_to_history(self, query: str, response: str) -> None:
        """Agrega una interacción al historial en memoria.

        Args:
            query: La consulta del usuario.
            response: La respuesta del agente.
        """
        entry = {
            "query": query,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.history.append(entry)

    def _save_history(self) -> None:
        """Persiste el historial de sesiones en JSON.

        Guarda las últimas MAX_HISTORY_ENTRIES entradas. Si el archivo
        ya existe, lo sobrescribe atómicamente usando write + rename.
        """
        try:
            # Cap history to MAX_HISTORY_ENTRIES
            capped_history = self.history[-MAX_HISTORY_ENTRIES:]

            # Write to temp file then rename for atomicity
            temp_path = self.session_history_file.with_suffix(".json.tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(capped_history, f, ensure_ascii=False, indent=2)

            temp_path.replace(self.session_history_file)
        except OSError as e:
            logger.error(f"Error saving history: {e}")

    def _load_history(self) -> None:
        """Carga el historial de sesiones desde JSON.

        Si el archivo no existe, inicia con historial vacío.
        Si el archivo está corrupto, loguea un warning y empieza vacío.
        """
        if not self.session_history_file.exists():
            return

        try:
            with open(
                self.session_history_file, "r", encoding="utf-8"
            ) as f:
                data = json.load(f)

            if isinstance(data, list):
                self.history = data
            else:
                logger.warning(
                    "History file has unexpected format, starting empty"
                )
                self.history = []

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                f"Error loading history file: {e}. Starting with empty history."
            )
            self.history = []

    def _show_invalid_command(self, cmd: str) -> None:
        """Muestra un error para comandos inválidos con sugerencias.

        Args:
            cmd: El comando inválido ingresado.
        """
        self.console.print(
            Panel(
                f"Comando desconocido: **{cmd}**\n\n"
                "Comandos válidos: `/help`, `/clear`, `/level`, "
                "`/history`, `/exit`\n\n"
                "Escribí `/help` para más información.",
                title="[yellow]Comando Inválido[/yellow]",
                border_style="yellow",
            )
        )


def run_simple_mode(user_level: str = "basic") -> None:
    """Modo REPL simple para entornos non-TTY.

    Usa input() y print() básicos sin Rich ni Prompt Toolkit.
    Compatible con pipes y redirección de entrada/salida.

    Args:
        user_level: Nivel de usuario (basic, advanced, admin).
    """
    from src.graph.maso_graph import run_agent

    print("=" * 50)
    print("MASO - Asistente de Soporte Técnico")
    print(f"Nivel: {user_level}")
    print("Escribe 'exit' para salir")
    print("=" * 50)

    while True:
        try:
            query = input("\n> ")

            if query.lower() in ("exit", "quit", "salir"):
                print("¡Hasta luego!")
                break

            if not query.strip():
                continue

            print("Procesando consulta...")
            response = run_agent(query, user_level)
            print(f"\n{response}\n")

        except KeyboardInterrupt:
            print("\n¡Hasta luego!")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"Error: {e}")
