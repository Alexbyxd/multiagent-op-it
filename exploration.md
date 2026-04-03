# Exploration: CLI/UX Improvement for MASO

## Current State

`src/main.py` (81 lines) has a bare-bones chat loop:
- `input()` for user input — no history, no auto-completion, no key bindings
- `print()` for output — no formatting, no markdown rendering, no colors
- No loading indicator while `run_agent()` processes (LLM calls can take 15-30s)
- No welcome screen beyond a `=` border with plain text
- Only recognizes `exit/quit/salir` — no slash commands
- The `--level` arg exists but can't be changed at runtime

The agent pipeline (`run_agent` → LangGraph → LLM) involves network calls to OpenRouter with timeouts of 15-30 seconds. Users currently stare at a blank terminal with zero feedback.

## Affected Areas

- `src/main.py` — Primary target. The `chat()` function will be rewritten with enhanced UX
- `requirements.txt` — New dependencies to add
- `src/graph/maso_graph.py` — May need a streaming variant of `run_agent()` for real-time progress (optional, future)

## Library Analysis

### 1. Rich (v14.3.3, Feb 2026)

**Pip:** `rich` | **Python 3.14:** ✅ Explicitly listed in classifiers

**What it does for us:**
- **Markdown rendering** — `rich.markdown.Markdown` renders bold, lists, code blocks, headers to terminal
- **Spinners/Status** — `console.status()` shows animated spinner with message while processing
- **Colors & styling** — `console.print()` with `[bold cyan]` style markup for welcome screen, prompts, errors
- **Tables** — Could display `/history` as a formatted table
- **Tracebacks** — Rich tracebacks for debugging
- **Syntax highlighting** — Code blocks in responses get pygments highlighting automatically
- **Progress bars** — Not directly useful for our use case (can't measure LLM progress)

**Pros:**
- 55k+ GitHub stars, extremely mature, actively maintained (released Feb 2026)
- Drop-in `rich.print()` replaces builtin `print` — minimal migration effort
- Markdown rendering is built-in and works out of the box
- Status/spinner works alongside normal console output (non-blocking)
- MIT license
- Already used by many popular CLI tools (Typer, etc.)

**Cons:**
- Does NOT provide input handling — no history, no auto-completion, no key bindings
- Would still need `input()` or another library for the prompt itself

### 2. Prompt Toolkit (v3.0.52, Aug 2025)

**Pip:** `prompt-toolkit` | **Python 3.14:** ⚠️ Not explicitly listed (stops at 3.13), but pure Python, no C extensions — should work. Last release was Aug 2025, before Python 3.14 final.

**What it does for us:**
- **Command history** — `FileHistory` persists history across sessions (up/down arrows)
- **Custom key bindings** — Bind `/help`, Ctrl+C, Ctrl+L, etc.
- **prompt_session()** — Drop-in replacement for `input()` with full editing support
- **Auto-completion** — Could suggest `/help`, `/clear`, `/history`, `/level` as user types `/`
- **Syntax highlighting** — Pygments integration for input highlighting
- **Multi-line input** — Support for longer queries

**Pros:**
- Pure Python — no compilation issues with new Python versions
- Used by ptpython, IPython, xonsh — battle-tested
- Emacs and Vi key bindings out of the box
- Lightweight (only dependency: Pygments + wcwidth)
- BSD license
- `prompt()` function is a one-line replacement for `input()`

**Cons:**
- No output formatting — plain text only for printing responses
- No markdown rendering
- No spinners or loading indicators
- Python 3.14 not yet in classifiers (risk, but low — pure Python)

### 3. Textual (v8.2.2, Apr 2026)

**Pip:** `textual` | **Python 3.14:** ✅ Explicitly listed in classifiers

**What it does for us:**
- Full TUI framework — could build a complete chat UI with panels, scrollable history, input bar
- Async under the hood — integrates well with async LLM calls
- Command palette (Ctrl+P) — built-in fuzzy search for commands
- Rich widgets — buttons, text areas, data tables, trees
- Can run in browser via `textual serve`

**Pros:**
- Most visually impressive option
- Built-in testing framework
- Can serve as web app
- Actively maintained (released today, Apr 2026)
- Same authors as Rich — good ecosystem

**Cons:**
- **OVERKILL** for our use case — we need a chat loop, not a full application framework
- Requires async rewrite of the entire `chat()` function and potentially the graph
- Steep learning curve — CSS-like styling (TCSS), widget composition, event handlers
- Takes over the entire terminal — can't easily mix with standard logging output
- Heavy dependency tree (depends on Rich + more)
- Would require restructuring `run_agent()` to work with Textual's async event loop
- Changes the paradigm from "simple CLI tool" to "full TUI application"

### 4. Alternative: Click + Rich

**Pip:** `click` + `rich`

Click is already a well-known CLI framework. Could use it for better argument parsing and command structure. However, Click is for command *parsing*, not interactive prompts. Not directly useful for our chat loop use case.

### 5. Alternative: Typer + Rich

**Pip:** `typer` + `rich`

Typer is built ON TOP of Rich and Click. Great for CLI commands but, like Click, not designed for interactive chat loops. Would need prompt-toolkit anyway for the input side.

## Comparison Matrix

| Feature | Rich | Prompt Toolkit | Textual |
|---------|------|---------------|---------|
| Markdown rendering | ✅ Built-in | ❌ | ✅ (Markdown widget) |
| Spinners/loading | ✅ `console.status()` | ❌ | ✅ (Loading widget) |
| Command history | ❌ | ✅ `FileHistory` | ✅ (custom) |
| Key bindings | ❌ | ✅ Full support | ✅ (event system) |
| Auto-completion | ❌ | ✅ | ✅ (command palette) |
| Colored output | ✅ | ❌ | ✅ |
| Syntax highlighting | ✅ (code blocks) | ✅ (input) | ✅ |
| Tables | ✅ | ❌ | ✅ |
| Effort to integrate | Low | Low | High |
| Python 3.14 support | ✅ | ⚠️ (likely) | ✅ |
| Async required | ❌ | ❌ | ✅ |

## Recommendation

### Use Rich + Prompt Toolkit together

This is the **optimal combination** for our use case:

- **Rich** handles all OUTPUT formatting: welcome screen, markdown responses, spinners, colored errors, tables for history
- **Prompt Toolkit** handles all INPUT: history, key bindings, auto-completion for slash commands

Both are lightweight, well-maintained, and solve complementary problems. Neither overlaps with the other.

### Why NOT Textual

Textual is a fantastic framework, but it's like using a sledgehammer to hang a picture. Our use case is a simple chat loop — Textual would require:
1. Rewriting the entire `chat()` as an async App
2. Learning TCSS for styling
3. Restructuring how `run_agent()` is called (async vs sync)
4. Taking over the full terminal (breaks logging visibility)

Textual makes sense if we want a full-screen dashboard with multiple panels, but for a chat interface, Rich + Prompt Toolkit delivers 90% of the UX improvement at 20% of the effort.

### Specific Features to Implement

**Phase 1 (Essential):**
1. **Welcome screen** (Rich) — Styled panel with MASO branding, current level, status info
2. **Loading spinner** (Rich) — `console.status("Procesando consulta...")` wraps `run_agent()` call
3. **Markdown rendering** (Rich) — Render LLM responses with `Markdown(response)` for bold, lists, code blocks
4. **Command history** (Prompt Toolkit) — `FileHistory` persisted to `~/.maso_history`
5. **Slash commands** — `/help`, `/clear`, `/level <basic|advanced|admin>`, `/history`, `/exit`

**Phase 2 (Nice to have):**
6. **Auto-completion** (Prompt Toolkit) — Tab-complete slash commands
7. **Colored prompts** (Rich) — Different colors for user input vs system output
8. **Error formatting** (Rich) — Styled error panels with traceback
9. **History table** (Rich) — `/history` shows recent queries in a formatted table
10. **Timestamps** — Log each interaction with timestamp

### Dependencies to Add

```
rich>=14.0.0
prompt-toolkit>=3.0.50
```

### Risks

1. **Prompt Toolkit + Python 3.14** — Not explicitly in classifiers yet, but it's pure Python with no C extensions. Very low risk. Can verify with a quick `pip install` test.
2. **Rich + terminal compatibility** — Works on Linux/macOS/Windows. Windows Terminal needed for true color/emoji. Fallback to 16 colors on legacy terminals. Rich handles this automatically.
3. **History file permissions** — Need to handle case where `~/.maso_history` can't be written (rare but possible).
4. **Spinner + async** — Rich's `console.status()` is synchronous. If we later move to async/streaming LLM calls, we'd need `rich.live.Live` instead.

### Ready for Proposal

**Yes.** The analysis is complete with clear library choices, feature breakdown, and risk assessment. The recommendation (Rich + Prompt Toolkit) is well-justified and directly addresses all requirements in the exploration prompt.
