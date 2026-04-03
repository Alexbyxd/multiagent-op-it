# Skill Registry - Nuevo

**Generated**: 2026-04-01
**Mode**: SDD engram

## User Skills (OpenCode)

| Skill | Location | Trigger |
|-------|----------|---------|
| sdd-init | ~/.config/opencode/skills/sdd-init/SKILL.md | sdd init |
| sdd-explore | ~/.config/opencode/skills/sdd-explore/SKILL.md | /sdd-explore |
| sdd-propose | ~/.config/opencode/skills/sdd-propose/SKILL.md | /sdd-new |
| sdd-spec | ~/.config/opencode/skills/sdd-spec/SKILL.md | /sdd-spec |
| sdd-design | ~/.config/opencode/skills/sdd-design/SKILL.md | /sdd-design |
| sdd-tasks | ~/.config/opencode/skills/sdd-tasks/SKILL.md | /sdd-tasks |
| sdd-apply | ~/.config/opencode/skills/sdd-apply/SKILL.md | /sdd-apply |
| sdd-verify | ~/.config/opencode/skills/sdd-verify/SKILL.md | /sdd-verify |
| sdd-archive | ~/.config/opencode/skills/sdd-archive/SKILL.md | /sdd-archive |
| sdd-onboard | ~/.config/opencode/skills/sdd-onboard/SKILL.md | /sdd-onboard |
| brainstorming | ~/.cache/opencode/node_modules/superpowers/skills/brainstorming/SKILL.md | creative work |
| gemini-api-dev | ~/.agents/skills/gemini-api-dev/SKILL.md | Gemini API |
| langchain-rag | ~/.agents/skills/langchain-rag/SKILL.md | RAG system |
| langgraph-fundamentals | ~/.agents/skills/langgraph-fundamentals/SKILL.md | LangGraph code |
| python-testing-patterns | ~/.agents/skills/python-testing-patterns/SKILL.md | Python tests |
| issue-creation | ~/.config/opencode/skills/issue-creation/SKILL.md | create issue |
| branch-pr | ~/.config/opencode/skills/branch-pr/SKILL.md | create PR |
| judgment-day | ~/.config/opencode/skills/judgment-day/SKILL.md | review |

## Project Conventions

| File | Description |
|------|-------------|
| AGENTS.md | Project conventions, stack, architecture patterns |
| PRD.md | Product requirements (MASO design) |

## Compact Rules (for sub-agents)

### Python Conventions
- **Imports**: stdlib → third-party → local, absolute imports
- **Typing**: Full typing required, no `Any`, use `TypedDict` for LangGraph state
- **Docstrings**: Google-style (description, args, returns, raises)
- **Naming**: snake_case (vars/funcs), PascalCase (classes), UPPER_SNAKE_CASE (constants)
- **Booleans**: prefix `is_`, `has_`, `can_`

### Quality Tools
- **Linting**: `ruff check src/`
- **Format**: `ruff format src/`
- **Typecheck**: `mypy src/`

### Testing
- **Command**: `pytest`
- **Coverage**: `pytest --cov=src --cov-report=html`
