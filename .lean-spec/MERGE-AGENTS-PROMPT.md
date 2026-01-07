# AI Prompt: Consolidate AGENTS.md

## Task
Consolidate the existing AGENTS.md with LeanSpec instructions into a single, coherent document.

## Instructions
1. Read both documents below
2. Merge them intelligently:
   - Preserve ALL existing project-specific information (workflows, SOPs, architecture, conventions)
   - Integrate LeanSpec sections where they fit naturally
   - Remove redundancy and ensure coherent flow
   - Keep the tone and style consistent
3. Replace the existing AGENTS.md with the consolidated version

## Existing AGENTS.md
```markdown
# Repository Guidelines

## Project Structure

- `swift/`: main Python package (CLI entrypoints, LLM training/inference, Megatron-SWIFT, trainers/tuners, UI, utilities).
- `tests/`: `unittest` suites (grouped by area such as `tests/llm/`, `tests/tuners/`, `tests/hub/`).
- `examples/`: runnable training/inference/export scripts and configs.
- `docs/`: Sphinx documentation (`docs/source/`, `docs/source_en/`, assets in `docs/resources/`).
- `asset/`: images used by README/docs.
- `scripts/`: benchmarking and helper utilities.

## Build, Test, and Development Commands

```bash
pip install -e .
pip install -r requirements/tests.txt
pre-commit run --all-files
python -m unittest discover -s tests -p 'test_*.py'
python tests/llm/test_run.py
make whl
make docs
```

- `pip install -e .`: editable install; provides `swift` and `megatron` CLIs (`swift --help`).
- `requirements/tests.txt` + `pre-commit`: runs formatting/lint checks (flake8/isort/yapf and basic hygiene hooks).
- `unittest discover`: runs fast unit tests; prefer adding/adjusting tests here for most changes.
- `python tests/llm/test_run.py`: end-to-end smoke test (may require CUDA and model/dataset downloads).
- `make whl`: builds sdist/wheel via `setup.py`.
- `make docs`: builds HTML docs (installs `requirements/docs.txt` and runs `docs/Makefile`).

## Coding Style & Naming Conventions

- Python: 4-space indentation, max line length 120.
- Names: `snake_case` for functions/vars, `PascalCase` for classes, `ALL_CAPS` for constants.
- Prefer running `pre-commit run --all-files` before pushing; it applies `isort` and `yapf` consistently.

## Testing Guidelines

- Use `unittest` (`test_*.py`, `Test*` classes, `test_*` methods).
- New features should include a unit test and, when applicable, a lightweight smoke test.
- Avoid making the default test path download large models; gate heavy tests behind explicit execution or environment checks, following existing patterns in `tests/llm/`.

## Configuration & Security

- Never commit credentials (for example, values passed to `--hub_token`); prefer environment variables or local secret storage.
- Model/dataset downloads can be large; use caches and set `HF_ENDPOINT` to a mirror when needed (see `tests/llm/test_run.py`).

## Commit & Pull Request Guidelines

- Commit messages commonly use a bracketed scope: `[feat] ...`, `[bugfix] ...`, `[docs] ...`, `[megatron] ...` (multiple scopes allowed: `[megatron, misc] ...`).
- PRs should include: a clear description, linked issues/PRs, reproduction steps (for bugs), and updates to docs/examples when behavior changes. Include screenshots for UI/docs changes when helpful.

```

## LeanSpec Instructions to Integrate
```markdown
# AI Agent Instructions

## Project: ms-swift

## 🚨 CRITICAL: Before ANY Task

**STOP and check these first:**

1. **Discover context** → Use `board` tool to see project state
2. **Search for related work** → Use `search` tool before creating new specs
3. **Never create files manually** → Always use `create` tool for new specs

> **Why?** Skipping discovery creates duplicate work. Manual file creation breaks LeanSpec tooling.

## 🔧 Managing Specs

### MCP Tools (Preferred) with CLI Fallback

| Action | MCP Tool | CLI Fallback |
|--------|----------|--------------|
| Project status | `board` | `lean-spec board` |
| List specs | `list` | `lean-spec list` |
| Search specs | `search` | `lean-spec search "query"` |
| View spec | `view` | `lean-spec view <spec>` |
| Create spec | `create` | `lean-spec create <name>` |
| Update spec | `update` | `lean-spec update <spec> --status <status>` |
| Link specs | `link` | `lean-spec link <spec> --depends-on <other>` |
| Unlink specs | `unlink` | `lean-spec unlink <spec> --depends-on <other>` |
| Dependencies | `deps` | `lean-spec deps <spec>` |
| Token count | `tokens` | `lean-spec tokens <spec>` |

## ⚠️ Core Rules

| Rule | Details |
|------|---------|
| **NEVER edit frontmatter manually** | Use `update`, `link`, `unlink` for: `status`, `priority`, `tags`, `assignee`, `transitions`, timestamps, `depends_on` |
| **ALWAYS link spec references** | Content mentions another spec → `lean-spec link <spec> --depends-on <other>` |
| **Track status transitions** | `planned` → `in-progress` (before coding) → `complete` (after done) |
| **No nested code blocks** | Use indentation instead |

### 🚫 Common Mistakes

| ❌ Don't | ✅ Do Instead |
|----------|---------------|
| Create spec files manually | Use `create` tool |
| Skip discovery | Run `board` and `search` first |
| Leave status as "planned" | Update to `in-progress` before coding |
| Edit frontmatter manually | Use `update` tool |

## 📋 SDD Workflow

```
BEFORE: board → search → check existing specs
DURING: update status to in-progress → code → document decisions → link dependencies
AFTER:  update status to complete → document learnings
```

**Status tracks implementation, NOT spec writing.**

## Spec Dependencies

Use `depends_on` to express blocking relationships between specs:
- **`depends_on`** = True blocker, work order matters, directional (A depends on B)

Link dependencies when one spec builds on another:
```bash
lean-spec link <spec> --depends-on <other-spec>
```

## When to Use Specs

| ✅ Write spec | ❌ Skip spec |
|---------------|--------------|
| Multi-part features | Bug fixes |
| Breaking changes | Trivial changes |
| Design decisions | Self-explanatory refactors |

## Token Thresholds

| Tokens | Status |
|--------|--------|
| <2,000 | ✅ Optimal |
| 2,000-3,500 | ✅ Good |
| 3,500-5,000 | ⚠️ Consider splitting |
| >5,000 | 🔴 Must split |

## First Principles (Priority Order)

1. **Context Economy** - <2,000 tokens optimal, >3,500 needs splitting
2. **Signal-to-Noise** - Every word must inform a decision
3. **Intent Over Implementation** - Capture why, let how emerge
4. **Bridge the Gap** - Both human and AI must understand
5. **Progressive Disclosure** - Add complexity only when pain is felt

---

**Remember:** LeanSpec tracks what you're building. Keep specs in sync with your work!
```

## Output
Create a single consolidated AGENTS.md that:
- Keeps all existing project context and workflows
- Adds LeanSpec commands and principles where appropriate
- Maintains clear structure and readability
- Removes any duplicate or conflicting guidance
