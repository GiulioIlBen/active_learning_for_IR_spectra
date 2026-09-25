# Workspace AGENTS

This workspace contains multiple projects. Use the project-level `AGENTS.md`
closest to the files being changed.

- For work in `active_learning/`, you must read first and follow `active_learning/AGENTS.md`.
- For work in `moliterate/`, you must read first and follow `moliterate/AGENTS.md`.
- If a task spans both projects, follow both files and keep changes scoped per
  project.

For top-level workspace files (for example prompts/docs in the repo root),
prefer minimal, non-invasive edits unless the task explicitly asks for broader
changes.

## Python Style Guide

- Prefer introducing new `BaseModel` classes and class composition over growing a
  single large class with many unrelated responsibilities.
- When validation, configuration, or domain concepts form a coherent subgroup,
  extract them into their own `BaseModel` and compose them from the parent
  class.
- When introducing a new `BaseModel`, prefer putting behavior methods that
  logically belong to that model on the new model itself instead of leaving the
  parent/orchestrator to own them.
- The more a class is composed from focused helper models, the fewer methods it
  should need; keep orchestration classes thin when their state already lives in
  structured submodels.
- If a class mostly stores simple JSON-like values (`str`, `int`, `float`,
  `dict`, `list`, ...) rather than rich composed objects, a larger method set is
  acceptable there.
