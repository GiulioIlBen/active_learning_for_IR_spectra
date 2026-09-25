# Project Overview
`moliterate` provides typed molecular dataset interfaces, transforms, partition
criteria, and filters. Keep changes small and composable.

## Repository Structure
- `src/scm/moliterate/`: package source
- `src/scm/moliterate/cli/`: command-line entrypoints and CLI helpers
- `src/scm/moliterate/apps/`: marimo or other app-style modules
- `tests/`: pytest suite
- `tutorials/`: notebooks and tutorial data

## Commands
```bash
just test-base
just test
just ruff
```

## Guardrails
- Read `pyproject.toml` before changing dependencies or console scripts.
- Treat optional extras as optional at import time; prefer lazy imports.
- Ask before running AMS-heavy commands like `just test-amspy`.
- Never edit generated artifacts such as `.amspy/`, `htmlcov/`, `.coverage`,
  `coverage.json`, `pytest.log`, or `PLAMS/`.
- Keep diffs targeted; avoid broad refactors unless requested.

## Conventions
- Python >= 3.8, Ruff formatting, double quotes, 120-char line length.
- Use `snake_case` for functions/modules and `PascalCase` for classes.
- For Pydantic discriminated-union models, define `type` as `Literal["<ClassName>"] = "<ClassName>"`.
  Keep serialized YAML/JSON fixtures and downstream consumers aligned with the concrete class name.
- Add new package code under `src/scm/moliterate/` and tests under `tests/`.
