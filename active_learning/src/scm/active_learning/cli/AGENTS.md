# CLI Overview
This folder contains the `active_learning` command-line apps.

## Structure
- Put every new CLI app in its own file under `src/scm/active_learning/cli/`.
- Keep each CLI module small and focused on argument parsing and dispatch.
- Put reusable implementation in the package proper when logic starts growing beyond a thin CLI wrapper.

## Command Layout
- `src/scm/active_learning/__main__.py` is the top-level `al` entrypoint and dispatches subcommands into modules in this folder.
- Subcommands should delegate into modules from this folder.
- Prefer command names that map directly to module names (`plot` -> `plot.py`).

## Editing Rules
- Preserve the lightweight style of the current CLI code.
- Prefer `pydantic_settings` for short declarative CLIs.
- Default empty-argument behavior to help output for user-facing commands.
- Add or update focused CLI tests in `tests/test_cli.py` when changing command behavior.
