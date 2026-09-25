"""Small, dependency-light helpers that keep interactive tutorials deterministic and self-explanatory."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Iterable

STATE_SUFFIXES = frozenset({".json", ".yaml", ".yml"})


def require_modules(modules: Iterable[str], *, install_hint: str) -> None:
    """Raise an actionable error when optional tutorial dependencies are unavailable to the current kernel."""
    missing = sorted(module for module in modules if find_spec(module) is None)
    if missing:
        names = ", ".join(missing)
        raise ModuleNotFoundError(f"This tutorial requires {names}. {install_hint}")


def find_final_state(
    state_path: Path | str | None = None,
    *,
    runs_directory: Path | str = "ALruns",
) -> Path:
    """Resolve a requested final-loop state or one available run state with a clear recovery instruction."""
    if state_path is not None:
        candidate = Path(state_path).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"The requested active-learning state does not exist: {candidate}")
        if candidate.suffix not in STATE_SUFFIXES:
            raise ValueError(f"Expected a JSON or YAML state file, got: {candidate}")
        return candidate.resolve()

    runs_path = Path(runs_directory).expanduser()
    candidates = sorted(
        path for path in runs_path.glob("*/al_final_state.*") if path.is_file() and path.suffix in STATE_SUFFIXES
    )
    if not candidates:
        raise FileNotFoundError(
            f"No completed active-learning state was found below {runs_path.resolve()}. "
            "Run sal-1-single_molecule_setup_run.ipynb first, or set state_path to an existing al_final_state file."
        )
    return candidates[-1].resolve()


def require_companion_state(final_state_path: Path | str, *, stem: str = "al_start_state") -> Path:
    """Find the state saved beside a final state so continuation tutorials fail before loading incomplete input."""
    final_path = Path(final_state_path)
    companion = final_path.with_name(f"{stem}{final_path.suffix}")
    if not companion.is_file():
        raise FileNotFoundError(
            f"The continuation tutorial also needs {companion.name} beside {final_path.name}. "
            "Choose a run created by the setup tutorial."
        )
    return companion.resolve()
