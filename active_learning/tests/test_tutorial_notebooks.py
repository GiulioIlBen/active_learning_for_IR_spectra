"""Keep every shipped notebook syntactically valid and wired to the tutorial safety helpers."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from scm.active_learning.tutorials import find_final_state, require_companion_state

NOTEBOOKS_DIRECTORY = Path(__file__).parents[1] / "tutorials" / "notebooks"


def notebook_sources(path: Path) -> list[str]:
    """Return code-cell source from one notebook for syntax checks that need no AMS installation or license."""
    notebook = json.loads(path.read_text())
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


@pytest.mark.parametrize("notebook_path", sorted(NOTEBOOKS_DIRECTORY.glob("*.ipynb")))
def test_notebook_code_cells_parse(notebook_path: Path) -> None:
    """Compile each distributable notebook cell so syntax errors cannot be committed unnoticed."""
    for index, source in enumerate(notebook_sources(notebook_path), start=1):
        ast.parse(source, filename=f"{notebook_path}:cell-{index}")


@pytest.mark.parametrize(
    "notebook_name",
    [
        "sal-2-single_molecule_results.ipynb",
        "sal-3-single_molecule_production_simulation.ipynb",
        "sal-4-continue_with_new_system.ipynb",
    ],
)
def test_state_dependent_notebooks_use_safe_state_resolution(notebook_name: str) -> None:
    """Require state-consuming notebooks to avoid an unhelpful IndexError when no prior run exists."""
    source = "\n".join(notebook_sources(NOTEBOOKS_DIRECTORY / notebook_name))
    assert "find_final_state" in source


def test_find_final_state_explains_missing_prerequisite(tmp_path: Path) -> None:
    """Explain how to recover when a continuation tutorial has no preceding active-learning run."""
    with pytest.raises(FileNotFoundError, match="sal-1-single_molecule_setup_run"):
        find_final_state(runs_directory=tmp_path)


def test_require_companion_state_requires_start_state(tmp_path: Path) -> None:
    """Prevent continuation notebooks from reaching model loading when the matching start state is unavailable."""
    final_state = tmp_path / "al_final_state.yaml"
    final_state.touch()
    with pytest.raises(FileNotFoundError, match="al_start_state.yaml"):
        require_companion_state(final_state)
