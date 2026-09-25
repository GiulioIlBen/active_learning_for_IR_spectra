from pathlib import Path

import pytest
from testbook import testbook

import scm.active_learning as al

al_path = Path(al.__path__[0])
DOC_FOLDER = al_path.parent.parent.parent / "tutorials"
assert DOC_FOLDER.exists()

pytest.importorskip("scm.base")


@testbook(DOC_FOLDER / "examples-molecules-journey.ipynb", execute=True, timeout=240)
def test_examples_molecules_journey(tb):
    pass


@testbook(DOC_FOLDER / "simple_active_learning/sal_single_molecule_setup_run.ipynb", execute=True, timeout=240)
def test_sal_single_molecule_setup_run(tb):
    pass


@testbook(DOC_FOLDER / "simple_active_learning/sal_single_molecule_results.ipynb", execute=True, timeout=240)
def test_sal_single_molecule_results(tb):
    pass


@testbook(
    DOC_FOLDER / "simple_active_learning/sal_single_molecule_production_simulation.ipynb", execute=True, timeout=240
)
def test_sal_single_molecule_production_simulation(tb):
    pass


@testbook(DOC_FOLDER / "examples-simple-active-learning.ipynb", execute=True, timeout=180)
def test_examples_simple_active_learning(tb):
    pass
