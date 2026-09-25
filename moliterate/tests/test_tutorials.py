import os
from pathlib import Path

import pytest

os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")

from testbook import testbook

import scm.moliterate as moliterate

moliterate_path = Path(moliterate.__path__[0])
DOC_FOLDER = moliterate_path.parent.parent.parent / "tutorials"
assert DOC_FOLDER.exists()

pytest.importorskip("scm.params")


@testbook(DOC_FOLDER / "quick-overview.ipynb", execute=True)
def test_quick_overview(tb):
    pass


@testbook(DOC_FOLDER / "core-moliterate-concepts.ipynb", execute=True)
def test_core_moliterate_concepts(tb):
    pass


@testbook(DOC_FOLDER / "advanced-features-filters.ipynb", execute=True)
def test_advanced_features_filters(tb):
    pass


@testbook(DOC_FOLDER / "advanced-create-an-interface.ipynb", execute=True)
def test_create_an_interface(tb):
    pass
