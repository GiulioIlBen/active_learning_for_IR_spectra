"""
Check if the data in memory is kept the same, for each subset
"""

import pytest

from scm.moliterate.interfaces.in_memory import InMemoryMolData
from scm.moliterate.interfaces.params_data import ParAMSData


def test_copy_in_memory(db_in_memory: InMemoryMolData):
    sliced_im = db_in_memory[[0, 1]]
    assert isinstance(sliced_im, type(db_in_memory))
    assert len(db_in_memory._data) == len(sliced_im._data)
    assert all([id(x) == id(y) for x, y in zip(db_in_memory._data, sliced_im._data)])


def test_copy_params(db):
    pytest.importorskip("scm.params")
    if isinstance(db, ParAMSData):
        sliced_im = db[[0, 1]]
        assert isinstance(sliced_im, type(db))
        assert len(db._keys) == len(sliced_im._keys)
        assert id(db._results_importer) == id(sliced_im._results_importer)
