from typing import Iterable, List, Literal, Optional

import numpy as np
from ase import Atoms

from scm.moliterate.core import BaseChemDataSet, BaseFilter, ChemDataEntry, PropertyInfo
from scm.moliterate.filters import SimpleContextFilter
from scm.moliterate.interfaces.in_memory import InMemoryMolData


class IndexFilter(BaseFilter):
    type: Literal["IndexFilter"] = "IndexFilter"
    indices: Optional[List[int]] = None

    def apply_filter(self, db: BaseChemDataSet) -> Optional[Iterable[int]]:
        if self.indices is None:
            return None
        return np.asarray(self.indices, dtype=np.int64)

    def __hash__(self) -> int:
        return super().__hash__()


def make_db(values: Iterable[float]) -> InMemoryMolData:
    db = InMemoryMolData.create(available_properties=[PropertyInfo(name="score", unit="")])
    for value in values:
        db.add_system(ChemDataEntry(system=Atoms(), properties={"score": value}))
    return db


def test_simple_context_filter_returns_none_for_empty_target_dataset():
    context = make_db([10.0, 20.0])
    db = make_db([])
    flt = SimpleContextFilter.model_construct(context_dataset=context, selection_filter=IndexFilter(indices=[0]))

    assert flt.apply_filter(db) is None


def test_simple_context_filter_delegates_selection_when_context_is_empty():
    context = make_db([])
    db = make_db([1.0, 2.0, 3.0])
    flt = SimpleContextFilter.model_construct(context_dataset=context, selection_filter=IndexFilter(indices=[1, 2]))

    result = np.asarray(flt.apply_filter(db))

    assert np.array_equal(result, np.array([1, 2]))


def test_simple_context_filter_propagates_none_from_selection_filter():
    context = make_db([10.0])
    db = make_db([1.0, 2.0, 3.0])
    flt = SimpleContextFilter.model_construct(context_dataset=context, selection_filter=IndexFilter(indices=None))

    assert flt.apply_filter(db) is None


def test_simple_context_filter_maps_concat_indices_back_to_target_dataset():
    context = make_db([10.0, 20.0])
    db = make_db([1.0, 2.0, 3.0])
    flt = SimpleContextFilter.model_construct(context_dataset=context, selection_filter=IndexFilter(indices=[0, 2, 4]))

    result = np.asarray(flt.apply_filter(db))

    assert np.array_equal(result, np.array([0, 2]))


def test_simple_context_filter_returns_empty_when_selection_hits_only_context():
    context = make_db([10.0, 20.0])
    db = make_db([1.0, 2.0, 3.0])
    flt = SimpleContextFilter.model_construct(context_dataset=context, selection_filter=IndexFilter(indices=[0, 1]))

    result = np.asarray(flt.apply_filter(db))

    assert np.array_equal(result, np.array([], dtype=np.int64))


def test_simple_context_filter_call_returns_subset_of_target_dataset():
    context = make_db([10.0, 20.0])
    db = make_db([1.0, 2.0, 3.0])
    flt = SimpleContextFilter.model_construct(context_dataset=context, selection_filter=IndexFilter(indices=[3, 4]))

    filtered = flt(db)

    assert len(filtered) == 2
    assert [row.properties["score"] for row in filtered] == [2.0, 3.0]
