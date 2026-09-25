from typing import Iterable, List, Literal, Optional

import numpy as np
from ase import Atoms

from scm.moliterate.core import BaseChemDataSet, BaseFilter, ChemDataEntry, PropertyInfo
from scm.moliterate.filters import AndFilter, ConditionsFilter, NotFilter, OrFilter, PipeFilter
from scm.moliterate.interfaces.in_memory import InMemoryMolData


class IndexFilter(BaseFilter):
    type: Literal["IndexFilter"] = "IndexFilter"
    indices: Optional[List[int]] = None

    def apply_filter(self, db: BaseChemDataSet) -> Optional[Iterable[int]]:
        if self.indices is None:
            return None
        return np.asarray(list(self.indices), dtype=np.int64)

    def __hash__(self) -> int:
        return super().__hash__()


def make_db(values: Iterable[float]) -> InMemoryMolData:
    db = InMemoryMolData.create(available_properties=[PropertyInfo(name="score", unit="")])
    for value in values:
        db.add_system(ChemDataEntry(system=Atoms(), properties={"score": value}))
    return db


def test_and_or_filters_basic():
    db = make_db([1, 2, 3, 4, 5, 6])
    ge_2 = ConditionsFilter(conditions="score>=2")
    le_4 = ConditionsFilter(conditions="score<=4")

    and_filter = AndFilter(filters={ge_2, le_4})
    or_filter = OrFilter(filters={ge_2, le_4})

    assert np.array_equal(np.asarray(and_filter.apply_filter(db)), np.array([1, 2, 3]))
    assert np.array_equal(np.asarray(or_filter.apply_filter(db)), np.array([0, 1, 2, 3, 4, 5]))


def test_not_filter_complement():
    db = make_db([1, 2, 3, 4, 5, 6])
    ge_4 = ConditionsFilter(conditions="score>=4")

    not_filter = NotFilter(filter=ge_4)

    assert np.array_equal(np.asarray(not_filter.apply_filter(db)), np.array([0, 1, 2]))


def test_pipe_filter_relative_indices():
    db = make_db([1, 2, 3, 4, 5, 6])
    first = IndexFilter(indices=[1, 2, 4, 5])
    second = IndexFilter(indices=[0, 2])

    pipe = PipeFilter.model_construct(filters=[first, second])

    assert np.array_equal(np.asarray(pipe.apply_filter(db)), np.array([1, 4]))


def test_bitwise_operator_overloads():
    f1 = ConditionsFilter(conditions="score>=2")
    f2 = ConditionsFilter(conditions="score<=4")

    and_filter = f1 & f2
    or_filter = f1 | f2
    not_filter = ~f1
    pipe_filter = f1 >> f2

    assert isinstance(and_filter, AndFilter)
    assert f1 in and_filter.filters and f2 in and_filter.filters
    assert isinstance(or_filter, OrFilter)
    assert f1 in or_filter.filters and f2 in or_filter.filters
    assert isinstance(not_filter, NotFilter)
    assert not_filter.filter is f1
    assert isinstance(pipe_filter, PipeFilter)
    assert pipe_filter.filters == [f1, f2]


def test_and_or_flatten_nested():
    f1 = ConditionsFilter(conditions="score>=2")
    f2 = ConditionsFilter(conditions="score<=4")

    nested_and = AndFilter(filters={f1, f2})
    top_and = AndFilter(filters={nested_and, f1})
    assert all(not isinstance(f, AndFilter) for f in top_and.filters)
    assert top_and.filters == {f1, f2}

    nested_or = OrFilter(filters={f1, f2})
    top_or = OrFilter(filters={nested_or, f1})
    assert all(not isinstance(f, OrFilter) for f in top_or.filters)
    assert top_or.filters == {f1, f2}


def test_and_or_ignore_none_and_handle_all_none():
    db = make_db([1, 2, 3])
    none_filter = IndexFilter(indices=None)
    some_filter = IndexFilter(indices=[0, 2])

    and_filter = AndFilter.model_construct(filters={none_filter, some_filter})
    or_filter = OrFilter.model_construct(filters={none_filter, some_filter})
    empty_and = AndFilter.model_construct(filters={none_filter})
    empty_or = OrFilter.model_construct(filters={none_filter})

    assert np.array_equal(np.asarray(and_filter.apply_filter(db)), np.array([0, 2]))
    assert np.array_equal(np.asarray(or_filter.apply_filter(db)), np.array([0, 2]))
    assert empty_and.apply_filter(db) is None
    assert empty_or.apply_filter(db) is None


def test_coerce_field_with_base_filter():
    from scm.moliterate.filters.coerce_base_filter_to_known import coerce_field_with_base_filter

    raw = {"type": "ConditionsFilter", "conditions": "score>=2"}
    coerced = coerce_field_with_base_filter(raw)
    assert isinstance(coerced, list)
    assert len(coerced) == 1
    assert isinstance(coerced[0], ConditionsFilter)
    assert coerced[0].conditions == "score>=2"

    existing = ConditionsFilter(conditions="score<=4")
    coerced_list = coerce_field_with_base_filter([raw, existing])
    assert isinstance(coerced_list, list)
    assert isinstance(coerced_list[0], ConditionsFilter)
    assert coerced_list[1] is existing
