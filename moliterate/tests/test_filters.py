import numpy as np
import pytest
from ase import Atoms

from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.core.filters import BaseFilterError, BaseFilterWarning
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.filters import (
    AndFilter,
    ConditionsFilter,
    FarthestPointFilter,
    LinearSteppedFilter,
    RandomFilter,
    TopNFilter,
)
from scm.moliterate.interfaces.in_memory import InMemoryMolData
from scm.moliterate.partition_criteria import PropertyCriterion


def make_db(values):
    db = InMemoryMolData.create(available_properties=[PropertyInfo(name="score", unit="")])
    for value in values:
        db.add_system(ChemDataEntry(system=Atoms(), properties={"score": value}))
    return db


def test_maxppercentilefilter_selects_top_fraction():
    values = [1.0, 3.0, 2.0, 5.0, 4.0]
    db = make_db(values)
    flt = TopNFilter(partition_criterion=PropertyCriterion(property_key="score"), num_samples=0.4)

    result = np.array(list(flt.apply_filter(db)))

    assert result.tolist() == [4, 3]
    assert [values[i] for i in result] == [4.0, 5.0]


def test_maxppercentilefilter_returns_all_when_request_exceeds_dataset():
    values = [0.0, 1.0, 2.0]
    db = make_db(values)
    flt = TopNFilter(partition_criterion=PropertyCriterion(property_key="score"), num_samples=10)

    result = np.array(list(flt.apply_filter(db)))

    assert np.array_equal(result, np.arange(len(values)))


def test_farthestpointfilter_sampling_is_seeded():
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    db = make_db(values)
    flt = FarthestPointFilter(partition_criterion=PropertyCriterion(property_key="score"), num_samples=3, seed=0)

    result = np.array(list(flt.apply_filter(db)))

    assert result.tolist() == [4, 0, 2]
    assert len(set(result.tolist())) == 3


def test_farthestpointfilter_warns_when_not_enough_data():
    values = [0.0, 1.0]
    db = make_db(values)
    flt = FarthestPointFilter(partition_criterion=PropertyCriterion(property_key="score"), num_samples=5)

    with pytest.warns(BaseFilterWarning):
        result = flt.apply_filter(db)

    assert result is None


def test_randomfilter_sampling_is_seeded():
    values = [0, 1, 2, 3, 4]
    db = make_db(values)
    flt = RandomFilter(num_samples=3, seed=42)

    result = np.array(list(flt.apply_filter(db)))

    assert result.tolist() == [4, 0, 3]
    assert len(set(result.tolist())) == 3


def test_randomfilter_warns_when_not_enough_data():
    values = [0.0, 1.0]
    db = make_db(values)
    flt = RandomFilter(num_samples=5)

    with pytest.warns(BaseFilterWarning):
        result = flt.apply_filter(db)

    assert result is None


def test_linearsteppedfilter_positive_step_starts_from_beginning():
    values = list(range(10))
    db = make_db(values)
    flt = LinearSteppedFilter(step=2, num_samples=4)

    result = np.array(list(flt.apply_filter(db)))

    assert result.tolist() == [0, 2, 4, 6]


def test_linearsteppedfilter_negative_step_starts_from_end():
    values = list(range(10))
    db = make_db(values)
    flt = LinearSteppedFilter(step=-2, num_samples=4)

    result = np.array(list(flt.apply_filter(db)))

    assert result.tolist() == [9, 7, 5, 3]


def test_linearsteppedfilter_warns_when_requested_samples_not_reachable():
    values = list(range(10))
    db = make_db(values)
    flt = LinearSteppedFilter(step=4, num_samples=4)

    with pytest.warns(BaseFilterWarning):
        result = np.array(list(flt.apply_filter(db)))

    assert result.tolist() == [0, 4, 8]


def test_linearsteppedfilter_raises_for_zero_step():
    values = list(range(10))
    db = make_db(values)
    flt = LinearSteppedFilter(step=0, num_samples=3)

    with pytest.raises(BaseFilterError):
        flt.apply_filter(db)


def test_multiandfilter_applies_all_filters():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    db = make_db(values)
    percentile_filter = TopNFilter(partition_criterion=PropertyCriterion(property_key="score"), num_samples=0.4)
    random_filter = RandomFilter(num_samples=3, seed=42)
    combined_filter = AndFilter(filters=[percentile_filter, random_filter])

    result = np.array(list(combined_filter.apply_filter(db)))

    assert result.tolist() == [3, 4]


def test_conditionsfilter_applies_all_conditions():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    db = make_db(values)
    flt = ConditionsFilter(conditions=["score>=3", "score<5"])

    result = np.array(list(flt.apply_filter(db)))

    assert result.tolist() == [2, 3]


def test_conditionsfilter_returns_empty_subset_when_no_rows_match():
    db = make_db([1.0, 2.0, 3.0]).subset(indices=[0, 2])

    result = ConditionsFilter(conditions="score>10")(db)

    assert len(result) == 0


def test_randomfilter_iterates_over_subset_rows():
    values = [0, 1, 2, 3, 4]
    db = make_db(values)
    flt = RandomFilter(num_samples=3, seed=42)

    rows = list(flt.__iter__(db))

    assert [row.idx_origin for row in rows] == [0, 3, 4]
    assert [row.properties["score"] for row in rows] == [0, 3, 4]


def test_conditionsfilter_eval_condition_variants():
    row = ChemDataEntry(
        system=Atoms(),
        properties={
            "score": 3,
            "flag": True,
            "label": "foo",
            "strnum": "5",
            "arr": np.array([2.0]),
            "arr2": np.array([1.0, 2.0]),
        },
    )
    flt = ConditionsFilter(conditions=[])

    assert flt.eval_condition("score>=3", row)
    assert flt.eval_condition("score<4", row)
    assert flt.eval_condition("score!=4", row)
    assert flt.eval_condition("score=3.0", row)
    assert flt.eval_condition("label=foo", row)
    assert flt.eval_condition("label!=bar", row)
    assert flt.eval_condition("strnum=5", row)
    assert flt.eval_condition("flag=true", row)
    assert flt.eval_condition("arr>1.5", row)

    assert not flt.eval_condition("flag=false", row)
    assert not flt.eval_condition("score=bad", row)
    assert not flt.eval_condition("arr2=1", row)
