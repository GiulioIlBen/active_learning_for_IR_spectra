import numpy as np
import pytest
from ase import Atoms

from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.core.partition_criteria import BasePartitionCriteria
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.interfaces.in_memory import InMemoryMolData
from scm.moliterate.partition_criteria.criteria_1d import PropertyCriterion
from scm.moliterate.partition_criteria.criteria_1d_splitting import (
    BinaryRandomSplitCriterion,
    SequentialSplitCriterion,
    StratifiedSplitCriterion,
    TrainValidationTestSplitCriterion,
)


def _make_db(length: int, property_name: str = "", values=None, put_metadata=False):
    props = [PropertyInfo(name=property_name, unit="")] if property_name and not put_metadata else []
    db = InMemoryMolData.create(available_properties=props)
    for i in range(length):
        props_dict = {}
        if property_name:
            if values is None:
                raise ValueError("values must be provided when property_name is set")
            props_dict[property_name] = values[i]
        if put_metadata:
            db.add_system(ChemDataEntry(system=Atoms(), metadata=props_dict))
        else:
            db.add_system(ChemDataEntry(system=Atoms(), properties=props_dict))

    return db


def test_sequential_split_criterion_matches_expected_pattern():
    db = _make_db(6)
    criterion = SequentialSplitCriterion(labelA="train", labelB="test", labelB_size=0.2)

    result = criterion(db)

    assert np.array_equal(result, np.array(["test", "train", "train", "train", "train", "test"]))


def test_binary_random_split_is_reproducible_with_seed():
    db = _make_db(5)
    criterion = BinaryRandomSplitCriterion(labelA="A", labelB="B", labelB_size=0.4, seed=0)

    first = criterion(db)
    second = criterion(db)

    assert np.array_equal(first, second)
    assert np.count_nonzero(first == "B") == 2


def test_train_validation_test_split_counts_are_correct():
    db = _make_db(20)
    criterion = TrainValidationTestSplitCriterion(val_size=0.2, test_size=0.3, random_state=1)

    labels = criterion(db)

    assert np.count_nonzero(labels == "validation_set") == 4
    assert np.count_nonzero(labels == "test_set") == 6
    assert np.count_nonzero(labels == "train_set") == 10


def test_stratified_split_respects_each_category():
    categories = ["a", "a", "b", "b", "b", "c", "c", "c", "c", "c"]
    db = _make_db(len(categories), property_name="category", values=categories, put_metadata=True)
    criterion = StratifiedSplitCriterion(
        stratify_criterion=PropertyCriterion(property_key="category"), labelB_size_per_cat=0.5, seed=0
    )

    labels = criterion(db)
    categories_arr = np.array(categories)

    assert np.count_nonzero(labels[categories_arr == "a"] == "test") == 1
    assert np.count_nonzero(labels[categories_arr == "b"] == "test") == 1
    assert np.count_nonzero(labels[categories_arr == "c"] == "test") == 2
    assert np.count_nonzero(labels[categories_arr == "c"] == "test") == 2


@pytest.mark.parametrize("labelB_size", [0.0, 1.0, -0.1])
def test_sequential_split_criterion_rejects_invalid_fraction(labelB_size):
    with pytest.raises(ValueError, match="Fraction must be greater than 0 and less than 1."):
        SequentialSplitCriterion.split_2_data_sequentially(3, labelB_size=labelB_size)


def test_sequential_split_criterion_single_item_returns_labelA():
    db = _make_db(1)
    criterion = SequentialSplitCriterion(labelA="train", labelB="test", labelB_size=0.5)

    labels = criterion(db)

    assert np.array_equal(labels, np.array(["train"]))


def test_binary_random_split_criterion_uses_weighted_property():
    weights = [1.0, 0.0, 0.0, 0.0]
    db = _make_db(len(weights), property_name="weight", values=weights)
    criterion = BinaryRandomSplitCriterion(
        labelA="A",
        labelB="B",
        labelB_size=1,
        weights_favors_B=PropertyCriterion(property_key="weight"),
    )

    labels = criterion(db)
    assert labels[0] == "B"
    assert np.count_nonzero(labels == "B") == 1


class _BadWeightsExtractor(BasePartitionCriteria):
    def __call__(self, db):
        return np.array([1.0, 2.0] * len(db))


def test_binary_random_split_criterion_rejects_non_column_weights():
    db = _make_db(3)
    criterion = BinaryRandomSplitCriterion(weights_favors_B=_BadWeightsExtractor())

    with pytest.raises(ValueError):
        criterion(db)


def test_split_in2_datasets_rejects_empty_dataset():
    from scm.moliterate.partition_criteria.criteria_1d_splitting import (
        split_in2_datasets,
    )

    with pytest.raises(ValueError, match="len_db must be larger than 1"):
        split_in2_datasets(0)


def test_split_in2_datasets_uses_absolute_count_when_size_ge_one():
    from scm.moliterate.partition_criteria.criteria_1d_splitting import (
        split_in2_datasets,
    )

    labels = split_in2_datasets(5, labelA="A", labelB="B", labelB_size=2, seed=0)

    assert np.count_nonzero(labels == "B") == 2


def test_split_random_simple_case_uses_test_label():
    from scm.moliterate.partition_criteria.criteria_1d_splitting import split_random

    labels = split_random(5, val_size=0.0, test_size=0.4, random_state=0)

    assert np.count_nonzero(labels == "test_set") == 2
    assert np.count_nonzero(labels == "train_set") == 3


def test_split_random_rejects_invalid_sizes():
    from scm.moliterate.partition_criteria.criteria_1d_splitting import split_random

    with pytest.raises(ValueError, match="test_size and val_size together must be less than 1"):
        split_random(5, val_size=0.6, test_size=0.4)
