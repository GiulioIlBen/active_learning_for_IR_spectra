import numpy as np
import pytest
from ase import Atoms

from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.interfaces.in_memory import InMemoryMolData
from scm.moliterate.partition_criteria.criteria_1d import MetadataCriterion
from scm.moliterate.partition_criteria.tree_1d_criteria import TreePartitionCriterion


@pytest.fixture
def db_with_levels():
    db = InMemoryMolData.create()
    db.add_system(ChemDataEntry(system=Atoms(), metadata={"level_1": "A", "level_2": "x0"}))
    db.add_system(ChemDataEntry(system=Atoms(), metadata={"level_1": "A", "level_2": "x1"}))
    db.add_system(ChemDataEntry(system=Atoms(), metadata={"level_1": "B", "level_2": "y0"}))
    db.add_system(ChemDataEntry(system=Atoms(), metadata={"level_1": "B", "level_2": "y1"}))
    return db


def test_tree_partition_criterion_keep_last_uses_only_last_level_labels(db_with_levels):
    criterion = TreePartitionCriterion(
        partition_criteria=[
            MetadataCriterion(metadata_keys=["level_1"]),
            MetadataCriterion(metadata_keys=["level_2"]),
        ],
        concat_symbol="KeepLast",
    )

    result = criterion(db_with_levels)

    assert result.tolist() == ["x0", "x1", "y0", "y1"]


def test_tree_partition_criterion_concat_symbol_builds_hierarchical_labels(db_with_levels):
    criterion = TreePartitionCriterion(
        partition_criteria=[
            MetadataCriterion(metadata_keys=["level_1"]),
            MetadataCriterion(metadata_keys=["level_2"]),
        ],
        concat_symbol="%",
    )

    result = criterion(db_with_levels)

    assert result.tolist() == ["%A%x0", "%A%x1", "%B%y0", "%B%y1"]


def test_tree_partition_make_clusters_returns_expected_relative_indices(db_with_levels):
    criterion = TreePartitionCriterion(
        partition_criteria=[
            MetadataCriterion(metadata_keys=["level_1"]),
            MetadataCriterion(metadata_keys=["level_2"]),
        ],
        concat_symbol="KeepLast",
    )

    clusters = criterion.make_clusters(db_with_levels)

    assert set(clusters) == {"x0", "x1", "y0", "y1"}
    np.testing.assert_array_equal(clusters["x0"], np.array([0]))
    np.testing.assert_array_equal(clusters["x1"], np.array([1]))
    np.testing.assert_array_equal(clusters["y0"], np.array([2]))
    np.testing.assert_array_equal(clusters["y1"], np.array([3]))


def test_tree_partition_with_empty_partition_criteria_returns_empty_labels(db_with_levels):
    criterion = TreePartitionCriterion(partition_criteria=[])

    labels = criterion(db_with_levels)
    clusters = criterion.make_clusters(db_with_levels)

    assert labels.tolist() == ["", "", "", ""]
    assert set(clusters) == {""}
    np.testing.assert_array_equal(clusters[""], np.array([0, 1, 2, 3]))
