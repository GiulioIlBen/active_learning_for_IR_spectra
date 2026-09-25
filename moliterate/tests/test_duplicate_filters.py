import numpy as np
import pytest
from ase import Atoms

from scm.moliterate.core import BaseFilter, ChemDataEntry, PropertyInfo
from scm.moliterate.filters import RemoveDuplicates
from scm.moliterate.interfaces.in_memory import InMemoryMolData
from scm.moliterate.transforms import ToArrayTransform


def _duplicate_filter_options():
    return ("AgglGyFmaxRand", "AgglXYZEucl", "AgglXYZEuclKB", "AgglMBTRCos")


def _build_duplicate_filter(option: str) -> RemoveDuplicates:
    if option == "AgglMBTRCos":
        pytest.importorskip("dscribe.descriptors")
    return RemoveDuplicates.build(options=option, num_samples_by_group=1, fcluster_threshold=0.3, seed=22)


@pytest.fixture()
def sample_duplicate_db():
    rows = [
        ChemDataEntry(
            system=Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 1]]), properties={"forces": [[0, 0, 1]]}
        ),
        ChemDataEntry(
            system=Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 1]]), properties={"forces": [[0, 0, 1]]}
        ),
        ChemDataEntry(
            system=Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 2]]), properties={"forces": [[0, 0, 1]]}
        ),
        ChemDataEntry(
            system=Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 2.1]]), properties={"forces": [[0, 0, 1]]}
        ),
        ChemDataEntry(system=Atoms(numbers=[2], positions=[[1, 1, 1]]), properties={"forces": [[0, 0, 1]]}),
    ]
    db = InMemoryMolData.create(
        available_properties=[PropertyInfo(name="forces")], transforms=[ToArrayTransform(prop_names=["forces"])]
    )
    db.add_systems(mol_data_rows=rows)
    return db


@pytest.mark.parametrize("option", _duplicate_filter_options())
def test_duplicate_filters_are_base_filter(option):
    duplicate_filter = _build_duplicate_filter(option)
    assert isinstance(duplicate_filter, BaseFilter)


@pytest.mark.parametrize("option", _duplicate_filter_options())
def test_duplicate_filters_select_one_per_group(option, sample_duplicate_db):
    duplicate_filter = _build_duplicate_filter(option)

    result = np.asarray(duplicate_filter.apply_filter(sample_duplicate_db), dtype=np.int64)
    print(result)

    assert np.array_equal(result, np.sort(result))
    assert len(np.unique(result)) == len(result)
    assert set(result).issubset({0, 1, 2, 3, 4})
    assert 4 in result  # unique formula should always be kept
    assert len(set(result) & {0, 1}) == 1
    assert len(set(result) & {3, 2}) == 1
