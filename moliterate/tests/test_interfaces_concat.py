from ase import Atoms

from scm.moliterate.core import ChemDataEntry, ConcatChemDataSet, PropertyInfo
from scm.moliterate.interfaces.in_memory import InMemoryMolData


def _make_in_memory(label, energies):
    props_info = [PropertyInfo(name="energy")]
    db = InMemoryMolData.create(available_properties=props_info)
    rows = [ChemDataEntry(system=Atoms(numbers=[1], positions=[[0, 0, 0]]), properties={"energy": e}) for e in energies]
    db.add_systems(rows)
    db.update_metadata(tag=label)
    return db


def test_flatten_idxs_to_each_iter():
    db_one = _make_in_memory("one", [1.0, 2.0, 3.0, 4.0, 5.0])
    db_two = _make_in_memory("two", [6.0, 7.0, 8.0])
    concat = ConcatChemDataSet(data_source=[db_one, db_two])
    flatten_idxs = concat[[1, 6, 7]].flatten_idxs_to_each_iter()

    assert all(all(a == b) for a, b in zip(flatten_idxs, [[1], [1, 2]]))


def test_concat_dataset_iter_and_subset():
    db_one = _make_in_memory("one", [1.0, 2.0])
    db_two = _make_in_memory("two", [3.0])
    concat = ConcatChemDataSet(data_source=[db_one, db_two])

    assert len(concat) == 3
    assert {p.name for p in concat.available_properties} == {"energy"}
    assert concat.metadata["tag"] == "two"

    rows = list(concat)
    assert [row.idx_origin for row in rows] == [0, 1, 0]
    assert [row.properties["energy"] for row in rows] == [1.0, 2.0, 3.0]

    subset = concat.subset([1, 2])
    subset_rows = list(subset)
    assert [row.idx_origin for row in subset_rows] == [1, 0]
    assert [row.properties["energy"] for row in subset_rows] == [2.0, 3.0]


def test_concat_dataset_get_row():
    db_one = _make_in_memory("one", [1.0, 2.0])
    db_two = _make_in_memory("two", [3.0])
    concat = ConcatChemDataSet(data_source=[db_one, db_two])

    row0 = concat.get_row(0)
    assert row0.idx_absolute == 0
    assert row0.idx_origin == 0
    assert row0.properties["energy"] == 1.0

    row1 = concat.get_row(1)
    assert row1.idx_absolute == 1
    assert row1.idx_origin == 1
    assert row1.properties["energy"] == 2.0

    row2 = concat.get_row(2)
    assert row2.idx_absolute == 2
    assert row2.idx_origin == 0
    assert row2.properties["energy"] == 3.0
