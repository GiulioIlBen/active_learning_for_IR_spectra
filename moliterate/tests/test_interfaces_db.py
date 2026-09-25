import pytest
from ase import Atoms

from scm.moliterate.core import ChemDataEntry
from scm.moliterate.core.base_chem_dataset_writer import BaseChemDataSetWriter
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.interfaces.ase_database import ASEMolData
from scm.moliterate.interfaces.in_memory import InMemoryMolData


@pytest.fixture(params=[InMemoryMolData, ASEMolData])
def db_interface_cls(request):
    return request.param


@pytest.fixture(
    params=[
        # (PropertyInfo(name="energy", unit="eV"),),
        (PropertyInfo(name="energy", unit="eV"), PropertyInfo(name="dipole", unit="eAng")),
    ]
)
def available_properties(request):
    return list(request.param)


@pytest.fixture
def db_with_dipole(db_interface_cls, tmp_path):
    data_source = tmp_path / f"{db_interface_cls.__name__}.db"
    available_properties = [PropertyInfo(name="energy", unit="eV"), PropertyInfo(name="dipole", unit="eAng")]
    # energy uniform, [dipole, stress] not uniform

    db = db_interface_cls.create(data_source=str(data_source), available_properties=available_properties)
    yield db
    db.unlink(missing_ok=True)


@pytest.fixture(
    params=[
        # [energy, dipole] uniform, stress not uniform
        (
            ChemDataEntry(system=Atoms(), properties={"energy": -1.0, "dipole": [1, 2, 3]}, metadata={"tag": "a"}),
            ChemDataEntry(
                system=Atoms(), properties={"energy": -2.0, "dipole": [1.1, 2.1, 3.1]}, metadata={"tag": "b"}
            ),
            ChemDataEntry(
                system=Atoms(),
                properties={"energy": -3.0, "dipole": [1.2, 2.2, 3.2], "stress": [1, 2, 3, 4, 5, 6]},
                metadata={"tag": "c"},
            ),
        ),
    ]
)
def mol_rows(request):
    return list(request.param)


@pytest.fixture
def db(db_interface_cls, available_properties, mol_rows, tmp_path):
    data_source = tmp_path / f"{db_interface_cls.__name__}.db"
    db = db_interface_cls.create(data_source=str(data_source), available_properties=available_properties)
    db.add_systems(mol_rows)
    yield db
    db.unlink(missing_ok=True)


def test_creation_sets_available_properties(db: BaseChemDataSetWriter, available_properties):
    assert db.available_properties == available_properties
    assert db.metadata["__properties__"] == available_properties

    copied_metadata = db.metadata
    copied_metadata["__properties__"] = []
    assert db.metadata["__properties__"] == available_properties


def test_non_consistent(db_with_dipole: BaseChemDataSetWriter):
    mol_rows = [
        ChemDataEntry(system=Atoms(), properties={"energy": -1.0, "dipole": [1, 2, 3]}, metadata={"tag": "a"}),
        ChemDataEntry(system=Atoms(), properties={"energy": -2.0}, metadata={"tag": "b"}),
        ChemDataEntry(system=Atoms(), properties={"energy": -3.0, "stress": [1, 2, 3, 4, 5, 6]}, metadata={"tag": "c"}),
    ]
    with pytest.raises(KeyError, match="dipole"):
        db_with_dipole.add_systems(mol_rows)


def test_subset_returns_expected_rows(db: BaseChemDataSetWriter, mol_rows):
    assert len(db) == len(mol_rows)
    assert db.total_len() == len(mol_rows)
    assert [x.properties["energy"] for x in db] == [x.properties["energy"] for x in mol_rows]

    subset = db.subset([2, 0])

    assert len(subset) == 2
    assert [x.idx_absolute for x in subset] == subset.select_indices().tolist()


def test_metadata_updates(db: BaseChemDataSetWriter, mol_rows):
    db.add_systems(mol_rows)

    db.update_metadata(source="generated")
    assert db.metadata["source"] == "generated"

    db.update_row_metadata(1, note="keep")
    assert db[1].metadata["note"] == "keep"

    db.update_row_metadata(0, absolute=True, priority="high")
    assert db[0].metadata["priority"] == "high"


def test_iterate_access_rows(db: BaseChemDataSetWriter, mol_rows):
    assert len(db) == len(mol_rows)
    assert db.total_len() == len(mol_rows)
    assert [x.properties["energy"] for x in db] == [x.properties["energy"] for x in mol_rows]


def test_available_properties_correspond(db: BaseChemDataSetWriter, mol_rows):
    available_properties = [x.name for x in db.available_properties]
    for r, r_init in zip(db, mol_rows):
        assert set(available_properties).issubset(set(r.properties.keys()))
        assert set(r_init.properties.keys()).issuperset(set(r.properties.keys()))


def test_data_types(db: BaseChemDataSetWriter, mol_rows, available_properties):
    avil = [x.name for x in available_properties]
    # print(avil)
    for r, r_init in zip(db, mol_rows):
        for p in avil:
            # print(p)
            assert type(r_init.properties[p]) == type(r.properties[p])
            # print(type(r_init.properties[p]), type(r.properties[p]))


def test_iterate_subset(db: BaseChemDataSetWriter, mol_rows):
    subset = db.subset([2, 0])

    assert subset.select_indices().tolist() == [0, 2]
    assert [x.idx_absolute for x in subset] == [0, 2]


def test_restore_original_from_subset(db: BaseChemDataSetWriter, mol_rows):
    subset = db.subset([2, 0])
    restored = subset.restore_original()

    assert restored.absolute_idxs is None
    assert len(restored) == len(mol_rows)
    assert restored.select_indices().tolist() == list(range(len(mol_rows)))
