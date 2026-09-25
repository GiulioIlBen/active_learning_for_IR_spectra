import json

import pytest
from ase import Atoms

from scm.moliterate.core import ChemDataEntry


def test_chem_data_entry_metadata_validation():
    entry = ChemDataEntry(system=Atoms(), metadata=None)
    assert entry.metadata == {}

    entry = ChemDataEntry(system=Atoms(), metadata=[("tag", "a")])
    assert entry.metadata == {"tag": "a"}

    with pytest.raises(TypeError, match="metadata keys must be strings"):
        ChemDataEntry(system=Atoms(), metadata={1: "bad"})

    with pytest.raises(TypeError, match="not JSON compatible"):
        ChemDataEntry(system=Atoms(), metadata={"bad": {1}})


def test_chem_data_entry_str_includes_keys_and_indices():
    mol = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 0.74]])
    props = {"energy": -1.0, "dipole": [0.0, 0.0, 1.0]}
    metadata = {"tag": "a", "source": "unit"}
    entry = ChemDataEntry(system=mol, properties=props, metadata=metadata, idx_absolute=3, idx_origin="origin")
    expected = (
        f"ChemDataEntry({mol.get_chemical_formula()}, prop:{list(props.keys())}, "
        f"md:{list(metadata.keys())}, idx_absolute=3, idx_origin=origin)"
    )
    assert str(entry) == expected


def test_chem_data_entry_json_dump_excludes_system():
    mol = Atoms(numbers=[8], positions=[[0, 0, 0]])
    entry = ChemDataEntry(
        system=mol,
        properties={"energy": -1.0, "dipole": [1, 2, 3]},
        metadata={"tag": "a", "flags": ["x"]},
        idx_absolute=5,
        idx_origin="origin",
    )
    dumped = json.loads(entry.model_dump_json(exclude={"system"}))
    assert dumped == {
        "properties": {"energy": -1.0, "dipole": [1, 2, 3]},
        "metadata": {"tag": "a", "flags": ["x"]},
        "idx_absolute": 5,
        "idx_origin": "origin",
    }


def test_chem_data_entry_to_ase_atoms_sets_info_and_results():
    mol = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 0.74]])
    entry = ChemDataEntry(
        system=mol,
        properties={"energy": -1.0},
        metadata={"tag": "unit"},
        idx_absolute=2,
        idx_origin="origin",
    )
    atoms = entry.to_ase_atoms()
    assert atoms.info["tag"] == "unit"
    assert atoms.info["idx_origin"] == "origin"
    assert atoms.info["idx_absolute"] == 2
    assert atoms.calc.results["energy"] == -1.0


def test_chem_data_entry_molecule_and_to_plams_molecule():
    plams = pytest.importorskip("scm.plams")
    from scm.plams.interfaces.molecule.ase import fromASE as ase_to_plams

    mol = Atoms(numbers=[6], positions=[[0, 0, 0]])
    plams_mol = ase_to_plams(mol)
    entry_from_plams = ChemDataEntry(system=plams_mol)
    assert isinstance(entry_from_plams.atoms, Atoms)
    assert isinstance(entry_from_plams.molecule, plams.Molecule)

    entry = ChemDataEntry(
        system=mol,
        metadata={"tag": "unit"},
        properties={"energy": -1.0},
        idx_absolute=4,
        idx_origin="origin",
    )
    out = entry.to_plams_molecule(transfer_properties=True)
    assert isinstance(out, plams.Molecule)
    assert out[1].properties.info["tag"] == "unit"
    assert out[1].properties.info["idx_origin"] == "origin"
    assert out[1].properties.info["idx_absolute"] == 4
    assert out.properties.get("energy") == -1.0


def test_chem_data_entry_chemical_system_roundtrip():
    libbase = pytest.importorskip("scm.base")

    mol = Atoms(numbers=[8], positions=[[0, 0, 0]])
    chem_sys = libbase.ChemicalSystem.from_ase_atoms(mol)
    entry = ChemDataEntry(system=chem_sys)
    assert isinstance(entry.atoms, Atoms)
    assert isinstance(entry.chemical_system, libbase.ChemicalSystem)
