import numpy as np
import pytest
from ase import Atoms

from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.interfaces.in_memory import InMemoryMolData
from scm.moliterate.partition_criteria.criteria_3d import GetXYZRepresentation


def _molecule_from_coords(coords, symbols=None):
    symbols = symbols or ["H"] * len(coords)
    return Atoms(symbols=symbols, positions=coords)


def test_get_xyz_representation_collects_positions_in_order():
    coords_a = [(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)]
    coords_b = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]
    db = InMemoryMolData.create()
    db.add_system(ChemDataEntry(system=_molecule_from_coords(coords_a), properties={}))
    db.add_system(ChemDataEntry(system=_molecule_from_coords(coords_b), properties={}))

    criterion = GetXYZRepresentation()
    descriptors = criterion(db)

    assert descriptors.shape == (2, 2, 3)
    np.testing.assert_allclose(descriptors[0], np.array(coords_a))
    np.testing.assert_allclose(descriptors[1], np.array(coords_b))


def test_get_xyz_representation_honors_wrap_flag():
    class DummyAtoms:
        def __init__(self, positions, numbers):
            self._positions = np.array(positions)
            self._numbers = np.array(numbers)
            self.wrap_calls = []

        def get_positions(self, wrap=False):
            self.wrap_calls.append(wrap)
            return self._positions

        @property
        def numbers(self):
            return self._numbers

    class DummyDB:
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            return iter(self._rows)

    atoms = DummyAtoms([[1.0, 2.0, 3.0]], [1])
    row = type("Row", (), {})()
    row.atoms = atoms
    db = DummyDB([row])

    descriptors = GetXYZRepresentation(wrap=True, check_atoms_ordering=True)(db)

    assert atoms.wrap_calls == [True]
    np.testing.assert_allclose(descriptors, np.array([[[1.0, 2.0, 3.0]]]))


def test_get_xyz_representation_raises_when_atom_order_changes():
    first = _molecule_from_coords([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)], symbols=["H", "O"])
    swapped = _molecule_from_coords([(0.0, 0.0, 0.0), (0.0, 0.0, 1.0)], symbols=["O", "H"])
    db = InMemoryMolData.create()
    db.add_system(ChemDataEntry(system=first, properties={}))
    db.add_system(ChemDataEntry(system=swapped, properties={}))

    with pytest.raises(ValueError):
        GetXYZRepresentation(check_atoms_ordering=True)(db)
