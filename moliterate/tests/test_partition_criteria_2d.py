import builtins
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.interfaces.in_memory import InMemoryMolData
from scm.moliterate.partition_criteria.criteria_2d import (
    GetMACERepresentation,
    GyrationRFmaxCriterion,
    MBTRCriterion,
    _ensure_ase_get_constraints_compat,
)


def _make_db_with_forces(*rows):
    db = InMemoryMolData.create(available_properties=[PropertyInfo(name="forces", unit="")])
    for row in rows:
        db.add_system(row)
    return db


def test_gyration_rfmax_returns_expected_values():
    forces_a = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    forces_b = np.array([[3.0, 4.0, 0.0]])
    molecule = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 1.5]])
    molecule2 = Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 2.0]])
    row_a = ChemDataEntry(system=molecule, properties={"forces": forces_a})
    row_b = ChemDataEntry(system=molecule2, properties={"forces": forces_b})
    db = _make_db_with_forces(row_a, row_b)

    criterion = GyrationRFmaxCriterion()
    result = criterion(db)

    expected = np.array(
        [
            [float(row_a.get_gyration_radius()), 1.0],
            [float(row_b.get_gyration_radius()), 5.0],
        ]
    )
    np.testing.assert_allclose(result, expected)


def test_mbtr_descriptor_runs_when_installed(db_in_memory):
    pytest.importorskip("dscribe.descriptors")
    criterion = MBTRCriterion(species=("H", "He"), geometry_function="atomic_number")

    descriptor = criterion(db_in_memory)
    assert descriptor.shape[0] == 3
    assert descriptor.ndim == 2
    assert np.isfinite(descriptor).all()


def test_mbtr_ase_private_constraints_compat(monkeypatch):
    monkeypatch.delattr(Atoms, "_get_constraints", raising=False)
    _ensure_ase_get_constraints_compat()

    atoms = Atoms(numbers=[1], positions=[[0, 0, 0]])
    assert hasattr(atoms, "_get_constraints")
    assert atoms._get_constraints() == atoms.constraints


def test_auto_determin_mbtr_grid_uses_geometry_bounds(db_in_memory):
    pytest.importorskip("dscribe.descriptors")
    criterion = MBTRCriterion.auto_determine_mbtr_grid(db_in_memory, delta_x=0.5)

    assert criterion.grid["min"] == 1.5
    assert criterion.grid["max"] == 2.5
    assert criterion.grid["n"] == int((criterion.grid["max"] - criterion.grid["min"]) / 0.5)
    assert criterion.periodic is False
    assert criterion.species == ("H", "He")
    assert criterion.species == ("H", "He")


def test_mace_representation_runs_when_installed(db_in_memory):
    pytest.importorskip("mace.calculators")
    criterion = GetMACERepresentation(model_type="off", model_size="small", device="cpu", warning_pickle_off=True)
    descriptor = criterion(db_in_memory[:2])
    assert descriptor.shape[0] == 2
    assert descriptor.ndim == 2
    assert np.isfinite(descriptor).all()

    with pytest.raises(ValueError, match="Found an atomic species that are not supported by the model:"):
        descriptor = criterion(db_in_memory[[2]])
