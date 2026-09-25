from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from scm.moliterate import PropertyInfo

from scm.active_learning.engines.ams import AMSEngine
from scm.active_learning.engines.ase_engine import ASEEngine
from scm.active_learning.engines.mlip_ams import ParAMSEngine
from scm.active_learning.task_parallelization import SerialStrategy


class DummyAMSResults:
    def __init__(self, energy=1.0, gradients=None, dipole=None):
        self._energy = energy
        self._gradients = np.array([[2.0, -4.0, 6.0]]) if gradients is None else gradients
        self._dipole = np.array([3.0, 6.0, 9.0]) if dipole is None else dipole

    def get_energy(self):
        return self._energy

    def get_gradients(self):
        return self._gradients

    def get_dipolemoment(self):
        return self._dipole


class DummyAMSJob:
    def __init__(self, results=None):
        self.results = results or DummyAMSResults()

    def ok(self):
        return True

    def check(self):
        return True

    def get_errormsg(self):
        return "boom"


class DummyCalculator:
    pass


class DummyASEAtoms:
    def __init__(self):
        self.calc = None

    def get_potential_energy(self, force_consistent: bool = False):
        return 2.5 if force_consistent else 1.5

    def get_forces(self):
        return np.array([[1.0, -2.0, 3.0]])

    def get_dipole_moment(self):
        return np.array([4.0, 5.0, 6.0])


def test_ams_engine_converts_from_native_ams_units(monkeypatch):
    engine = AMSEngine(engine_id="ams")
    job = DummyAMSJob(results=DummyAMSResults(energy=1.5))

    monkeypatch.setattr(
        PropertyInfo,
        "unit_conversion_from",
        lambda self, from_units: {
            ("Energy", "Hartree"): 2.0,
            ("Gradients", "Hartree/Bohr"): 5.0,
            ("Forces", "Hartree/Bohr"): 5.0,
            ("DipoleMoment", "e*Bohr"): 7.0,
        }[(self.name, from_units)],
    )

    results = engine._extract_single_point_results(
        job,
        [
            PropertyInfo(name="Energy", unit="eV"),
            PropertyInfo(name="Gradients", unit="eV/Ang"),
            PropertyInfo(name="Forces", unit="eV/Ang"),
            PropertyInfo(name="DipoleMoment", unit="Debye"),
        ],
    )

    assert results["Energy"] == 3.0
    np.testing.assert_allclose(results["Gradients"], np.array([[10.0, -20.0, 30.0]]))
    np.testing.assert_allclose(results["Forces"], np.array([[-10.0, 20.0, -30.0]]))
    np.testing.assert_allclose(results["DipoleMoment"], np.array([21.0, 42.0, 63.0]))


def test_ase_engine_converts_from_native_ase_units(monkeypatch):
    engine = ASEEngine(engine_id="ase", calculator=f"{__name__}.DummyCalculator")
    atoms = DummyASEAtoms()

    monkeypatch.setattr(ASEEngine, "_to_ase_atoms", staticmethod(lambda system: atoms))
    monkeypatch.setattr(ASEEngine, "_build_calculator", lambda self: DummyCalculator())
    monkeypatch.setattr(
        PropertyInfo,
        "unit_conversion_from",
        lambda self, from_units: {
            ("energy", "eV"): 2.0,
            ("free_energy", "eV"): 3.0,
            ("forces", "eV/Ang"): 5.0,
            ("dipole_moment", "e*Ang"): 7.0,
        }[(self.name, from_units)],
    )

    results = list(
        engine.run_single_point(
            [
                PropertyInfo(name="energy", unit="Hartree"),
                PropertyInfo(name="free_energy", unit="Hartree"),
                PropertyInfo(name="forces", unit="Hartree/Bohr"),
                PropertyInfo(name="dipole_moment", unit="Debye"),
            ],
            [SimpleNamespace(chemical_system="system-1")],
            SerialStrategy(),
        )
    )

    assert len(results) == 1
    assert results[0]["energy"] == 3.0
    assert results[0]["free_energy"] == 7.5
    np.testing.assert_allclose(results[0]["forces"], np.array([[5.0, -10.0, 15.0]]))
    np.testing.assert_allclose(results[0]["dipole_moment"], np.array([28.0, 35.0, 42.0]))
    assert isinstance(atoms.calc, DummyCalculator)


def test_params_engine_delegates_unitful_requests_to_ams_engine(monkeypatch):
    seen = {}

    def fake_run_single_point(self, properties, dataset, parallel_settings, **kwargs):
        seen["properties"] = [(prop.name, prop.unit) for prop in properties]
        seen["dataset"] = list(dataset)
        seen["parallel_settings"] = parallel_settings
        seen["kwargs"] = kwargs
        yield {"Energy": 1.23}

    monkeypatch.setattr(AMSEngine, "run_single_point", fake_run_single_point)
    engine = ParAMSEngine(engine_id="params", source_settings=[{}], job_path="/tmp/params")
    props = [PropertyInfo(name="Energy", unit="eV"), PropertyInfo(name="Forces", unit="eV/Ang")]
    dataset = [SimpleNamespace(chemical_system="system-1")]
    parallel = SerialStrategy()

    results = list(engine.run_single_point(props, dataset, parallel, folder="ignored"))

    assert results == [{"Energy": 1.23}]
    assert seen["properties"] == [("Energy", "eV"), ("Forces", "eV/Ang")]
    assert seen["dataset"] == dataset
    assert seen["parallel_settings"] is parallel
    assert seen["kwargs"] == {}
