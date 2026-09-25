from __future__ import annotations

from types import SimpleNamespace

import pytest
from scm.moliterate import PropertyInfo
from scm.plams import Atom, Molecule, Settings

from scm.active_learning.engines.ams import AMSEngine
from scm.active_learning.task_parallelization import SerialStrategy


class DummyResults:
    def __init__(self, energy=1.0, gradients=2.0, dipole=3.0):
        self._energy = energy
        self._gradients = gradients
        self._dipole = dipole

    def get_energy(self):
        return self._energy

    def get_gradients(self):
        return self._gradients

    def get_dipolemoment(self):
        return self._dipole


class DummyJob:
    def __init__(self, ok=True, check=True, err="boom", results=None):
        self._ok = ok
        self._check = check
        self._err = err
        self.results = results or DummyResults()

    def ok(self):
        return self._ok

    def check(self):
        return self._check

    def get_errormsg(self):
        return self._err


def test_validate_properties_rejects_unknown():
    engine = AMSEngine(engine_id="e")

    with pytest.raises(ValueError):
        engine._validate_properties([PropertyInfo(name="Unknown")])


def test_convert_property_settings_sets_non_energy_flags():
    engine = AMSEngine(engine_id="e")
    props = [PropertyInfo(name="Energy"), PropertyInfo(name="Forces"), PropertyInfo(name="DipoleMoment")]

    settings = engine._convert_property_to_ams_sp_settings(props)

    props_dict = settings.input.ams.Properties.as_dict()
    assert props_dict["Gradients"] is True
    assert props_dict["DipoleMoment"] is True
    assert "Energy" not in props_dict


def test_extract_single_point_results_failure():
    engine = AMSEngine(engine_id="e")
    job = DummyJob(ok=False, check=True, err="bad")

    res = engine._extract_single_point_results(job, [PropertyInfo(name="Energy")])

    assert res == {"FAILURE": "bad"}


def test_extract_single_point_results_success():
    engine = AMSEngine(engine_id="e")
    job = DummyJob(results=DummyResults(energy=1.5, gradients=2.5, dipole=3.5))
    props = [
        PropertyInfo(name="Energy"),
        PropertyInfo(name="Forces"),
        PropertyInfo(name="Gradients"),
        PropertyInfo(name="DipoleMoment"),
    ]

    res = engine._extract_single_point_results(job, props)

    assert res["Energy"] == 1.5
    assert res["Forces"] == -2.5
    assert res["Gradients"] == 2.5
    assert res["DipoleMoment"] == 3.5


def test_from_amsjob_strips_ams_settings():
    settings = Settings()
    settings.input.ams.task = "SinglePoint"
    job = SimpleNamespace(settings=settings)

    engine = AMSEngine.from_amsjob(job, engine_id="e")

    assert "ams" not in engine.settings.input


def test_run_single_point_rejects_duplicate_inputs_before_running(monkeypatch):
    molecule = Molecule()
    molecule.add_atom(Atom(symbol="H", coords=(0.0, 0.0, 0.0)))
    dataset = [
        SimpleNamespace(chemical_system=molecule.copy()),
        SimpleNamespace(chemical_system=molecule.copy()),
    ]
    run_called = False

    def fake_run(self, *args, **kwargs):
        nonlocal run_called
        run_called = True

    monkeypatch.setattr("scm.active_learning.engines.ams.AMSJob.run", fake_run)
    engine = AMSEngine(engine_id="e")

    with pytest.raises(ValueError, match=r"duplicate->original indices: 1->0"):
        list(
            engine.run_single_point(
                properties=[PropertyInfo(name="energy", unit="eV")],
                dataset=dataset,
                parallel_settings=SerialStrategy(),
            )
        )

    assert run_called is False
