from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
from scm.plams import Atom, Molecule

from scm.active_learning.engines import ASEEngine, PLAMSMolecule
from scm.active_learning.results import ASEMolecularDynamicsResult
from scm.active_learning.tasks import ASEMolecularDynamics

ase_md_module = importlib.import_module("scm.active_learning.tasks.ase_md")


class DummyCalculator:
    def __init__(self):
        self.results = {
            "energy_var": 0.125,
            "forces_comm": np.array(
                [
                    [[1.0, 0.0, 0.0]],
                    [[3.0, 0.0, 0.0]],
                ]
            ),
        }


class DummyAtoms:
    def __init__(self):
        self.calc = None
        self.info = {}
        self.cell = None
        self.pbc = None
        self.constraints = None
        self.positions = np.array([[0.0, 0.0, 0.0]])

    def __len__(self):
        return 1

    def set_cell(self, cell):
        self.cell = cell

    def set_pbc(self, pbc):
        self.pbc = pbc

    def set_constraint(self, constraints):
        self.constraints = constraints

    def get_potential_energy(self):
        return 4.0

    def get_kinetic_energy(self):
        return 2.0

    def get_forces(self):
        return np.array([[2.0, 0.0, 0.0]])

    def get_dipole_moment(self):
        assert self.calc is not None
        self.calc.results["dipole"] = np.array([0.1, 0.2, 0.3])
        return self.calc.results["dipole"]

    def get_positions(self):
        return self.positions


class FakeRow:
    def __init__(self, atoms):
        self._atoms = atoms

    def toatoms(self, add_additional_information=False):
        del add_additional_information
        return self._atoms


class FakeDBConnection:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = []

    def count(self):
        return len(self.rows)

    def write(self, atoms, **kwargs):
        del kwargs
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("db\n", encoding="utf-8")
        self.rows.append(FakeRow(atoms))

    def select(self):
        return list(self.rows)


class FakeDBFactory:
    def __init__(self):
        self.connections = {}

    def __call__(self, path, append=True):
        del append
        path = str(path)
        if path not in self.connections:
            self.connections[path] = FakeDBConnection(path)
        return self.connections[path]


class FakeLangevin:
    def __init__(self, atoms, timestep, temperature_K, friction, fixcm):
        self.atoms = atoms
        self.timestep = timestep
        self.temperature_K = temperature_K
        self.friction = friction
        self.fixcm = fixcm
        self.callbacks = []
        self.max_steps = None
        self.nsteps = 0

    def attach(self, callback, interval):
        self.callbacks.append((callback, interval))

    def get_time(self):
        return float(self.nsteps)

    def run(self, nsteps):
        self.max_steps = nsteps
        self.nsteps = nsteps
        for callback, _interval in self.callbacks:
            callback()


class FakeVelocityVerlet:
    def __init__(self, atoms, timestep):
        self.atoms = atoms
        self.timestep = timestep
        self.callbacks = []
        self.max_steps = None
        self.nsteps = 0

    def attach(self, callback, interval):
        self.callbacks.append((callback, interval))

    def get_time(self):
        return float(self.nsteps)

    def run(self, nsteps):
        self.max_steps = nsteps
        self.nsteps = nsteps


class FakeConstraint:
    def __init__(self, indices):
        self.indices = indices


def _make_plams_container() -> PLAMSMolecule:
    mol = Molecule()
    mol.add_atom(Atom(symbol="Ar", coords=(0.0, 0.0, 0.0)))
    return PLAMSMolecule.from_molecules(system_id="Ar", systems=mol)


def test_ase_md_resolves_default_attachments_from_task_settings():
    task = ASEMolecularDynamics(
        atomistic_system=_make_plams_container(),
        samplingfreq=7,
        exit_condition_freq=3,
        minimum_distance=0.6,
        error_threshold=0.2,
    )

    attachments = task._resolved_attachments()

    assert [type(attachment).__name__ for attachment in attachments] == [
        "ASEMDDBWriter",
        "ASEMDStopOnDistance",
        "ASEMDStopOnError",
    ]
    assert attachments[0].interval == 7
    assert attachments[1].interval == 3
    assert attachments[2].interval == 3


def test_ase_md_task_run_creates_db_result_and_outputs(monkeypatch, tmp_path):
    task = ASEMolecularDynamics(
        atomistic_system=_make_plams_container(),
        save_folder=str(tmp_path),
        nsteps=20,
        timestep=0.5,
        samplingfreq=5,
        temperature=350.0,
        exit_condition_freq=5,
        error_threshold=1.0,
        minimum_distance=None,
        use_cell=True,
        cell=[10.0, 10.0, 10.0],
        pbc=[True, True, True],
        constraints=["[FixAtoms(indices=[0])]"],
    )
    engine = ASEEngine(engine_id="ase", calculator="tests.test_ase_md_task.DummyCalculator")
    atoms = DummyAtoms()
    fake_db_factory = FakeDBFactory()

    monkeypatch.setattr(ASEEngine, "_to_ase_atoms", staticmethod(lambda system: atoms))
    monkeypatch.setattr(ASEEngine, "_build_calculator", lambda self: DummyCalculator())
    monkeypatch.setattr(
        ase_md_module,
        "_require_ase_md_dependencies",
        lambda: (
            type("FakeConstraintsModule", (), {"FixAtoms": FakeConstraint}),
            type("FakeUnits", (), {"fs": 1.0}),
            fake_db_factory,
            {
                "VelocityVerlet": FakeVelocityVerlet,
                "Langevin": FakeLangevin,
                "NVTBerendsen": FakeVelocityVerlet,
                "NPTBerendsen": FakeVelocityVerlet,
            },
        ),
    )

    result = task.run(engine=engine)

    assert isinstance(result, ASEMolecularDynamicsResult)
    assert result.from_task == task
    assert result.from_engine == engine
    assert result.trajectory_path == str(tmp_path / "ase_md_results" / "trajectory.db")
    assert result.restart_used is False
    assert result.stop_triggered is False
    assert result.stop_reasons == []
    assert result.executed_steps == 20
    assert result.written_frames == 1
    assert result.dynamics_type == "FakeLangevin"

    assert atoms.cell == [10.0, 10.0, 10.0]
    assert atoms.pbc == [True, True, True]
    assert len(atoms.constraints) == 1
    assert atoms.info["eerr"] == np.sqrt(0.125)
    assert atoms.info["ferr"] == np.sqrt(2.0)
    assert Path(result.trajectory_path).exists()


def test_ase_md_force_uncertainty_prefers_calibrated_variance():
    atoms = DummyAtoms()
    atoms.calc = DummyCalculator()
    atoms.calc.results["forces_var"] = np.full((1, 3), 4.0)

    assert np.isclose(ASEMolecularDynamics._max_force_error(atoms), np.sqrt(12.0))


def test_ase_md_task_can_cache_dipole_before_writing(monkeypatch, tmp_path):
    task = ASEMolecularDynamics(
        atomistic_system=_make_plams_container(),
        save_folder=str(tmp_path),
        nsteps=5,
        samplingfreq=1,
        save_dipole_moment=True,
        minimum_distance=None,
        error_threshold=None,
    )
    engine = ASEEngine(engine_id="ase", calculator="tests.test_ase_md_task.DummyCalculator")
    atoms = DummyAtoms()
    fake_db_factory = FakeDBFactory()

    monkeypatch.setattr(ASEEngine, "_to_ase_atoms", staticmethod(lambda system: atoms))
    monkeypatch.setattr(ASEEngine, "_build_calculator", lambda self: DummyCalculator())
    monkeypatch.setattr(
        ase_md_module,
        "_require_ase_md_dependencies",
        lambda: (
            type("FakeConstraintsModule", (), {"FixAtoms": FakeConstraint}),
            type("FakeUnits", (), {"fs": 1.0}),
            fake_db_factory,
            {
                "VelocityVerlet": FakeVelocityVerlet,
                "Langevin": FakeLangevin,
                "NVTBerendsen": FakeVelocityVerlet,
                "NPTBerendsen": FakeVelocityVerlet,
            },
        ),
    )

    task.run(engine=engine)

    np.testing.assert_allclose(atoms.calc.results["dipole"], np.array([0.1, 0.2, 0.3]))
