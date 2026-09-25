from __future__ import annotations

from types import SimpleNamespace

import pytest
from ase import Atoms
from scm.moliterate import PropertyInfo
from scm.plams import Atom, Molecule

from scm.active_learning.engines import PLAMSMolecule, TorchEngine
from scm.active_learning.task_parallelization import SerialStrategy
from scm.active_learning.tasks import TorchSimGOTask, TorchSimMDTask


def build_lennard_jones_model(*, stress: bool = False):
    pytest.importorskip("torch_sim", reason="Install the torchsim extra to run TorchSim smoke tests.")
    torch = pytest.importorskip("torch", reason="Install the torchsim extra to run TorchSim smoke tests.")
    from torch_sim.models.lennard_jones import LennardJonesModel

    return LennardJonesModel(device=torch.device("cpu"), compute_stress=stress)


def _make_container() -> PLAMSMolecule:
    molecule = Molecule()
    molecule.add_atom(Atom(symbol="Ar", coords=(0.0, 0.0, 0.0)))
    molecule.add_atom(Atom(symbol="Ar", coords=(3.5, 0.0, 0.0)))
    return PLAMSMolecule.from_molecules(system_id="Ar2", systems=molecule)


def test_torchsim_single_point_smoke():
    pytest.importorskip("torch_sim", reason="Install the torchsim extra to run TorchSim smoke tests.")
    engine = TorchEngine(engine_id="torchsim", model_factory=f"{__name__}.build_lennard_jones_model")
    dataset = [SimpleNamespace(chemical_system=Atoms("Ar2", positions=[(0.0, 0.0, 0.0), (3.5, 0.0, 0.0)]))]
    props = [PropertyInfo(name="energy", unit="eV"), PropertyInfo(name="forces", unit="eV/Ang")]

    results = list(engine.run_single_point(props, dataset, SerialStrategy()))

    assert "energy" in results[0]
    assert "forces" in results[0]


def test_torchsim_md_task_smoke(tmp_path):
    pytest.importorskip("torch_sim", reason="Install the torchsim extra to run TorchSim smoke tests.")
    engine = TorchEngine(engine_id="torchsim", model_factory=f"{__name__}.build_lennard_jones_model")
    task = TorchSimMDTask(
        atomistic_system=_make_container(),
        save_folder=str(tmp_path),
        integrator="nve",
        nsteps=1,
        timestep=0.001,
        trajectory_reporter_kwargs={"state_frequency": 1},
    )

    result = task.run(engine)

    assert result.trajectory_path.endswith(".h5")


def test_torchsim_go_task_smoke(tmp_path):
    pytest.importorskip("torch_sim", reason="Install the torchsim extra to run TorchSim smoke tests.")
    engine = TorchEngine(engine_id="torchsim", model_factory=f"{__name__}.build_lennard_jones_model")
    task = TorchSimGOTask(
        atomistic_system=_make_container(),
        save_folder=str(tmp_path),
        optimizer="fire",
        convergence_kind="force",
        force_tol=1e6,
        trajectory_reporter_kwargs={"state_frequency": 1},
    )

    result = task.run(engine)

    assert result.trajectory_path.endswith(".h5")
