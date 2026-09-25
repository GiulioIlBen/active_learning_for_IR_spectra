from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from scm.plams import Atom, Molecule

from scm.active_learning.engines import ASEEngine, PLAMSMolecule, TorchEngine
from scm.active_learning.results import TorchSimGOResult, TorchSimMDResult
from scm.active_learning.tasks import TorchSimGOTask, TorchSimMDTask


class FakeTorchTaskModel:
    def __init__(self, *, stress: bool = False):
        self.implemented_properties = ("energy", "forces", "stress") if stress else ("energy", "forces")
        self.compute_forces = True
        self.compute_stress = stress


def build_task_model(stress: bool = False):
    return FakeTorchTaskModel(stress=stress)


def install_fake_torchsim(monkeypatch):
    calls = {"integrate": [], "optimize": [], "force_conv": [], "energy_conv": []}

    def integrate(**kwargs):
        calls["integrate"].append(kwargs)
        _write_trajectory(kwargs["trajectory_reporter"]["filenames"][0], "md")
        return SimpleNamespace(energy=[2.5], n_steps=kwargs["n_steps"])

    def optimize(**kwargs):
        calls["optimize"].append(kwargs)
        _write_trajectory(kwargs["trajectory_reporter"]["filenames"][0], "go")
        return SimpleNamespace(energy=[-1.5])

    fake_module = SimpleNamespace(
        Integrator=SimpleNamespace(nvt_langevin="integrator:nvt_langevin"),
        Optimizer=SimpleNamespace(fire="optimizer:fire"),
        CellFilter=SimpleNamespace(frechet="cell_filter:frechet"),
        integrate=integrate,
        optimize=optimize,
        generate_force_convergence_fn=lambda tol: calls["force_conv"].append(tol) or ("force", tol),
        generate_energy_convergence_fn=lambda tol: calls["energy_conv"].append(tol) or ("energy", tol),
    )
    monkeypatch.setitem(sys.modules, "torch_sim", fake_module)
    return calls


def _write_trajectory(path: str, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _make_container() -> PLAMSMolecule:
    molecule = Molecule()
    molecule.add_atom(Atom(symbol="Ar", coords=(0.0, 0.0, 0.0)))
    return PLAMSMolecule.from_molecules(system_id="Ar", systems=molecule)


def test_torchsim_md_task_runs_and_writes_trajectory(monkeypatch, tmp_path):
    calls = install_fake_torchsim(monkeypatch)
    engine = TorchEngine(engine_id="torchsim", model_factory=f"{__name__}.build_task_model")
    task = TorchSimMDTask(
        atomistic_system=_make_container(),
        save_folder=str(tmp_path),
        nsteps=25,
        timestep=0.002,
        temperature=350.0,
        trajectory_reporter_kwargs={"state_frequency": 5},
    )

    result = task.run(engine)

    assert isinstance(result, TorchSimMDResult)
    assert result.executed_steps == 25
    assert result.final_energy == pytest.approx(2.5)
    assert Path(result.trajectory_path).read_text(encoding="utf-8") == "md"
    integrate_kwargs = calls["integrate"][0]
    assert integrate_kwargs["integrator"] == "integrator:nvt_langevin"
    assert integrate_kwargs["temperature"] == pytest.approx(350.0)
    assert integrate_kwargs["trajectory_reporter"]["state_frequency"] == 5
    assert integrate_kwargs["trajectory_reporter"]["filenames"] == [str(Path(result.trajectory_path))]


def test_torchsim_go_task_builds_convergence_and_cell_filter(monkeypatch, tmp_path):
    calls = install_fake_torchsim(monkeypatch)
    engine = TorchEngine(
        engine_id="torchsim",
        model_factory=f"{__name__}.build_task_model",
        model_kwargs={"stress": True},
    )
    task = TorchSimGOTask(
        atomistic_system=_make_container(),
        save_folder=str(tmp_path),
        optimizer="fire",
        convergence_kind="force",
        force_tol=0.2,
        cell_filter="frechet",
        runner_kwargs={"max_steps": 10},
    )

    result = task.run(engine)

    assert isinstance(result, TorchSimGOResult)
    assert result.final_energy == pytest.approx(-1.5)
    assert Path(result.trajectory_path).read_text(encoding="utf-8") == "go"
    assert calls["force_conv"] == [0.2]
    optimize_kwargs = calls["optimize"][0]
    assert optimize_kwargs["optimizer"] == "optimizer:fire"
    assert optimize_kwargs["convergence_fn"] == ("force", 0.2)
    assert optimize_kwargs["init_kwargs"]["cell_filter"] == "cell_filter:frechet"
    assert optimize_kwargs["max_steps"] == 10


def test_torchsim_go_task_rejects_cell_filter_without_stress(monkeypatch, tmp_path):
    install_fake_torchsim(monkeypatch)
    engine = TorchEngine(engine_id="torchsim", model_factory=f"{__name__}.build_task_model")
    task = TorchSimGOTask(
        atomistic_system=_make_container(),
        save_folder=str(tmp_path),
        cell_filter="frechet",
        force_tol=0.1,
    )

    with pytest.raises(ValueError, match="implements stress"):
        task.run(engine)


def test_torchsim_md_task_rejects_wrong_engine(tmp_path):
    task = TorchSimMDTask(atomistic_system=_make_container(), save_folder=str(tmp_path))

    with pytest.raises(TypeError, match="TorchEngine"):
        task.run(ASEEngine(engine_id="ase", calculator="ase.calculators.lj.LennardJones"))
