from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from scm.moliterate import PropertyInfo

from scm.active_learning.engines import TorchEngine
from scm.active_learning.task_parallelization import SerialStrategy


class FakeTorchState:
    def __init__(self, atoms_list):
        self.atoms_list = atoms_list
        self.atom_counts = [len(atoms) for atoms in atoms_list]


class FakeTorchModel:
    implemented_properties = ("energy", "forces")

    def __init__(self, base_energy: float = 1.0, fail_forward: bool = False):
        self.base_energy = base_energy
        self.fail_forward = fail_forward
        self.compute_forces = True
        self._device = "cpu"
        self._dtype = "float32"

    def forward(self, state):
        if self.fail_forward:
            raise RuntimeError("forward failed")
        energies = np.array([self.base_energy + idx for idx in range(len(state.atom_counts))], dtype=float)
        force_rows = []
        for idx, atom_count in enumerate(state.atom_counts):
            force_rows.append(np.full((atom_count, 3), fill_value=float(idx + 1)))
        return {
            "energy": energies,
            "forces": np.concatenate(force_rows, axis=0) if force_rows else np.zeros((0, 3), dtype=float),
        }


def build_fake_model(base_energy: float = 1.0, fail_forward: bool = False):
    return FakeTorchModel(base_energy=base_energy, fail_forward=fail_forward)


def install_fake_torchsim(monkeypatch):
    fake_module = SimpleNamespace(
        io=SimpleNamespace(atoms_to_state=lambda atoms_list, device=None, dtype=None: FakeTorchState(atoms_list))
    )
    monkeypatch.setitem(sys.modules, "torch_sim", fake_module)
    return fake_module


def test_torch_engine_run_single_point_batches_results(monkeypatch):
    install_fake_torchsim(monkeypatch)
    engine = TorchEngine(engine_id="torchsim", model_factory=f"{__name__}.build_fake_model")
    dataset = [
        SimpleNamespace(chemical_system=Atoms("H", positions=[(0.0, 0.0, 0.0)])),
        SimpleNamespace(chemical_system=Atoms("He2", positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])),
    ]
    props = [
        PropertyInfo(name="energy", unit="eV"),
        PropertyInfo(name="free_energy", unit="eV"),
        PropertyInfo(name="forces", unit="eV/Ang"),
    ]

    results = list(engine.run_single_point(props, dataset, SerialStrategy()))

    assert results[0]["energy"] == pytest.approx(1.0)
    assert results[0]["free_energy"] == pytest.approx(1.0)
    assert np.allclose(results[0]["forces"], np.ones((1, 3)))
    assert results[1]["energy"] == pytest.approx(2.0)
    assert results[1]["free_energy"] == pytest.approx(2.0)
    assert np.allclose(results[1]["forces"], np.full((2, 3), 2.0))


def test_torch_engine_reports_row_conversion_failures(monkeypatch):
    install_fake_torchsim(monkeypatch)
    engine = TorchEngine(engine_id="torchsim", model_factory=f"{__name__}.build_fake_model")
    dataset = [
        SimpleNamespace(chemical_system=object()),
        SimpleNamespace(chemical_system=Atoms("H", positions=[(0.0, 0.0, 0.0)])),
    ]

    results = list(engine.run_single_point([PropertyInfo(name="energy", unit="eV")], dataset, SerialStrategy()))

    assert "FAILURE" in results[0]
    assert results[1] == {"energy": pytest.approx(1.0)}


def test_torch_engine_reports_forward_failures(monkeypatch):
    install_fake_torchsim(monkeypatch)
    engine = TorchEngine(
        engine_id="torchsim",
        model_factory=f"{__name__}.build_fake_model",
        model_kwargs={"fail_forward": True},
    )
    dataset = [SimpleNamespace(chemical_system=Atoms("H", positions=[(0.0, 0.0, 0.0)]))]

    results = list(engine.run_single_point([PropertyInfo(name="energy", unit="eV")], dataset, SerialStrategy()))

    assert results == [{"FAILURE": "forward failed"}]


def test_torch_engine_requires_optional_dependency(monkeypatch):
    monkeypatch.setattr(
        "scm.active_learning.engines.torchsim._require_torchsim",
        lambda: (_ for _ in ()).throw(
            ImportError(
                "TorchEngine requires the optional dependency `torch-sim-atomistic` (`pip install -e .[torchsim]`)."
            )
        ),
    )
    engine = TorchEngine(engine_id="torchsim", model_factory=f"{__name__}.build_fake_model")
    dataset = [SimpleNamespace(chemical_system=Atoms("H", positions=[(0.0, 0.0, 0.0)]))]

    with pytest.raises(ImportError, match="torch-sim-atomistic"):
        list(engine.run_single_point([PropertyInfo(name="energy", unit="eV")], dataset, SerialStrategy()))
