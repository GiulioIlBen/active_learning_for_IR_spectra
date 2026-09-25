from __future__ import annotations

import json

import numpy as np
import pytest
from ase import Atoms
from scm.moliterate import PropertyInfo
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.interfaces.in_memory import InMemoryMolData

pytest.importorskip("scm.params")
pytest.importorskip("scm.plams")
pytest.importorskip("scm.input_classes", reason="requires optional dependency scm.input_classes")

from scm.active_learning.engines import ParAMSEngine
from scm.active_learning.mlip import DataAndSplittingStrategy
from scm.active_learning.mlip.params_trainer import ParAMSTrainer
from scm.active_learning.results import MLIPTrainerResults


def _make_data_split(num_train: int, num_val: int) -> DataAndSplittingStrategy:
    props = [
        PropertyInfo(name="energy", unit="eV"),
        PropertyInfo(name="forces", unit="eV/Ang"),
    ]
    dataset = InMemoryMolData.create(available_properties=props)
    base_atoms = Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    zero_forces = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    for idx in range(num_train):
        dataset.add_system(
            ChemDataEntry(
                system=base_atoms.copy(),
                properties={"energy": -1.0 - 0.01 * idx, "forces": zero_forces},
                metadata={"dataset": "training_set"},
            )
        )

    for idx in range(num_val):
        dataset.add_system(
            ChemDataEntry(
                system=base_atoms.copy(),
                properties={"energy": -0.5 - 0.01 * idx, "forces": zero_forces},
                metadata={"dataset": "validation_set"},
            )
        )

    return DataAndSplittingStrategy(dataset=dataset)


def _assert_trainer_results(results: MLIPTrainerResults) -> None:
    assert isinstance(results, MLIPTrainerResults)
    assert isinstance(results.engine, ParAMSEngine)
    assert results.log_metrics
    assert set(results.log_metrics["dataset"]) >= {"training_set", "validation_set"}


def test_m3gnet_backend_availability_check(tmp_path):
    trainer = ParAMSTrainer.Builder.M3GNet_EF().build(datapath=str(tmp_path / "params_data_m3gnet"))
    trainer._ensure_backend_ready()


def test_params_trainer_test_backend_run(tmp_path):
    data_split = _make_data_split(num_train=2, num_val=1)

    trainer = (
        ParAMSTrainer.Builder.TEST()
        .set_committee(1)
        .build(
            datapath=str(tmp_path / "params_data_test"),
        )
    )

    results = trainer.train(data_split=data_split, engine_id="test")

    _assert_trainer_results(results)


def test_params_trainer_m3gnet_short_run(tmp_path):
    availability_error = ParAMSTrainer._get_m3gnet_backend_error("UniversalPotential")
    if availability_error is not None:
        pytest.skip(availability_error)

    data_split = _make_data_split(num_train=10, num_val=10)

    trainer = (
        ParAMSTrainer.Builder.M3GNet_EF()
        .set_committee(1)
        .set_max_epochs(2)
        .build(datapath=str(tmp_path / "params_data_m3gnet"))
    )

    results = trainer.train(data_split=data_split, engine_id="m3gnet")

    _assert_trainer_results(results)


def test_params_trainer_m3gnet_preflight_surfaces_missing_model(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ParAMSTrainer,
        "_get_m3gnet_backend_error",
        classmethod(lambda cls, model: "missing bundled M3GNet model"),
    )

    trainer = ParAMSTrainer.Builder.M3GNet_EF().build(datapath=str(tmp_path / "params_data_m3gnet"))

    with pytest.raises(FileNotFoundError, match="missing bundled M3GNet model"):
        trainer._ensure_backend_ready()


def test_params_trainer_optionally_saves_settings_and_results_json(monkeypatch, tmp_path):
    trainer = ParAMSTrainer(
        datapath=str(tmp_path / "params_data"),
        source_settings={"input": {"MachineLearning": {"Backend": "TEST"}}},
        save={"settings": True, "results": True},
    )

    class FakeJob:
        def __init__(self, path):
            self.path = str(path)
            self.settings = trainer.settings

        def run(self):
            return type("FakeResults", (), {"job": self})()

    class FakeParAMSJob:
        @staticmethod
        def from_yaml(datapath, settings, name):
            del datapath, settings, name
            return FakeJob(tmp_path / "ParAMSTrainer")

    monkeypatch.setattr(ParAMSTrainer, "_ensure_backend_ready", lambda self: None)
    monkeypatch.setattr(ParAMSTrainer, "to_data_format", lambda self, data_split: tmp_path / "params_data")
    monkeypatch.setattr(ParAMSTrainer, "_load_params_deps", staticmethod(lambda: (FakeParAMSJob, None, None)))
    monkeypatch.setattr(
        ParAMSTrainer,
        "_build_results",
        lambda self, params_results, engine_id: MLIPTrainerResults(
            engine=ParAMSEngine(
                engine_id=engine_id,
                source_settings=[{"input": {"AMS": {}}}],
                job_path=params_results.job.path,
            ),
            log_metrics={"dataset": ["training_set"], "property": ["energy"], "mae": [0.1]},
            train_infos={"n_epochs": 2},
        ),
    )

    results = trainer.train(data_split=_make_data_split(num_train=2, num_val=1), engine_id="params00")

    run_dir = (tmp_path / "ParAMSTrainer").resolve(strict=False)
    settings_path = run_dir / "params_trainer_settings.json"
    results_path = run_dir / "mlip_trainer_results.json"

    settings_payload = json.loads(settings_path.read_text(encoding="utf-8"))
    results_payload = json.loads(results_path.read_text(encoding="utf-8"))

    assert settings_path.is_file()
    assert results_path.is_file()
    assert settings_payload["save"]["settings"] is True
    assert settings_payload["save"]["results"] is True
    assert results_payload["engine"]["engine_id"] == "params00"
    assert results_payload["train_infos"]["settings_json_path"] == "params_trainer_settings.json"
    assert results.train_infos["settings_json_path"] == str(settings_path)

    reloaded_trainer = ParAMSTrainer.load_model_json(settings_path)
    reloaded_results = MLIPTrainerResults.load_model_json(results_path)

    assert isinstance(reloaded_trainer, ParAMSTrainer)
    assert isinstance(reloaded_results, MLIPTrainerResults)
    assert reloaded_results.engine.engine_id == "params00"
    assert reloaded_results.train_infos["settings_json_path"] == str(settings_path)
