from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from pydantic import ValidationError
from scm.moliterate import PropertyInfo
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.interfaces.in_memory import InMemoryMolData

from scm.active_learning.engines import ASEEngine
from scm.active_learning.mlip import DataAndSplittingStrategy
from scm.active_learning.mlip.mace_trainer import runner as mace_runner_module
from scm.active_learning.mlip.mace_trainer import trainer as mace_trainer_module
from scm.active_learning.mlip.mace_trainer.metrics import MACETrainingMetrics
from scm.active_learning.mlip.mace_trainer.settings import MACECommitteeSettings
from scm.active_learning.mlip.mace_trainer.trainer import MACEModelSettings, MACETrainer, MACETrainSettings
from scm.active_learning.results import MLIPTrainerResults


def _make_split(
    rows: list[tuple[str, dict[str, float | np.ndarray], dict[str, str]]],
    available_properties: list[PropertyInfo] | None = None,
) -> DataAndSplittingStrategy:
    props = available_properties or [
        PropertyInfo(name="energy", unit="eV"),
        PropertyInfo(name="forces", unit="eV/Ang"),
    ]
    dataset = InMemoryMolData.create(available_properties=props)
    for symbols, properties, metadata in rows:
        natoms = len(Atoms(symbols))
        positions = [[0.0, 0.0, 0.74 * idx] for idx in range(natoms)]
        dataset.add_system(
            ChemDataEntry(
                system=Atoms(symbols, positions=positions),
                properties=properties,
                metadata=metadata,
            )
        )
    return DataAndSplittingStrategy(dataset=dataset)


def _make_data_split() -> DataAndSplittingStrategy:
    return _make_split(
        [
            (
                "H2",
                {"energy": -1.0, "forces": np.zeros((2, 3))},
                {"dataset": "training_set"},
            ),
            (
                "He",
                {"energy": -0.1, "forces": np.zeros((1, 3))},
                {"dataset": "validation_set"},
            ),
        ]
    )


def _make_dipole_data_split(property_name: str = "dipole_moment") -> DataAndSplittingStrategy:
    return _make_split(
        [
            (
                "H2",
                {
                    "energy": -1.0,
                    "forces": np.zeros((2, 3)),
                    property_name: np.array([0.0, 0.0, 0.5]),
                },
                {"dataset": "training_set"},
            ),
            (
                "He",
                {
                    "energy": -0.1,
                    "forces": np.zeros((1, 3)),
                    property_name: np.array([0.0, 0.0, 0.0]),
                },
                {"dataset": "validation_set"},
            ),
        ],
        available_properties=[
            PropertyInfo(name="energy", unit="eV"),
            PropertyInfo(name="forces", unit="eV/Ang"),
            PropertyInfo(name=property_name, unit="Debye"),
        ],
    )


def _make_training_only_split(n_rows: int, symbols: str = "H2") -> DataAndSplittingStrategy:
    rows = []
    natoms = len(Atoms(symbols))
    for idx in range(n_rows):
        rows.append(
            (
                symbols,
                {
                    "energy": -1.0 - 0.1 * idx,
                    "forces": np.zeros((natoms, 3)),
                },
                {"dataset": "training_set"},
            )
        )
    return _make_split(rows)


def _prepare_context(trainer: MACETrainer, tmp_path, data_split: DataAndSplittingStrategy | None = None):
    run_dir = tmp_path / "run"
    return trainer.data.prepare_training_context(data_split=data_split or _make_data_split(), run_dir=run_dir)


def _export_data_format(trainer: MACETrainer, tmp_path, data_split: DataAndSplittingStrategy | None = None) -> Path:
    run_dir = tmp_path / "run"
    return trainer.to_data_format(data_split=data_split or _make_data_split(), run_dir=run_dir)


def _build_config(trainer: MACETrainer, tmp_path, data_split: DataAndSplittingStrategy | None = None) -> dict:
    context = _prepare_context(trainer, tmp_path, data_split=data_split)
    return trainer._build_config(
        context=context,
        model_dir=context.run_dir / trainer.data.model_subfolder,
        base_model_file=None,
    )


def test_mace_trainer_builds_zero_e0s_from_dataset(tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path})

    config = _build_config(trainer, tmp_path)

    assert config["E0s"] == "{1: 0.0, 2: 0.0}"
    assert config["compute_avg_num_neighbors"] is False
    assert config["avg_num_neighbors"] == 1.0
    assert config["energy_key"] == "REF_energy"
    assert config["forces_key"] == "REF_forces"


def test_mace_trainer_exports_safe_ref_property_keys(tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path})
    _export_data_format(trainer, tmp_path)
    context = _prepare_context(trainer, tmp_path)

    train_contents = context.train_file.read_text(encoding="utf-8")
    valid_contents = context.valid_file.read_text(encoding="utf-8")

    assert "REF_energy=-1.0" in train_contents
    assert "REF_forces:R:3" in train_contents
    assert " energy=" not in train_contents
    assert " forces:R:3" not in train_contents
    assert "REF_energy=-0.1" in valid_contents


@pytest.mark.parametrize("property_name", ["dipole", "dipole_moment"])
def test_mace_trainer_round_trips_dipoles_through_extxyz(tmp_path, property_name):
    trainer = MACETrainer(data={"folder": tmp_path})
    data_split = _make_dipole_data_split(property_name=property_name)
    _export_data_format(trainer, tmp_path, data_split=data_split)
    context = _prepare_context(trainer, tmp_path, data_split=data_split)

    train_atoms = read(context.train_file, format="extxyz")
    valid_atoms = read(context.valid_file, format="extxyz")

    np.testing.assert_allclose(train_atoms.info["REF_dipole"], np.array([0.0, 0.0, 0.5]))
    np.testing.assert_allclose(valid_atoms.info["REF_dipole"], np.array([0.0, 0.0, 0.0]))


@pytest.mark.parametrize("property_name", ["dipole", "dipole_moment"])
def test_mace_trainer_uses_ref_dipole_key_for_dipole_aliases(tmp_path, property_name):
    trainer = MACETrainer(
        data={"folder": tmp_path},
        train_settings={
            "architecture": {
                "model": "EnergyDipolesMACE",
                "loss": "energy_forces_dipole",
                "error_table": "EnergyDipoleRMSE",
            },
        },
    )

    config = _build_config(trainer, tmp_path, data_split=_make_dipole_data_split(property_name=property_name))

    assert config["dipole_key"] == "REF_dipole"


def test_mace_trainer_routes_all_side_outputs_into_run_dir(tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path})

    config = _build_config(trainer, tmp_path)
    run_dir = (tmp_path / "run").resolve(strict=False)

    assert config["log_dir"] == str((run_dir / "logs").resolve(strict=False))
    assert config["results_dir"] == str((run_dir / "results").resolve(strict=False))
    assert config["checkpoints_dir"] == str((run_dir / "checkpoints").resolve(strict=False))
    assert config["downloads_dir"] == str((run_dir / "downloads").resolve(strict=False))


def test_train_mace_runs_in_requested_workdir(monkeypatch, tmp_path):
    seen: dict[str, Path] = {}
    config_path = tmp_path / "config.yaml"
    workdir = tmp_path / "run"
    config_path.write_text("name: test\n", encoding="utf-8")
    workdir.mkdir()

    def fake_main():
        seen["cwd"] = Path.cwd()

    monkeypatch.setattr(mace_runner_module, "_load_mace_train_main", lambda: fake_main)
    previous_cwd = Path.cwd()

    mace_trainer_module.train_mace(config_file_path=config_path, workdir=workdir)

    assert seen["cwd"] == workdir.resolve(strict=False)
    assert Path.cwd() == previous_cwd


def test_mace_trainer_allows_explicit_e0s_override(tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path}, extra_config={"E0s": "average"})

    config = _build_config(trainer, tmp_path)

    assert config["E0s"] == "average"


def test_mace_trainer_falls_back_to_one_avg_num_neighbors_for_isolated_atoms(tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path})

    config = _build_config(trainer, tmp_path, data_split=_make_training_only_split(n_rows=1, symbols="He"))

    assert config["compute_avg_num_neighbors"] is False
    assert config["avg_num_neighbors"] == 1.0


def test_mace_trainer_clamps_batch_sizes_to_explicit_dataset_sizes(tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path}, train_settings={"batch_size": 10, "valid_batch_size": 10})

    config = _build_config(trainer, tmp_path)

    assert config["batch_size"] == 1
    assert config["valid_batch_size"] == 1


def test_mace_trainer_clamps_batch_size_to_mace_internal_train_split(tmp_path):
    trainer = MACETrainer(
        data={"folder": tmp_path},
        train_settings={"batch_size": 10, "valid_batch_size": 10, "valid_fraction": 0.1},
    )

    config = _build_config(trainer, tmp_path, data_split=_make_training_only_split(n_rows=10))

    assert config["batch_size"] == 9
    assert config["valid_batch_size"] == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("device", "gpu"),
        ("log_level", "QUIET"),
        ("default_dtype", "float16"),
    ],
)
def test_mace_train_settings_reject_invalid_literals(field, value):
    with pytest.raises(ValidationError):
        MACETrainSettings(**{field: value})


def test_mace_train_settings_patches_load_foundations_by_default():
    assert MACETrainSettings().patch_load_foundations is True


def test_mace_committee_settings_generates_member_seeds():
    assert MACECommitteeSettings(size=1).member_seeds() == [None]
    assert MACECommitteeSettings(size=3).member_seeds() == [123, 124, 125]
    assert MACECommitteeSettings(size=3, seed=10, seed_stride=5).member_seeds() == [10, 15, 20]


def test_mace_committee_settings_rejects_invalid_member_index():
    with pytest.raises(IndexError):
        MACECommitteeSettings(size=2).member_seed(2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_num_epochs", 0),
        ("batch_size", 0),
        ("valid_batch_size", 0),
        ("eval_interval", 0),
        ("patience", 0),
        ("learning_rate", 0.0),
    ],
)
def test_mace_train_settings_reject_non_positive_numbers(field, value):
    with pytest.raises(ValidationError):
        MACETrainSettings(**{field: value})


@pytest.mark.parametrize("value", [-0.1, 1.0, 1.5])
def test_mace_train_settings_reject_invalid_valid_fraction(value):
    with pytest.raises(ValidationError):
        MACETrainSettings(valid_fraction=value)


def test_mace_train_settings_reject_blank_model_name():
    with pytest.raises(ValidationError):
        MACETrainSettings(model_name="   ")


def test_mace_train_settings_defaults_are_internally_consistent():
    settings = MACETrainSettings()

    assert settings.architecture.model == "MACE"
    assert settings.architecture.loss == "weighted"
    assert settings.architecture.error_table == "PerAtomRMSE"
    assert settings.log_level == "INFO"
    assert settings.architecture.forces_weight == 100.0
    assert settings.architecture.energy_weight == 1.0
    assert settings.architecture.radial_MLP is None
    assert settings.swa is True
    assert settings.start_swa == 20
    assert settings.start_swa < settings.max_num_epochs


def test_mace_trainer_includes_log_level_in_config(tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path}, train_settings={"log_level": "ERROR"})

    config = _build_config(trainer, tmp_path)

    assert config["log_level"] == "ERROR"


def test_mace_trainer_includes_explicit_seed_in_config(tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path})
    context = _prepare_context(trainer, tmp_path)

    config = trainer._build_config(
        context=context,
        model_dir=context.run_dir / trainer.data.model_subfolder,
        base_model_file=None,
        seed=99,
    )

    assert config["seed"] == 99


def test_mace_train_settings_stores_nested_architecture_settings():
    settings = MACETrainSettings(
        architecture={
            "model": "EnergyDipolesMACE",
            "loss": "energy_forces_dipole",
            "error_table": "EnergyDipoleRMSE",
            "forces_weight": 250.0,
            "energy_weight": 2.5,
            "hidden_irreps": "256x0e + 256x1o",
        }
    )

    assert settings.architecture.model == "EnergyDipolesMACE"
    assert settings.architecture.loss == "energy_forces_dipole"
    assert settings.architecture.error_table == "EnergyDipoleRMSE"
    assert settings.architecture.forces_weight == 250.0
    assert settings.architecture.energy_weight == 2.5
    assert settings.architecture.hidden_irreps == "256x0e + 256x1o"


def test_mace_model_settings_uses_energy_dipole_error_table_by_default():
    settings = MACEModelSettings(model="EnergyDipolesMACE", loss="energy_forces_dipole")

    assert settings.error_table == "EnergyDipoleRMSE"


def test_mace_model_settings_rejects_energy_dipole_model_with_wrong_error_table():
    with pytest.raises(ValidationError, match="EnergyDipoleRMSE"):
        MACEModelSettings(
            model="EnergyDipolesMACE",
            loss="energy_forces_dipole",
            error_table="PerAtomRMSE",
        )


def test_mace_model_settings_reject_energy_dipole_model_with_wrong_loss():
    with pytest.raises(ValidationError, match="energy_forces_dipole"):
        MACEModelSettings(model="EnergyDipolesMACE", loss="weighted")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("error_table", "BadTable"),
        ("loss", "bad_loss"),
        ("model", "DipoleMACE"),
    ],
)
def test_mace_model_settings_reject_invalid_literals(field, value):
    with pytest.raises(ValidationError):
        MACEModelSettings(**{field: value})


@pytest.mark.parametrize(("field", "value"), [("forces_weight", 0.0), ("energy_weight", 0.0)])
def test_mace_model_settings_reject_non_positive_weights(field, value):
    with pytest.raises(ValidationError):
        MACEModelSettings(**{field: value})


def test_mace_train_settings_reject_start_swa_at_or_after_max_epochs_when_enabled():
    with pytest.raises(ValidationError, match="start_swa must be smaller than max_num_epochs"):
        MACETrainSettings(swa=True, start_swa=30, max_num_epochs=30)


def test_mace_train_settings_allow_large_start_swa_when_swa_disabled():
    settings = MACETrainSettings(swa=False, start_swa=30, max_num_epochs=30)

    assert settings.start_swa == 30


def test_mace_trainer_adds_swa_config_when_enabled(tmp_path):
    trainer = MACETrainer(
        data={"folder": tmp_path},
        train_settings={"swa": True, "start_swa": 5, "swa_energy_weight": 750.0},
    )

    config = _build_config(trainer, tmp_path)

    assert config["swa"] is True
    assert config["start_swa"] == 5
    assert config["swa_energy_weight"] == 750.0


def test_mace_trainer_omits_swa_config_when_disabled(tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path}, train_settings={"swa": False})

    config = _build_config(trainer, tmp_path)

    assert "swa" not in config
    assert "start_swa" not in config
    assert "swa_energy_weight" not in config


def test_mace_trainer_does_not_retain_dataset_state_between_runs(tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path})
    first_run = tmp_path / "run_one"
    second_run = tmp_path / "run_two"
    first_run.mkdir()
    second_run.mkdir()

    first_context = trainer.data.prepare_training_context(
        data_split=_make_training_only_split(n_rows=1, symbols="He"),
        run_dir=first_run,
    )
    second_context = trainer.data.prepare_training_context(
        data_split=_make_training_only_split(n_rows=1, symbols="Li"),
        run_dir=second_run,
    )

    first_config = trainer._build_config(
        context=first_context,
        model_dir=first_context.run_dir / trainer.data.model_subfolder,
        base_model_file=None,
    )
    second_config = trainer._build_config(
        context=second_context,
        model_dir=second_context.run_dir / trainer.data.model_subfolder,
        base_model_file=None,
    )

    assert first_config["E0s"] == "{2: 0.0}"
    assert second_config["E0s"] == "{3: 0.0}"
    assert "valid_file" not in second_config


def test_mace_trainer_to_data_format_returns_dataset_dir_and_writes_files(tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path})
    dataset_dir = trainer.to_data_format(data_split=_make_data_split())
    assert dataset_dir == (tmp_path / "datasets").resolve(strict=False)
    assert (dataset_dir / "training_set.xyz").is_file()
    assert (dataset_dir / "validation_set.xyz").is_file()


def test_prepare_training_context_does_not_write_dataset_files(tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path})
    context = _prepare_context(trainer, tmp_path)
    assert context.dataset_dir == tmp_path / "run" / "datasets"
    assert not context.train_file.exists()
    assert not context.valid_file.exists()


def test_mace_trainer_returns_finetunable_ase_engine(monkeypatch, tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path})

    monkeypatch.setattr(
        mace_trainer_module,
        "train_mace",
        lambda config_file_path, workdir=None, patched_load_foundations=None: None,
    )
    monkeypatch.setattr(MACETrainer, "_write_yaml", lambda self, path, payload: None)
    monkeypatch.setattr(
        MACETrainer,
        "_resolve_model_path",
        lambda self, model_dir: str((Path(model_dir) / "model_compiled.model").resolve(strict=False)),
    )

    result = trainer.train(data_split=_make_data_split(), engine_id="MACE00")

    assert result.engine.is_finetunable() is True
    assert result.engine.calculator_kwargs["model_paths"].endswith("model_compiled.model")


def test_mace_trainer_trains_committee_members_with_distinct_seeds(monkeypatch, tmp_path):
    trainer = MACETrainer(
        data={"folder": tmp_path},
        train_settings={"committee": {"size": 3, "seed": 10}},
    )
    captured_configs: dict[Path, dict] = {}
    train_calls: list[tuple[Path, Path]] = []

    def capture_yaml(self, path, payload):
        del self
        captured_configs[Path(path)] = payload

    def fake_train_mace(config_file_path, workdir=None, patched_load_foundations=None):
        del patched_load_foundations
        train_calls.append((Path(config_file_path), Path(workdir)))

    monkeypatch.setattr(mace_trainer_module, "train_mace", fake_train_mace)
    monkeypatch.setattr(MACETrainer, "_write_yaml", capture_yaml)
    monkeypatch.setattr(
        MACETrainer,
        "_resolve_model_path",
        lambda self, model_dir: str((Path(model_dir) / "model.model").resolve(strict=False)),
    )

    result = trainer.train(data_split=_make_data_split(), engine_id="MACE00")

    assert len(train_calls) == 3
    assert [captured_configs[config_path]["seed"] for config_path, _ in train_calls] == [10, 11, 12]
    assert [workdir.name for _, workdir in train_calls] == ["member_000", "member_001", "member_002"]
    assert result.engine.calculator_kwargs["model_paths"] == [
        str((tmp_path / "MACE00" / f"member_{index:03d}" / "results" / "model.model").resolve(strict=False))
        for index in range(3)
    ]
    assert result.train_infos["committee_size"] == 3
    assert result.train_infos["member_0_run_directory"].endswith("member_000")
    assert result.train_infos["run_directory"] == str((tmp_path / "MACE00").resolve(strict=False))


def test_mace_trainer_optionally_saves_settings_and_results_json(monkeypatch, tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path}, save={"settings": True, "results": True})

    monkeypatch.setattr(
        mace_trainer_module,
        "train_mace",
        lambda config_file_path, workdir=None, patched_load_foundations=None: None,
    )
    monkeypatch.setattr(
        MACETrainer,
        "_resolve_model_path",
        lambda self, model_dir: str((Path(model_dir) / "model_compiled.model").resolve(strict=False)),
    )
    monkeypatch.setattr(MACETrainer, "_write_yaml", lambda self, path, payload: None)

    result = trainer.train(data_split=_make_data_split(), engine_id="MACE00")

    run_dir = (tmp_path / "MACE00").resolve(strict=False)
    settings_path = run_dir / "mace_trainer_settings.json"
    results_path = run_dir / "mlip_trainer_results.json"

    settings_payload = json.loads(settings_path.read_text(encoding="utf-8"))
    results_payload = json.loads(results_path.read_text(encoding="utf-8"))

    assert settings_path.is_file()
    assert results_path.is_file()
    assert settings_payload["save"]["settings"] is True
    assert settings_payload["save"]["results"] is True
    assert results_payload["engine"]["engine_id"] == "MACE00"
    assert results_payload["train_infos"]["run_directory"] == "."
    assert results_payload["train_infos"]["settings_json_path"] == "mace_trainer_settings.json"
    assert result.train_infos["settings_json_path"] == str(settings_path)

    reloaded_trainer = MACETrainer.load_model_json(settings_path)
    reloaded_results = MLIPTrainerResults.load_model_json(results_path)

    assert isinstance(reloaded_trainer, MACETrainer)
    assert reloaded_trainer.data.folder == tmp_path.resolve()
    assert isinstance(reloaded_results, MLIPTrainerResults)
    assert reloaded_results.engine.engine_id == "MACE00"
    assert reloaded_results.train_infos["run_directory"] == str(run_dir)
    assert reloaded_results.train_infos["settings_json_path"] == str(settings_path)


def test_mace_trainer_returns_dipole_aware_calculator_for_dipole_models(monkeypatch, tmp_path):
    trainer = MACETrainer(
        data={"folder": tmp_path},
        train_settings={
            "architecture": {
                "model": "EnergyDipolesMACE",
                "loss": "energy_forces_dipole",
                "error_table": "EnergyDipoleRMSE",
            },
        },
    )

    monkeypatch.setattr(
        mace_trainer_module,
        "train_mace",
        lambda config_file_path, workdir=None, patched_load_foundations=None: None,
    )
    monkeypatch.setattr(MACETrainer, "_write_yaml", lambda self, path, payload: None)
    monkeypatch.setattr(
        MACETrainer,
        "_resolve_model_path",
        lambda self, model_dir: str((Path(model_dir) / "model_compiled.model").resolve(strict=False)),
    )

    result = trainer.train(data_split=_make_dipole_data_split(), engine_id="MACE00")

    assert result.engine.calculator == "scm.active_learning.mlip.mace_trainer.calculator.AMSMACECalculator"
    assert result.engine.calculator_kwargs["model_type"] == "EnergyDipoleMACE"
    assert result.engine.calculator_kwargs["dipole_units"] == "Debye"


def test_mace_trainer_calculator_kwargs_initialize_ams_mace_calculator_with_keywords(monkeypatch, tmp_path):
    pytest.importorskip("mace")
    pytest.importorskip("scm.amspipe")
    from scm.active_learning.mlip.mace_trainer.calculator import AMSMACECalculator

    trainer = MACETrainer(data={"folder": tmp_path})
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "scm.active_learning.mlip.mace_trainer.calculator.MACECalculator.__init__",
        lambda self, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        "scm.active_learning.mlip.mace_trainer.calculator.apply_ams_external_capabilities",
        lambda calculator, props: None,
    )
    monkeypatch.setattr(AMSMACECalculator, "implemented_properties", [])

    kwargs = trainer._build_calculator_kwargs(
        model_path=str((tmp_path / "model_compiled.model").resolve(strict=False)),
        data_split=_make_data_split(),
    )
    calculator = AMSMACECalculator(**kwargs)

    assert captured["model_paths"] == kwargs["model_paths"]
    assert captured["device"] == kwargs["device"]
    assert captured["default_dtype"] == kwargs["default_dtype"]
    assert captured["model_type"] == "MACE"
    assert "models" not in captured
    assert calculator.dipole_units == "Debye"


def test_mace_trainer_records_architecture_metadata_on_returned_engine(monkeypatch, tmp_path):
    trainer = MACETrainer(
        data={"folder": tmp_path},
        train_settings={
            "architecture": {
                "model": "EnergyDipolesMACE",
                "loss": "energy_forces_dipole",
                "error_table": "EnergyDipoleRMSE",
            },
        },
    )

    monkeypatch.setattr(
        mace_trainer_module,
        "train_mace",
        lambda config_file_path, workdir=None, patched_load_foundations=None: None,
    )
    monkeypatch.setattr(MACETrainer, "_write_yaml", lambda self, path, payload: None)
    monkeypatch.setattr(
        MACETrainer,
        "_resolve_model_path",
        lambda self, model_dir: str((Path(model_dir) / "model_compiled.model").resolve(strict=False)),
    )
    result = trainer.train(data_split=_make_dipole_data_split(), engine_id="MACE00")
    assert result.engine.calculator_kwargs["mace_model_architecture"] == {
        "model": "EnergyDipolesMACE",
        "hidden_irreps": "128x0e + 128x1o",
        "max_ell": 3,
        "correlation": 3,
        "num_interactions": 2,
        "MLP_irreps": "16x0e",
        "gate": "silu",
    }


def test_mace_trainer_deletes_checkpoints_after_training(monkeypatch, tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path})

    def fake_train_mace(config_file_path, workdir=None, patched_load_foundations=None):
        del config_file_path
        del patched_load_foundations
        checkpoints_dir = Path(workdir) / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        (checkpoints_dir / "epoch-1.pt").write_text("checkpoint", encoding="utf-8")
        (checkpoints_dir / "epoch-3.pt").write_text("checkpoint", encoding="utf-8")

    monkeypatch.setattr(mace_trainer_module, "train_mace", fake_train_mace)
    monkeypatch.setattr(MACETrainer, "_write_yaml", lambda self, path, payload: None)
    monkeypatch.setattr(
        MACETrainer,
        "_resolve_model_path",
        lambda self, model_dir: str((Path(model_dir) / "model.model").resolve(strict=False)),
    )

    result = trainer.train(data_split=_make_data_split(), engine_id="MACE00")

    run_dir = Path(result.train_infos["run_directory"])
    assert not (run_dir / "checkpoints").exists()
    assert result.train_infos["actual_num_epochs"] == 3


def test_mace_trainer_omits_actual_num_epochs_when_no_checkpoints_exist(monkeypatch, tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path})

    monkeypatch.setattr(
        mace_trainer_module,
        "train_mace",
        lambda config_file_path, workdir=None, patched_load_foundations=None: None,
    )
    monkeypatch.setattr(MACETrainer, "_write_yaml", lambda self, path, payload: None)
    monkeypatch.setattr(
        MACETrainer,
        "_resolve_model_path",
        lambda self, model_dir: str((Path(model_dir) / "model.model").resolve(strict=False)),
    )

    result = trainer.train(data_split=_make_data_split(), engine_id="MACE00")

    assert "actual_num_epochs" not in result.train_infos


def test_mace_trainer_prefers_plain_model_for_finetuning(tmp_path):
    model_dir = tmp_path / "results"
    model_dir.mkdir()
    compiled_model = model_dir / "MACE_swa_compiled.model"
    compiled_model.write_text("compiled", encoding="utf-8")
    plain_model = model_dir / "MACE_swa.model"
    plain_model.write_text("plain", encoding="utf-8")

    model_path = MACETrainer._resolve_model_path(model_dir)

    assert model_path == str(plain_model.resolve(strict=False))


def test_mace_trainer_reuses_engine_architecture_metadata_for_finetuning(monkeypatch, tmp_path):
    trainer = MACETrainer(
        data={"folder": tmp_path},
        train_settings={
            "architecture": {
                "model": "EnergyDipolesMACE",
                "loss": "energy_forces_dipole",
                "error_table": "EnergyDipoleRMSE",
            },
        },
    )
    captured: dict[str, dict] = {}

    monkeypatch.setattr(
        mace_trainer_module,
        "train_mace",
        lambda config_file_path, workdir=None, patched_load_foundations=None: None,
    )
    monkeypatch.setattr(
        MACETrainer,
        "_resolve_model_path",
        lambda self, model_dir: str(Path(model_dir) / "model.model"),
    )

    def capture_yaml(self, path, payload):
        del self, path
        captured["config"] = payload

    monkeypatch.setattr(MACETrainer, "_write_yaml", capture_yaml)
    start_engine = ASEEngine(
        engine_id="seed",
        calculator="scm.active_learning.mlip.mace_trainer.calculator.AMSMACECalculator",
        calculator_kwargs={
            "model_paths": "/tmp/foundation.model",
            "model_type": "EnergyDipoleMACE",
            "mace_model_architecture": {
                "hidden_irreps": "256x0e + 256x1o",
                "max_ell": 2,
                "correlation": 4,
                "num_interactions": 3,
                "MLP_irreps": "32x0e",
                "gate": "tanh",
            },
        },
        finetunable=True,
    )
    trainer.finetune(engine=start_engine, data_split=_make_dipole_data_split(), engine_id="MACE01")
    assert captured["config"]["foundation_model"] == str(Path("/tmp/foundation.model").resolve(strict=False))
    assert "patch_load_foundations" not in captured["config"]
    assert captured["config"]["hidden_irreps"] == "256x0e + 256x1o"
    assert captured["config"]["max_ell"] == 2
    assert captured["config"]["correlation"] == 4
    assert captured["config"]["num_interactions"] == 3
    assert captured["config"]["MLP_irreps"] == "32x0e"
    assert captured["config"]["gate"] == "tanh"


def test_mace_trainer_finetunes_committee_members_from_matching_model_paths(monkeypatch, tmp_path):
    trainer = MACETrainer(
        data={"folder": tmp_path},
        train_settings={"committee": {"size": 2, "seed": 30}},
    )
    captured_configs: dict[Path, dict] = {}
    train_calls: list[Path] = []

    def capture_yaml(self, path, payload):
        del self
        captured_configs[Path(path)] = payload

    def fake_train_mace(config_file_path, workdir=None, patched_load_foundations=None):
        del workdir, patched_load_foundations
        train_calls.append(Path(config_file_path))

    monkeypatch.setattr(mace_trainer_module, "train_mace", fake_train_mace)
    monkeypatch.setattr(MACETrainer, "_write_yaml", capture_yaml)
    monkeypatch.setattr(
        MACETrainer,
        "_resolve_model_path",
        lambda self, model_dir: str((Path(model_dir) / "model.model").resolve(strict=False)),
    )

    trainer.finetune(
        engine=ASEEngine(
            engine_id="seed",
            calculator="scm.active_learning.mlip.mace_trainer.calculator.AMSMACECalculator",
            calculator_kwargs={"model_paths": ["/tmp/foundation-0.model", "/tmp/foundation-1.model"]},
            finetunable=True,
        ),
        data_split=_make_data_split(),
        engine_id="MACE01",
    )

    assert [captured_configs[path]["foundation_model"] for path in train_calls] == [
        str(Path("/tmp/foundation-0.model").resolve(strict=False)),
        str(Path("/tmp/foundation-1.model").resolve(strict=False)),
    ]
    assert [captured_configs[path]["seed"] for path in train_calls] == [30, 31]


def test_mace_trainer_rejects_mismatched_finetune_committee_model_paths(tmp_path):
    trainer = MACETrainer(
        data={"folder": tmp_path},
        train_settings={"committee": {"size": 3, "seed": 30}},
    )
    start_engine = ASEEngine(
        engine_id="seed",
        calculator="scm.active_learning.mlip.mace_trainer.calculator.AMSMACECalculator",
        calculator_kwargs={"model_paths": ["/tmp/foundation-0.model", "/tmp/foundation-1.model"]},
        finetunable=True,
    )

    with pytest.raises(ValueError, match="Cannot fine-tune MACE committee"):
        trainer.finetune(engine=start_engine, data_split=_make_data_split(), engine_id="MACE01")


def test_mace_trainer_temporarily_patches_load_foundations_during_finetune(monkeypatch, tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path})
    observed: dict[str, object] = {}

    def original_load_foundations_elements(*args, **kwargs):
        del args, kwargs
        return None

    fake_run_train_module = SimpleNamespace()
    fake_model_script_utils_module = SimpleNamespace(load_foundations_elements=original_load_foundations_elements)

    def fake_main():
        observed["during"] = fake_model_script_utils_module.load_foundations_elements

    monkeypatch.setattr(mace_runner_module, "_load_mace_train_module", lambda: fake_run_train_module)
    monkeypatch.setattr(
        mace_runner_module,
        "_load_mace_model_script_utils_module",
        lambda: fake_model_script_utils_module,
    )
    monkeypatch.setattr(mace_runner_module, "_load_mace_train_main", lambda: fake_main)
    monkeypatch.setattr(MACETrainer, "_write_yaml", lambda self, path, payload: None)
    monkeypatch.setattr(
        MACETrainer,
        "_resolve_model_path",
        lambda self, model_dir: str((Path(model_dir) / "model.model").resolve(strict=False)),
    )

    trainer.finetune(
        engine=ASEEngine(
            engine_id="seed",
            calculator="scm.active_learning.mlip.mace_trainer.calculator.AMSMACECalculator",
            calculator_kwargs={"model_paths": "/tmp/foundation.model"},
            finetunable=True,
        ),
        data_split=_make_data_split(),
        engine_id="MACE01",
    )

    assert observed["during"] is mace_runner_module._patched_mace_load_foundations
    assert fake_model_script_utils_module.load_foundations_elements is original_load_foundations_elements


def test_patched_mace_load_foundations_accepts_default_dtype_without_passing_it_to_legacy_loader(monkeypatch):
    captured: dict[str, object] = {}
    sentinel_dtype = object()
    expected_result = object()

    def original_load_foundations(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return expected_result

    monkeypatch.setattr(
        mace_runner_module,
        "_load_mace_finetuning_utils_module",
        lambda: SimpleNamespace(load_foundations=original_load_foundations),
    )
    model = SimpleNamespace(r_max=5.0)
    foundation_model = SimpleNamespace(r_max=5.0, atomic_numbers=[1, 8])
    table = SimpleNamespace(zs=[1, 6])

    result = mace_runner_module._patched_mace_load_foundations(
        model,
        foundation_model,
        table,
        default_dtype=sentinel_dtype,
    )

    assert result is expected_result
    assert captured["args"] == (model, foundation_model, table)
    assert "default_dtype" not in captured["kwargs"]


def test_mace_trainer_temporarily_patches_legacy_run_train_load_foundations(monkeypatch, tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path})
    observed: dict[str, object] = {}

    def original_load_foundations(*args, **kwargs):
        del args, kwargs
        return None

    fake_run_train_module = SimpleNamespace(load_foundations=original_load_foundations)
    fake_model_script_utils_module = SimpleNamespace()

    def fake_main():
        observed["during"] = fake_run_train_module.load_foundations

    monkeypatch.setattr(mace_runner_module, "_load_mace_train_module", lambda: fake_run_train_module)
    monkeypatch.setattr(
        mace_runner_module,
        "_load_mace_model_script_utils_module",
        lambda: fake_model_script_utils_module,
    )
    monkeypatch.setattr(mace_runner_module, "_load_mace_train_main", lambda: fake_main)
    monkeypatch.setattr(MACETrainer, "_write_yaml", lambda self, path, payload: None)
    monkeypatch.setattr(
        MACETrainer,
        "_resolve_model_path",
        lambda self, model_dir: str((Path(model_dir) / "model.model").resolve(strict=False)),
    )

    trainer.finetune(
        engine=ASEEngine(
            engine_id="seed",
            calculator="scm.active_learning.mlip.mace_trainer.calculator.AMSMACECalculator",
            calculator_kwargs={"model_paths": "/tmp/foundation.model"},
            finetunable=True,
        ),
        data_split=_make_data_split(),
        engine_id="MACE01",
    )

    assert observed["during"] is mace_runner_module._patched_mace_load_foundations
    assert fake_run_train_module.load_foundations is original_load_foundations


def test_mace_trainer_can_disable_load_foundations_patch(monkeypatch, tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path}, train_settings={"patch_load_foundations": False})
    observed: dict[str, object] = {}

    def original_load_foundations_elements(*args, **kwargs):
        del args, kwargs
        return None

    fake_run_train_module = SimpleNamespace()
    fake_model_script_utils_module = SimpleNamespace(load_foundations_elements=original_load_foundations_elements)

    def fake_main():
        observed["during"] = fake_model_script_utils_module.load_foundations_elements

    monkeypatch.setattr(mace_runner_module, "_load_mace_train_module", lambda: fake_run_train_module)
    monkeypatch.setattr(
        mace_runner_module,
        "_load_mace_model_script_utils_module",
        lambda: fake_model_script_utils_module,
    )
    monkeypatch.setattr(mace_runner_module, "_load_mace_train_main", lambda: fake_main)
    monkeypatch.setattr(MACETrainer, "_write_yaml", lambda self, path, payload: None)
    monkeypatch.setattr(
        MACETrainer,
        "_resolve_model_path",
        lambda self, model_dir: str((Path(model_dir) / "model.model").resolve(strict=False)),
    )

    trainer.finetune(
        engine=ASEEngine(
            engine_id="seed",
            calculator="scm.active_learning.mlip.mace_trainer.calculator.AMSMACECalculator",
            calculator_kwargs={"model_paths": "/tmp/foundation.model"},
            finetunable=True,
        ),
        data_split=_make_data_split(),
        engine_id="MACE01",
    )

    assert observed["during"] is original_load_foundations_elements
    assert fake_model_script_utils_module.load_foundations_elements is original_load_foundations_elements


def test_extract_model_from_ase_engine_prefers_model_paths_before_checkpoint_path(tmp_path):
    model_path = (tmp_path / "model.model").resolve(strict=False)
    model_path.write_text("model", encoding="utf-8")
    checkpoint_path = (tmp_path / "model_checkpoint.pt").resolve(strict=False)
    checkpoint_path.write_text("checkpoint", encoding="utf-8")
    engine = ASEEngine(
        engine_id="MACE00",
        calculator="mace.calculators.mace.MACECalculator",
        calculator_kwargs={
            "model_paths": str(model_path),
            "checkpoint_path": str(checkpoint_path),
        },
    )

    extracted = MACETrainer._extract_model_from_ase_engine(engine)

    assert extracted == str(model_path)


def test_mace_trainer_requires_dipoles_for_dipole_loss(tmp_path):
    trainer = MACETrainer(
        data={"folder": tmp_path},
        train_settings={
            "architecture": {
                "model": "EnergyDipolesMACE",
                "loss": "energy_forces_dipole",
                "error_table": "EnergyDipoleRMSE",
            },
        },
    )

    with pytest.raises(ValueError, match="requires dipole labels"):
        _build_config(trainer, tmp_path)


def test_mace_trainer_can_keep_checkpoints_when_requested(monkeypatch, tmp_path):
    trainer = MACETrainer(
        data={"folder": tmp_path},
        train_settings={"delete_checkpoints_after_training": False},
    )

    def fake_train_mace(config_file_path, workdir=None, patched_load_foundations=None):
        del config_file_path
        del patched_load_foundations
        checkpoints_dir = Path(workdir) / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        (checkpoints_dir / "epoch-1.pt").write_text("checkpoint", encoding="utf-8")
        (checkpoints_dir / "epoch-2.pt").write_text("checkpoint", encoding="utf-8")

    monkeypatch.setattr(mace_trainer_module, "train_mace", fake_train_mace)
    monkeypatch.setattr(MACETrainer, "_write_yaml", lambda self, path, payload: None)
    monkeypatch.setattr(
        MACETrainer,
        "_resolve_model_path",
        lambda self, model_dir: str((Path(model_dir) / "model.model").resolve(strict=False)),
    )

    result = trainer.train(data_split=_make_data_split(), engine_id="MACE00")

    run_dir = Path(result.train_infos["run_directory"])
    assert (run_dir / "checkpoints").exists()
    assert result.train_infos["actual_num_epochs"] == 2


def test_mace_trainer_collects_validation_metrics_from_ascii_results_tables():
    pytest.importorskip("mace")
    metrics = MACETrainingMetrics.parse_validation_log_metrics(
        "\n".join(
            [
                "2026-04-22 16:46:08.042 INFO: Loaded Stage one model from epoch 17 for evaluation",
                "2026-04-22 16:46:10.446 INFO: Error-table on TRAIN and VALID:",
                "+---------------+---------------------+------------------+--------------+-------------------------+---------------+",
                (
                    "|  config_type  | RMSE E / meV / atom | RMSE F / meV / A | rel F RMSE % "
                    "| RMSE MU / mDebye / atom | rel MU RMSE % |"
                ),
                "+---------------+---------------------+------------------+--------------+-------------------------+---------------+",
                (
                    "| train_Default |          393.7      |        112.3     |       12.8   "
                    "|             10.1        |        65.2   |"
                ),
                (
                    "| valid_Default |          389.3      |        109.3     |       13.4   "
                    "|             12.0        |        65.7   |"
                ),
                "+---------------+---------------------+------------------+--------------+-------------------------+---------------+",
                "2026-04-22 16:46:10.541 INFO: Loaded Stage two model from epoch 43 for evaluation",
                "2026-04-22 16:46:13.059 INFO: Error-table on TRAIN and VALID:",
                "+---------------+---------------------+------------------+--------------+-------------------------+---------------+",
                (
                    "|  config_type  | RMSE E / meV / atom | RMSE F / meV / A | rel F RMSE % "
                    "| RMSE MU / mDebye / atom | rel MU RMSE % |"
                ),
                "+---------------+---------------------+------------------+--------------+-------------------------+---------------+",
                (
                    "| train_Default |           33.2      |         59.9     |        6.9   "
                    "|              4.2        |        19.9   |"
                ),
                (
                    "| valid_Default |           31.6      |         61.6     |        7.6   "
                    "|              5.0        |        18.6   |"
                ),
                "+---------------+---------------------+------------------+--------------+-------------------------+---------------+",
            ]
        )
    )

    assert metrics == {
        "dataset": [
            "training_set",
            "training_set",
            "training_set",
            "training_set",
            "training_set",
            "validation_set",
            "validation_set",
            "validation_set",
            "validation_set",
            "validation_set",
            "training_set",
            "training_set",
            "training_set",
            "training_set",
            "training_set",
            "validation_set",
            "validation_set",
            "validation_set",
            "validation_set",
            "validation_set",
        ],
        "property": [
            "energy",
            "forces",
            "forces",
            "dipole",
            "dipole",
            "energy",
            "forces",
            "forces",
            "dipole",
            "dipole",
            "energy",
            "forces",
            "forces",
            "dipole",
            "dipole",
            "energy",
            "forces",
            "forces",
            "dipole",
            "dipole",
        ],
        "metric": [
            "rmse_per_atom",
            "rmse",
            "rel_rmse",
            "rmse_per_atom",
            "rel_rmse",
            "rmse_per_atom",
            "rmse",
            "rel_rmse",
            "rmse_per_atom",
            "rel_rmse",
            "rmse_per_atom",
            "rmse",
            "rel_rmse",
            "rmse_per_atom",
            "rel_rmse",
            "rmse_per_atom",
            "rmse",
            "rel_rmse",
            "rmse_per_atom",
            "rel_rmse",
        ],
        "value": [
            393.7,
            112.3,
            12.8,
            10.1,
            65.2,
            389.3,
            109.3,
            13.4,
            12.0,
            65.7,
            33.2,
            59.9,
            6.9,
            4.2,
            19.9,
            31.6,
            61.6,
            7.6,
            5.0,
            18.6,
        ],
        "source": [
            "train_log_stage_one",
            "train_log_stage_one",
            "train_log_stage_one",
            "train_log_stage_one",
            "train_log_stage_one",
            "train_log_stage_one",
            "train_log_stage_one",
            "train_log_stage_one",
            "train_log_stage_one",
            "train_log_stage_one",
            "train_log_stage_two",
            "train_log_stage_two",
            "train_log_stage_two",
            "train_log_stage_two",
            "train_log_stage_two",
            "train_log_stage_two",
            "train_log_stage_two",
            "train_log_stage_two",
            "train_log_stage_two",
            "train_log_stage_two",
        ],
    }


def test_mace_trainer_iter_tidy_error_metric_rows_skips_opt_rows(tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path})
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    train_log = results_dir / "dipole_mace_run-123_train.txt"
    train_log.write_text(
        "\n".join(
            [
                '{"loss": 2.5, "time": 1.0, "mode": "opt", "epoch": 3}',
                (
                    '{"loss": 1.2, "rmse_e_per_atom": 0.3, "rel_mae_f": 4.5, '
                    '"mae_mu": 0.6, "time": 0.7, "mode": "eval", "epoch": 3, "head": "Default"}'
                ),
            ]
        ),
        encoding="utf-8",
    )

    rows = list(trainer.iter_tidy_error_metric_rows(tmp_path))

    assert rows == [
        {
            "dataset": "validation_set",
            "epoch": 3,
            "step": None,
            "property": None,
            "metrics": "loss",
            "unit": None,
            "value": 1.2,
            "info": {
                "mode": "eval",
                "head": "Default",
                "time": 0.7,
                "raw_path": str(train_log.resolve(strict=False)),
                "line_number": 2,
                "raw_metric_key": "loss",
            },
        },
        {
            "dataset": "validation_set",
            "epoch": 3,
            "step": None,
            "property": "energy",
            "metrics": "rmse_per_atom",
            "unit": None,
            "value": 0.3,
            "info": {
                "mode": "eval",
                "head": "Default",
                "time": 0.7,
                "raw_path": str(train_log.resolve(strict=False)),
                "line_number": 2,
                "raw_metric_key": "rmse_e_per_atom",
            },
        },
        {
            "dataset": "validation_set",
            "epoch": 3,
            "step": None,
            "property": "forces",
            "metrics": "rel_mae",
            "unit": None,
            "value": 4.5,
            "info": {
                "mode": "eval",
                "head": "Default",
                "time": 0.7,
                "raw_path": str(train_log.resolve(strict=False)),
                "line_number": 2,
                "raw_metric_key": "rel_mae_f",
            },
        },
        {
            "dataset": "validation_set",
            "epoch": 3,
            "step": None,
            "property": "dipole",
            "metrics": "mae",
            "unit": None,
            "value": 0.6,
            "info": {
                "mode": "eval",
                "head": "Default",
                "time": 0.7,
                "raw_path": str(train_log.resolve(strict=False)),
                "line_number": 2,
                "raw_metric_key": "mae_mu",
            },
        },
    ]


def test_mace_trainer_plot_training_metrics_discovers_train_log_under_results(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    train_log = results_dir / "dipole_mace_run-123_train.txt"
    train_log.write_text(
        "\n".join(
            [
                '{"loss": 2.5, "time": 1.0, "mode": "opt", "epoch": 0}',
                '{"loss": 1.5, "time": 1.0, "mode": "opt", "epoch": 1}',
                (
                    '{"loss": 1.2, "mae_e": 0.7, "rmse_f": 0.5, "time": 0.7, '
                    '"mode": "eval", "epoch": 0, "head": "Default"}'
                ),
                (
                    '{"loss": 0.8, "mae_e": 0.4, "rmse_f": 0.3, "time": 0.7, '
                    '"mode": "eval", "epoch": 1, "head": "Default"}'
                ),
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "mace_training_metrics.png"

    fig = MACETrainer.plot_training_metrics(
        tmp_path,
        metrics=["loss", "mae_e", "rmse_f"],
        output_path=output_path,
        max_columns=2,
    )

    assert output_path.exists()
    assert [axis.get_title() for axis in fig.axes[:3]] == ["loss", "mae_e", "rmse_f"]
    assert len(fig.axes[0].lines) == 2
    assert len(fig.axes[1].lines) == 1
    assert fig.axes[0].get_xlabel() == "epoch"
    plt.close(fig)


def test_mace_trainer_iter_tidy_error_metric_rows_raises_for_missing_train_log(tmp_path):
    trainer = MACETrainer(data={"folder": tmp_path})

    with pytest.raises(FileNotFoundError, match=r"training log matching '\*_train\.txt'"):
        list(trainer.iter_tidy_error_metric_rows(tmp_path))


def test_mace_trainer_trains_committee_members_in_parallel_subprocesses(monkeypatch, tmp_path):
    import threading

    trainer = MACETrainer(
        data={"folder": tmp_path},
        train_settings={"committee": {"size": 3, "seed": 10, "parallel_members": 3, "gpu_ids": [0, 1]}},
    )
    captured_configs: dict[Path, dict] = {}
    subprocess_calls: list[dict] = []
    all_started = threading.Barrier(3, timeout=5)

    def capture_yaml(self, path, payload):
        del self
        captured_configs[Path(path)] = payload

    def fake_subprocess(config_file_path, workdir, log_path, patch_load_foundations, gpu_id, suppress_warnings):
        subprocess_calls.append(
            {"config": Path(config_file_path), "workdir": Path(workdir), "gpu": gpu_id, "log": Path(log_path)}
        )
        all_started.wait()  # deadlocks (timeout -> BrokenBarrierError) unless the 3 members run concurrently

    def fail_in_process(*args, **kwargs):
        raise AssertionError("parallel members must not train in the current process")

    monkeypatch.setattr(mace_trainer_module, "run_train_mace_subprocess", fake_subprocess)
    monkeypatch.setattr(mace_trainer_module, "train_mace", fail_in_process)
    monkeypatch.setattr(MACETrainer, "_write_yaml", capture_yaml)
    monkeypatch.setattr(
        MACETrainer,
        "_resolve_model_path",
        lambda self, model_dir: str((Path(model_dir) / "model.model").resolve(strict=False)),
    )

    result = trainer.train(data_split=_make_data_split(), engine_id="MACE00")

    calls = sorted(subprocess_calls, key=lambda call: call["workdir"].name)
    assert [call["workdir"].name for call in calls] == ["member_000", "member_001", "member_002"]
    assert [captured_configs[call["config"]]["seed"] for call in calls] == [10, 11, 12]
    assert [call["gpu"] for call in calls] == [0, 1, 0]
    assert all(call["log"].parent == call["workdir"] for call in calls)
    assert result.engine.calculator_kwargs["model_paths"] == [
        str((tmp_path / "MACE00" / f"member_{index:03d}" / "results" / "model.model").resolve(strict=False))
        for index in range(3)
    ]


def test_mace_trainer_parallel_member_failure_is_raised(monkeypatch, tmp_path):
    trainer = MACETrainer(
        data={"folder": tmp_path},
        train_settings={"committee": {"size": 2, "parallel_members": 2}},
    )

    def fake_subprocess(config_file_path, workdir, log_path, **kwargs):
        if Path(workdir).name == "member_001":
            raise RuntimeError("MACE training failed")

    monkeypatch.setattr(mace_trainer_module, "run_train_mace_subprocess", fake_subprocess)
    monkeypatch.setattr(MACETrainer, "_write_yaml", lambda self, path, payload: None)

    with pytest.raises(RuntimeError, match="MACE training failed"):
        trainer.train(data_split=_make_data_split(), engine_id="MACE00")


def test_run_train_mace_subprocess_sets_gpu_and_reports_failures(monkeypatch, tmp_path):
    import subprocess

    from scm.active_learning.mlip.mace_trainer import runner

    captured = {}

    def fake_run(command, stdout, stderr, env, check):
        captured.update(command=command, env=env)
        stdout.write("line 1\nboom\n")
        return subprocess.CompletedProcess(command, returncode=3)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="exit code 3") as excinfo:
        runner.run_train_mace_subprocess(
            config_file_path=tmp_path / "config.yaml",
            workdir=tmp_path,
            log_path=tmp_path / "train.log",
            patch_load_foundations=True,
            gpu_id=1,
        )

    assert "boom" in str(excinfo.value)
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert captured["command"][1:3] == ["-m", "scm.active_learning.mlip.mace_trainer.runner"]
    assert "--patch-load-foundations" in captured["command"]
