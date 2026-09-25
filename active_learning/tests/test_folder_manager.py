from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import scm.plams as plams
import yaml
from ase import Atoms
from pydantic import BaseModel, ConfigDict, field_serializer
from scm.moliterate import ChemDataSetFormat, PropertyInfo, create_dataset
from scm.moliterate.core.chem_data_entry import ChemDataEntry

from scm.active_learning.callbacks.folder_manager import FolderManagerCallback, IterationCleanup, RootDir
from scm.active_learning.engines import PLAMSMolecule
from scm.active_learning.journey_scheduler import MoleculesJourney
from scm.active_learning.journey_scheduler.molecule_journey import MoleculeImportConfig
from scm.active_learning.logging.path_serializable_model import PathSerializableModel
from scm.active_learning.loop.loop import ActiveLearningLoop
from scm.active_learning.loop.run_control import ActiveLearningRunControl
from scm.active_learning.tasks import ASEMolecularDynamics


class DummyJobWithLock:
    def __init__(self, path: str):
        self.path = path
        self.lock = threading.Lock()


class DummyJourney(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    previous_job_: Optional[Any] = None

    @field_serializer("previous_job_", check_fields=False)
    def serialize_previous_job_path(self, previous_job_: Optional[Any]) -> Optional[str]:
        if previous_job_ is None:
            return None
        return previous_job_.path


class DummyLabeller(BaseModel):
    properties: List[PropertyInfo]
    out_path: Optional[str] = "labeller.db"


class DummyTrainerData(BaseModel):
    folder: Path = Path("trainer_data")


class DummyTrainer(BaseModel):
    datapath: Optional[str] = None
    data: DummyTrainerData = DummyTrainerData()


class DummySplitting(BaseModel):
    dataset: Any = None


class DummyLoop(PathSerializableModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    journey: Any
    labeller: DummyLabeller
    splitting: DummySplitting = DummySplitting()
    mlip_trainer: Optional[Any] = None
    callbacks: List[FolderManagerCallback]
    run_control: ActiveLearningRunControl = ActiveLearningRunControl()
    current_state: Any = None


class DummyTaskJourney(DummyJourney):
    task: ASEMolecularDynamics


class CleanupJourney(BaseModel):
    previous_job_: Optional[Path] = None


class CleanupEngine(BaseModel):
    job_path: Optional[Path] = None


class CleanupTrainResults(BaseModel):
    engine: CleanupEngine


class CleanupIteration(BaseModel):
    train_results: Optional[CleanupTrainResults] = None


class CleanupResult(BaseModel):
    iterations: List[CleanupIteration] = []


class CleanupCurrentState(BaseModel):
    start_engine: Optional[CleanupEngine] = None


class CleanupLoop(BaseModel):
    journey: CleanupJourney = CleanupJourney()
    current_state: CleanupCurrentState = CleanupCurrentState()
    result: CleanupResult = CleanupResult()


def _write_file(path: Path, content: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _make_ase_md_task() -> ASEMolecularDynamics:
    from scm.plams import Atom, Molecule

    molecule = Molecule()
    molecule.add_atom(Atom(symbol="H", coords=(0.0, 0.0, 0.0)))
    molecule.add_atom(Atom(symbol="H", coords=(0.0, 0.0, 0.74)))
    return ASEMolecularDynamics(
        atomistic_system=PLAMSMolecule.from_molecules(system_id="H2", systems=molecule),
        nsteps=10,
        samplingfreq=1,
    )


def _make_file_backed_molecules(path: Path, n: int = 1):
    dataset = create_dataset(
        path,
        fmt=ChemDataSetFormat.ASE,
        available_properties=[PropertyInfo(name="energy", unit="eV")],
    )
    for idx in range(n):
        dataset.add_system(
            ChemDataEntry(
                system=Atoms("H"),
                properties={"energy": float(idx)},
                metadata={"origin": f"file-{idx}"},
            )
        )
    return dataset


def _make_molecules_journey(molecules, *, enabled: bool) -> MoleculesJourney:
    return MoleculesJourney(
        molecules=molecules,
        molecule_import=MoleculeImportConfig(enabled=enabled),
        task_checker_getter={},
        max_attempts_per_task={},
        batch_size=1,
    )


def test_folder_manager_before_loop_writes_start_snapshot_without_deepcopying_jobs(tmp_path):
    callback = FolderManagerCallback(
        root_dir=RootDir(base_dir=tmp_path, run_name="restart_copy"),
        logging_file=None,
        state_format="json",
    )
    loop = DummyLoop(
        journey=DummyJourney(previous_job_=DummyJobWithLock("previous-job-path")),
        labeller=DummyLabeller(properties=[PropertyInfo(name="energy", unit="eV")]),
        callbacks=[callback],
    )

    callback.before_loop(loop)

    snapshot_path = tmp_path / "restart_copy" / "al_start_state.json"
    assert snapshot_path.is_file()

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["journey"]["previous_job_"] == "previous-job-path"
    assert payload["callbacks"][0]["root_dir"]["run_root"] == "."


def test_folder_manager_writes_yaml_state_snapshots_by_default(tmp_path):
    callback = FolderManagerCallback(
        root_dir=RootDir(base_dir=tmp_path, run_name="yaml_run"),
        logging_file=None,
        set_plams_jobmanager_folder=False,
    )
    loop = DummyLoop(
        journey=DummyJourney(),
        labeller=DummyLabeller(properties=[PropertyInfo(name="energy", unit="eV")]),
        callbacks=[callback],
    )

    callback.before_loop(loop)
    callback.start_iter_loop(loop, SimpleNamespace(iteration_al=0))
    callback.after_loop(loop)

    start_path = tmp_path / "yaml_run" / "al_start_state.yaml"
    iteration_path = tmp_path / "yaml_run" / "iter_00" / "al_state.yaml"
    final_path = tmp_path / "yaml_run" / "al_final_state.yaml"
    assert start_path.is_file()
    assert iteration_path.is_file()
    assert final_path.is_file()
    assert yaml.safe_load(start_path.read_text(encoding="utf-8"))["callbacks"][0]["state_format"] == "yaml"
    assert not (tmp_path / "yaml_run" / "al_final_state.json").exists()


def test_folder_manager_accepts_legacy_log_al_state_json_field():
    callback = FolderManagerCallback.model_validate({"log_al_state_json": False})

    assert callback.log_al_state is False
    assert callback.log_al_state_json is False

    callback.log_al_state_json = True
    assert callback.log_al_state is True


def test_folder_manager_resume_before_loop_cleans_current_iteration_without_resetting_dataset(tmp_path):
    run_dir = tmp_path / "resume_run"
    callback = FolderManagerCallback(
        root_dir=RootDir(run_root=run_dir),
        logging_file=None,
    )
    dataset_path = _write_file(run_dir / "train_validation_AL.db", "keep")
    stale_file = _write_file(run_dir / "iter_02" / "partial_job" / "ams.rkf", "partial")
    loop = DummyLoop(
        journey=DummyJourney(),
        labeller=DummyLabeller(properties=[PropertyInfo(name="energy", unit="eV")]),
        callbacks=[callback],
        run_control=ActiveLearningRunControl(mode="resume"),
        current_state=SimpleNamespace(iteration_al=2, start_engine=None),
    )

    callback.before_loop(loop)

    assert stale_file.exists() is False
    assert (run_dir / "iter_02").exists() is False
    assert dataset_path.read_text(encoding="utf-8") == "keep"
    assert (run_dir / "al_start_state.yaml").exists() is False


def test_active_learning_loop_tag_updates_folder_manager_timestamp_format(tmp_path):
    callback = FolderManagerCallback(
        root_dir=RootDir(base_dir=tmp_path),
        logging_file=None,
    )
    loop = ActiveLearningLoop.model_construct(
        tag="demo",
        splitting=SimpleNamespace(split_key="configured"),
        callbacks=[callback],
    )

    loop.model_post_init(None)

    assert callback.root_dir.timestamp_format == "%Y%m%d_%H%M%S-demo"


def test_folder_manager_cleanup_slim_preserves_registered_paths_and_root_metadata(tmp_path):
    run_dir = tmp_path / "run"
    callback = FolderManagerCallback(
        root_dir=RootDir(run_root=run_dir),
        logging_file=None,
        cleanup=IterationCleanup(mode="slim", keep_last_full_iterations=1),
    )

    _write_file(run_dir / "al_start_state.json")
    _write_file(run_dir / "al_final_state.json")
    _write_file(run_dir / "train_validation_AL.db")

    protected_previous_job = run_dir / "iter_00" / "restart_job"
    _write_file(protected_previous_job / "ams.rkf")
    protected_trainer = _write_file(run_dir / "iter_01" / "ParAMSTrainer" / "results" / "model.bin")
    _write_file(run_dir / "iter_00" / "SinglePoints" / "frame.xyz")
    _write_file(run_dir / "iter_00" / "al_state.json")
    _write_file(run_dir / "iter_00" / "al_state.yaml")
    _write_file(run_dir / "iter_00" / "job_logfile.csv")
    _write_file(run_dir / "iter_00" / "logfile")
    _write_file(run_dir / "iter_01" / "labelled_data.db")
    current_heavy = _write_file(run_dir / "iter_02" / "SinglePoints" / "current.xyz")

    loop = CleanupLoop(
        journey=CleanupJourney(previous_job_=protected_previous_job),
        current_state=CleanupCurrentState(start_engine=CleanupEngine(job_path=protected_trainer.parent.parent)),
        result=CleanupResult(
            iterations=[
                CleanupIteration(
                    train_results=CleanupTrainResults(engine=CleanupEngine(job_path=protected_trainer.parent.parent))
                )
            ]
        ),
    )

    callback.after_train(loop, SimpleNamespace(iteration_al=2))

    assert (run_dir / "al_start_state.json").is_file()
    assert (run_dir / "al_final_state.json").is_file()
    assert (run_dir / "train_validation_AL.db").is_file()

    assert (run_dir / "iter_00").is_dir()
    assert (run_dir / "iter_00" / "restart_job" / "ams.rkf").is_file()
    assert (run_dir / "iter_00" / "SinglePoints").exists() is False
    assert (run_dir / "iter_00" / "al_state.json").is_file()
    assert (run_dir / "iter_00" / "al_state.yaml").is_file()
    assert (run_dir / "iter_00" / "job_logfile.csv").is_file()
    assert (run_dir / "iter_00" / "logfile").is_file()

    assert (run_dir / "iter_01").is_dir()
    assert (run_dir / "iter_01" / "ParAMSTrainer").is_dir()
    assert (run_dir / "iter_01" / "labelled_data.db").exists() is False

    assert current_heavy.is_file()


def test_folder_manager_cleanup_delete_removes_only_unprotected_old_iterations(tmp_path):
    run_dir = tmp_path / "run"
    callback = FolderManagerCallback(
        root_dir=RootDir(run_root=run_dir),
        logging_file=None,
        cleanup=IterationCleanup(mode="delete", keep_last_full_iterations=1),
    )

    _write_file(run_dir / "al_start_state.json")
    _write_file(run_dir / "al_final_state.json")
    _write_file(run_dir / "train_validation_AL.db")

    protected_previous_job = run_dir / "iter_00" / "restart_job"
    _write_file(protected_previous_job / "ams.rkf")
    _write_file(run_dir / "iter_00" / "SinglePoints" / "frame.xyz")
    _write_file(run_dir / "iter_00" / "al_state.json")
    _write_file(run_dir / "iter_01" / "SinglePoints" / "stale.xyz")
    current_heavy = _write_file(run_dir / "iter_02" / "SinglePoints" / "current.xyz")

    loop = CleanupLoop(journey=CleanupJourney(previous_job_=protected_previous_job))

    callback.after_train(loop, SimpleNamespace(iteration_al=2))

    assert (run_dir / "al_start_state.json").is_file()
    assert (run_dir / "al_final_state.json").is_file()
    assert (run_dir / "train_validation_AL.db").is_file()

    assert (run_dir / "iter_00").is_dir()
    assert (run_dir / "iter_00" / "restart_job" / "ams.rkf").is_file()
    assert (run_dir / "iter_00" / "SinglePoints").exists() is False
    assert (run_dir / "iter_00" / "al_state.json").is_file()

    assert (run_dir / "iter_01").exists() is False
    assert current_heavy.is_file()


def test_folder_manager_start_iter_loop_moves_mlip_paths_into_iteration_dir(tmp_path):
    callback = FolderManagerCallback(
        root_dir=RootDir(base_dir=tmp_path, run_name="managed"),
        logging_file=None,
        log_al_state_json=False,
        set_plams_jobmanager_folder=False,
        mlip_dataname="mlip_store",
    )
    trainer = DummyTrainer()
    loop = DummyLoop(
        journey=DummyJourney(),
        labeller=DummyLabeller(properties=[PropertyInfo(name="energy", unit="eV")]),
        mlip_trainer=trainer,
        callbacks=[callback],
    )

    callback.start_iter_loop(loop, SimpleNamespace(iteration_al=3))

    expected = tmp_path / "managed" / "iter_03" / "mlip_store"
    assert trainer.data.folder == expected
    assert trainer.datapath == str(expected)


def test_folder_manager_start_iter_loop_moves_ase_md_save_folder_into_iteration_dir(tmp_path):
    callback = FolderManagerCallback(
        root_dir=RootDir(base_dir=tmp_path, run_name="managed"),
        logging_file=None,
        log_al_state_json=False,
        set_plams_jobmanager_folder=False,
    )
    task = _make_ase_md_task()
    loop = DummyLoop(
        journey=DummyTaskJourney(task=task),
        labeller=DummyLabeller(properties=[PropertyInfo(name="energy", unit="eV")]),
        callbacks=[callback],
    )

    callback.start_iter_loop(loop, SimpleNamespace(iteration_al=4))

    assert task.save_folder == str(tmp_path / "managed" / "iter_04")


def test_folder_manager_after_loop_restores_original_folder_settings(tmp_path):
    original_jobmanager = plams.config.default_jobmanager
    callback = FolderManagerCallback(
        root_dir=RootDir(base_dir=tmp_path, run_name="managed"),
        logging_file=None,
        log_al_state_json=False,
    )
    task = _make_ase_md_task()
    task.save_folder = "original_md_folder"
    trainer = DummyTrainer(
        datapath="original_trainer_path", data=DummyTrainerData(folder=Path("original_trainer_data"))
    )
    loop = DummyLoop(
        journey=DummyTaskJourney(task=task),
        labeller=DummyLabeller(
            properties=[PropertyInfo(name="energy", unit="eV")],
            out_path="original_labeller.db",
        ),
        mlip_trainer=trainer,
        callbacks=[callback],
    )

    callback.before_loop(loop)
    callback.start_iter_loop(loop, SimpleNamespace(iteration_al=1))

    assert task.save_folder == str(tmp_path / "managed" / "iter_01")
    assert loop.labeller.out_path == str(tmp_path / "managed" / "iter_01" / "labelled_data.db")
    assert trainer.data.folder == tmp_path / "managed" / "iter_01" / "mlip_data"
    assert trainer.datapath == str(tmp_path / "managed" / "iter_01" / "mlip_data")
    assert plams.config.default_jobmanager is not original_jobmanager

    callback.after_loop(loop)

    assert task.save_folder == "original_md_folder"
    assert loop.labeller.out_path == "original_labeller.db"
    assert trainer.data.folder == Path("original_trainer_data")
    assert trainer.datapath == "original_trainer_path"
    assert plams.config.default_jobmanager is original_jobmanager


def test_folder_manager_before_loop_imports_journey_molecules_before_writing_start_snapshot(tmp_path):
    callback = FolderManagerCallback(
        root_dir=RootDir(base_dir=tmp_path, run_name="managed"),
        logging_file=None,
    )
    original_molecules = _make_file_backed_molecules(tmp_path / "source.db")
    journey = _make_molecules_journey(original_molecules, enabled=True)
    loop = DummyLoop(
        journey=journey,
        labeller=DummyLabeller(properties=[PropertyInfo(name="energy", unit="eV")]),
        callbacks=[callback],
    )

    callback.before_loop(loop)

    snapshot_path = tmp_path / "managed" / "al_start_state.yaml"
    payload = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))

    assert journey.molecules.type == "ASEMolData"
    assert (tmp_path / "managed" / "external" / "_imported" / "journey_molecules.db").is_file()
    assert payload["journey"]["molecules"]["data_source"] == "external/_imported/journey_molecules.db"
    assert payload["journey"]["molecule_import"]["original_molecules"]["data_source"] == "../source.db"


def test_folder_manager_after_loop_keeps_imported_journey_molecules_and_logs_provenance(tmp_path):
    callback = FolderManagerCallback(
        root_dir=RootDir(base_dir=tmp_path, run_name="managed"),
        logging_file=None,
    )
    original_molecules = _make_file_backed_molecules(tmp_path / "source.db")
    journey = _make_molecules_journey(original_molecules, enabled=True)
    loop = DummyLoop(
        journey=journey,
        labeller=DummyLabeller(properties=[PropertyInfo(name="energy", unit="eV")]),
        callbacks=[callback],
    )

    callback.before_loop(loop)
    imported_path = Path(journey.molecules.data_source)

    callback.after_loop(loop)

    final_payload = yaml.safe_load((tmp_path / "managed" / "al_final_state.yaml").read_text(encoding="utf-8"))

    assert journey.molecules.type == "ASEMolData"
    assert Path(journey.molecules.data_source) == imported_path
    assert final_payload["journey"]["molecules"]["data_source"] == "external/_imported/journey_molecules.db"
    assert final_payload["journey"]["molecule_import"]["original_molecules"]["data_source"] == "../source.db"


def test_folder_manager_before_loop_keeps_original_journey_source_when_import_disabled(tmp_path):
    callback = FolderManagerCallback(
        root_dir=RootDir(base_dir=tmp_path, run_name="managed"),
        logging_file=None,
    )
    original_molecules = _make_file_backed_molecules(tmp_path / "source.db")
    loop = DummyLoop(
        journey=_make_molecules_journey(original_molecules, enabled=False),
        labeller=DummyLabeller(properties=[PropertyInfo(name="energy", unit="eV")]),
        callbacks=[callback],
    )

    callback.before_loop(loop)

    snapshot_path = tmp_path / "managed" / "al_start_state.yaml"
    payload = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))

    assert not (tmp_path / "managed" / "external" / "_imported" / "journey_molecules.db").exists()
    assert payload["journey"]["molecules"]["data_source"] == "../source.db"
