from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write as ase_write
from scm.moliterate import ChemDataSetFormat, ConcatChemDataSet, PropertyInfo, create_dataset
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.interfaces import InMemoryMolData
from scm.moliterate.transforms import MetadataAddTransform
from scm.plams import Settings

from scm.active_learning.checker_getters import DecoupledCheckerGetter
from scm.active_learning.checker_getters.checkers import AMSTrajChecker
from scm.active_learning.checker_getters.getters import AMSTrajGetter
from scm.active_learning.engines.atomic_container import SCMChemicalSystem
from scm.active_learning.journey_scheduler import MoleculesJourney
from scm.active_learning.journey_scheduler.molecule_journey import MJState, MoleculeImportConfig
from scm.active_learning.logging.path_serializable_model import PathSerializableModel
from scm.active_learning.results import CollectionCheckGetResults, EngineCheckResult

pytest.importorskip("scm.base")
pytest.importorskip("scm.plams")

_tests_dir = Path(__file__).resolve().parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from test_journey_scheduler import (  # noqa: E402
    _make_check_get_results,
    _make_engine_check_result,
)


def _make_molecules(n: int) -> InMemoryMolData:
    dataset = InMemoryMolData.create(available_properties=[PropertyInfo(name="energy", unit="eV")])
    for _ in range(n):
        dataset.add_system(ChemDataEntry(system=Atoms("H"), properties={"energy": 0.0}))
    return dataset


def _make_file_backed_molecules(path: Path, n: int, prefix: str = "file"):
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
                metadata={"origin": f"{prefix}-{idx}"},
            )
        )
    return dataset


def _make_validity_results(system_id: str, task_ids: list[str], successes: list[bool]):
    results = []
    for task_id, success in zip(task_ids, successes, strict=True):
        results.append(_make_check_get_results(success, task_id=task_id, system_id=system_id))
    return results


def _make_collection_results_for(system_id: str, task_ids: list[str], successes: list[bool]):
    return CollectionCheckGetResults(validity_get_results=_make_validity_results(system_id, task_ids, successes))


def _make_accuracy_results_for(system_id: str, task_ids: list[str], success: bool):
    return [
        _make_engine_check_result(
            success,
            checker_id="accuracy",
            property_name="energy",
            task_id=task_id,
            system_id=system_id,
        )
        for task_id in task_ids
    ]


def _make_couples(n: int):
    checker = DecoupledCheckerGetter(
        checker=AMSTrajChecker(),
        getter=AMSTrajGetter(),
    )
    couples = {}
    for i in range(n):
        settings = Settings()
        settings.input.ams.task = "SinglePoint"
        task_id = f"task-{i + 1}"
        couples[task_id] = ("ams", settings.as_dict(), checker)
    return couples


class JourneySnapshot(PathSerializableModel):
    journey: MoleculesJourney


def test_molecules_journey_batches_and_replaces_completed_molecules():
    molecules = _make_molecules(3)
    couples = _make_couples(2)
    max_attempts = {task_id: 0 for task_id in couples}
    journey = MoleculesJourney(
        molecules=molecules,
        task_checker_getter=couples,
        max_attempts_per_task=max_attempts,
        batch_size=1,
    )

    journey.register_attempt()
    tasks = journey._current_batch_tasks()
    assert len(tasks) == 2
    assert {t.system_id for t in tasks} == {"M0000"}

    task_ids = list(couples.keys())
    result = journey.advance(
        _make_collection_results_for("M0000", task_ids, [True, True]),
        _make_accuracy_results_for("M0000", task_ids, True),
    )
    assert result.skip_training_reason == "All current tasks converged"
    assert journey.registry_.molecules[0].status == "Succeeded"

    journey.register_attempt()
    tasks = journey._current_batch_tasks()
    assert len(tasks) == 2
    assert {t.system_id for t in tasks} == {"M0001"}


def test_molecules_journey_accepts_opted_in_no_data_accuracy_result():
    couples = _make_couples(1)
    task_ids = list(couples)
    journey = MoleculesJourney(
        molecules=_make_molecules(1),
        task_checker_getter=couples,
        max_attempts_per_task={task_id: 0 for task_id in task_ids},
        batch_size=1,
    )
    journey.register_attempt()
    no_data_accuracy_result = EngineCheckResult(
        engine_id="engine",
        task_id=task_ids[0],
        system_id="M0000",
        checker_id="accuracy",
        property="forces",
        metric="mae",
        value=float("nan"),
        n_entries=0,
        success="NoData",
        no_data_is_success=True,
    )

    result = journey.advance(
        _make_collection_results_for("M0000", task_ids, [True]),
        [no_data_accuracy_result],
    )

    assert result.skip_training_reason == "All current tasks converged"
    assert journey.registry_.molecules[0].status == "Succeeded"


def test_molecules_journey_skips_completed_tasks_in_batch():
    molecules = _make_molecules(1)
    couples = _make_couples(3)
    max_attempts = {task_id: 0 for task_id in couples}
    journey = MoleculesJourney(
        molecules=molecules,
        task_checker_getter=couples,
        max_attempts_per_task=max_attempts,
        batch_size=1,
    )
    journey.register_attempt()
    journey.registry_.molecules[0].tasks["task-1"].status = "Finished"

    tasks = journey._current_batch_tasks()
    assert [t.task_id for t in tasks] == ["task-2", "task-3"]
    assert list(journey.registry_.current_running()) == [(0, "task-2"), (0, "task-3")]


def test_molecules_journey_register_attempt_tracks_all_active_tasks():
    molecules = _make_molecules(2)
    couples = _make_couples(2)
    max_attempts = {task_id: 0 for task_id in couples}
    journey = MoleculesJourney(
        molecules=molecules,
        task_checker_getter=couples,
        max_attempts_per_task=max_attempts,
        batch_size=2,
    )

    journey.register_attempt()
    for mol in journey.registry_.molecules:
        assert mol.attempt == 1
        assert [t.attempt for t in mol.tasks.values()] == [1, 1]


def test_molecules_journey_default_state_and_max_attempts():
    molecules = _make_molecules(2)
    couples = _make_couples(2)
    max_attempts = {task_id: 2 for task_id in couples}

    journey = MoleculesJourney(
        molecules=molecules,
        task_checker_getter=couples,
        max_attempts_per_task=max_attempts,
        batch_size=1,
    )

    assert [m.status for m in journey.registry_.molecules] == ["Pending", "Pending"]
    assert journey.n_steps() == 2
    assert journey.n_steps_left() == 0
    assert journey.is_finished() is False
    assert journey.registry_.molecules[0].tasks["task-1"].max_attempt == 2
    assert journey.registry_.molecules[0].tasks["task-2"].max_attempt == 2


def test_molecules_journey_respects_preseeded_state_and_register_attempt():
    molecules = _make_molecules(2)
    couples = _make_couples(1)
    max_attempts = {task_id: 1 for task_id in couples}

    journey = MoleculesJourney(
        molecules=molecules,
        task_checker_getter=couples,
        max_attempts_per_task=max_attempts,
        batch_size=1,
    )

    journey.registry_.molecules[0].set_running()
    journey.registry_.molecules[0].attempt = 2
    journey.registry_.molecules[0].tasks["task-1"].attempt = 2

    journey.register_attempt()
    assert journey.registry_.molecules[0].attempt == 3
    assert journey.registry_.molecules[0].tasks["task-1"].attempt == 3


def test_molecules_journey_snapshot_reload_preserves_registry_state(tmp_path):
    couples = _make_couples(1)
    journey = MoleculesJourney(
        molecules=_make_molecules(2),
        task_checker_getter=couples,
        max_attempts_per_task={"task-1": 2},
        batch_size=1,
    )
    journey.register_attempt()
    journey.registry_.molecules[0].tasks["task-1"].record_failure("AccuracyFailure")

    snapshot_path = JourneySnapshot(journey=journey).write_model_json(tmp_path / "journey.json")
    reloaded = JourneySnapshot.load_model_json(snapshot_path).journey

    assert reloaded.registry_.molecules[0].status == "Running"
    assert reloaded.registry_.molecules[0].attempt == 1
    task = reloaded.registry_.molecules[0].tasks["task-1"]
    assert task.status == "Running"
    assert task.attempt == 1
    assert task.attempt_messages == {1: "AccuracyFailure"}
    assert reloaded.registry_.molecules[1].status == "Pending"


def test_molecules_journey_advance_and_failure_modes():
    molecules = _make_molecules(1)
    couples = _make_couples(1)
    max_attempts = {task_id: 0 for task_id in couples}
    task_ids = list(couples.keys())

    journey = MoleculesJourney(
        molecules=molecules,
        task_checker_getter=couples,
        max_attempts_per_task=max_attempts,
        batch_size=1,
    )
    journey.register_attempt()
    result = journey.advance(
        _make_collection_results_for("M0000", task_ids, [True]),
        _make_accuracy_results_for("M0000", task_ids, False),
    )
    assert result.skip_training_reason == ""
    assert journey.registry_.molecules[0].status == "Finished"
    assert journey.is_finished() is True
    assert journey.history[-1].tot_succeeded == 0
    assert (
        journey.finished_reason()
        == "MoleculesJourney did not converge: 0/1 molecules succeeded; 1 stopped without succeeding"
    )

    journey = MoleculesJourney(
        molecules=molecules,
        task_checker_getter=couples,
        max_attempts_per_task=max_attempts,
        batch_size=1,
    )
    journey.register_attempt()
    result = journey.advance(
        _make_collection_results_for("M0000", task_ids, [False]),
        _make_accuracy_results_for("M0000", task_ids, True),
    )
    assert result.skip_training_reason == ""
    assert journey.registry_.molecules[0].status == "Finished"

    journey = MoleculesJourney(
        molecules=molecules,
        task_checker_getter=couples,
        max_attempts_per_task=max_attempts,
        batch_size=1,
    )
    journey.register_attempt()
    result = journey.advance(
        _make_collection_results_for("M0000", task_ids, [True]),
        _make_accuracy_results_for("M0000", task_ids, True),
    )
    assert result.skip_training_reason == "All current tasks converged"
    assert journey.registry_.molecules[0].status == "Succeeded"
    assert journey.is_finished() is True
    assert journey.finished_reason() == "CONVERGED"


def test_molecules_journey_preserves_successful_tasks_when_siblings_fail():
    molecules = _make_molecules(1)
    couples = _make_couples(3)
    max_attempts = {task_id: 5 for task_id in couples}
    task_ids = list(couples.keys())
    failed_task_ids = task_ids[:2]
    successful_task_id = task_ids[2]

    journey = MoleculesJourney(
        molecules=molecules,
        task_checker_getter=couples,
        max_attempts_per_task=max_attempts,
        batch_size=1,
    )
    journey.register_attempt()
    result = journey.advance(
        _make_collection_results_for("M0000", task_ids, [True, True, True]),
        [
            _make_engine_check_result(
                success=False,
                checker_id="accuracy",
                property_name="energy",
                task_id=failed_task_ids[0],
                system_id="M0000",
            ),
            _make_engine_check_result(
                success=False,
                checker_id="accuracy",
                property_name="energy",
                task_id=failed_task_ids[1],
                system_id="M0000",
            ),
            _make_engine_check_result(
                success=True,
                checker_id="accuracy",
                property_name="energy",
                task_id=successful_task_id,
                system_id="M0000",
            ),
        ],
    )

    assert result.skip_training_reason == ""
    molecule = journey.registry_.molecules[0]
    assert molecule.status == "Running"
    assert molecule.tasks[failed_task_ids[0]].status == "Running"
    assert molecule.tasks[failed_task_ids[1]].status == "Running"
    assert molecule.tasks[successful_task_id].status == "Succeeded"

    journey.register_attempt()
    tasks = journey._current_batch_tasks()
    assert [t.task_id for t in tasks] == failed_task_ids


def test_molecules_journey_current_batch_and_invalid_task_type():
    molecules = _make_molecules(1)
    couples = _make_couples(1)
    max_attempts = {task_id: 0 for task_id in couples}
    journey = MoleculesJourney(
        molecules=molecules,
        task_checker_getter=couples,
        max_attempts_per_task=max_attempts,
        batch_size=1,
    )

    journey.register_attempt()
    tasks = journey._current_batch_tasks()
    assert len(tasks) == 1
    assert tasks[0].task_id == "task-1"
    assert tasks[0].system_id == "M0000"

    with pytest.raises(ValueError):
        journey._build_task("task-1", "invalid", {}, SCMChemicalSystem(system_id="sys", chemical_systems={"": "H"}))


def test_molecules_journey_history_table_explains_missing_summary_with_running_tasks():
    molecules = _make_molecules(2)
    couples = _make_couples(2)
    journey = MoleculesJourney(
        molecules=molecules,
        task_checker_getter=couples,
        max_attempts_per_task={task_id: 1 for task_id in couples},
        batch_size=2,
    )

    journey.register_attempt()

    history = journey.history_table()

    assert "No journey history has been recorded yet." in history
    assert "convergence/advance has not produced a summary snapshot" in history
    assert "Recent Failures" in history
    assert "M0000" in history


def test_molecules_journey_history_table_explains_fully_empty_view():
    molecules = _make_molecules(1)
    couples = _make_couples(1)
    journey = MoleculesJourney(
        molecules=molecules,
        task_checker_getter=couples,
        max_attempts_per_task={task_id: 1 for task_id in couples},
        batch_size=1,
    )

    history = journey.history_table(options="ExcludePending")

    assert "No journey history has been recorded yet." in history
    assert "No molecule/task attempts are available for the selected history view." in history
    assert "Recent Failures" not in history


def test_molecules_journey_history_table_keeps_banner_width_bounded():
    molecules = _make_molecules(1)
    couples = _make_couples(1)
    journey = MoleculesJourney(
        molecules=molecules,
        task_checker_getter=couples,
        max_attempts_per_task={task_id: 5 for task_id in couples},
        batch_size=1,
    )

    journey.history.append(
        MJState(
            tot_pending=1,
            tot_inactivated=0,
            tot_succeeded=0,
            n_running=1,
            n_inactivated=0,
            n_succeeded=0,
            all_running_succeeded=False,
        )
    )
    journey.register_attempt()
    task = next(iter(journey.registry_.molecules[0].tasks.values()))
    for attempt in range(1, 7):
        task.attempt_messages[attempt] = "AccuracyFailure"

    history = journey.history_table()
    title_line = history.splitlines()[0]

    assert "MoleculesJourney History" in title_line
    assert len(title_line) <= journey.max_banner_width + len(journey.history_separator)


def test_molecules_journey_prepare_for_run_folder_imports_in_memory_dataset(tmp_path):
    original = _make_molecules(2)
    original_first_row = original[0]
    journey = MoleculesJourney(
        molecules=original,
        molecule_import=MoleculeImportConfig(enabled=True),
        task_checker_getter=_make_couples(1),
        max_attempts_per_task={"task-1": 0},
        batch_size=1,
    )

    journey.prepare_for_run_folder(tmp_path / "run")

    assert journey.molecules.type == "ASEMolData"
    assert Path(journey.molecules.data_source).is_file()
    assert Path(journey.molecules.data_source) == (tmp_path / "run" / "external" / "_imported" / "journey_molecules.db")
    imported_rows = list(journey.molecules)
    assert len(imported_rows) == 2
    assert imported_rows[0].properties == original_first_row.properties
    assert journey.molecule_import.original_molecules is not None
    assert journey.molecule_import.original_molecules["type"] == "InMemoryMolData"


def test_molecule_import_is_enabled_by_default():
    assert MoleculeImportConfig().enabled is True


@pytest.mark.parametrize("path_type", [Path, str])
def test_molecules_journey_path_input_is_imported_only_during_run_preparation(tmp_path, path_type):
    xyz_path = tmp_path / "molecules.xyz"
    atoms = Atoms("H", info={"name": "hydrogen", "nested": {"source": "test"}})
    ase_write(xyz_path, [atoms, Atoms("He")], format="extxyz")
    journey = MoleculesJourney(
        molecules=path_type(xyz_path),
        molecule_import=MoleculeImportConfig(enabled=False),
        task_checker_getter=_make_couples(1),
        max_attempts_per_task={"task-1": 0},
        batch_size=1,
    )

    imported_path = tmp_path / "run" / "external" / "_imported" / "journey_molecules.db"
    assert not imported_path.exists()
    assert journey.molecule_import.enabled is True
    assert journey.registry_.molecules == []

    snapshot_path = JourneySnapshot(journey=journey).write_model_json(tmp_path / "saved" / "journey.json")
    snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot_payload["journey"]["molecules"] == "../molecules.xyz"

    journey = JourneySnapshot.load_model_json(snapshot_path).journey
    assert Path(journey.molecules) == xyz_path.resolve()
    assert journey.molecule_import.enabled is True

    journey.prepare_for_run_folder(tmp_path / "run")

    assert imported_path.is_file()
    assert journey.molecules.type == "ASEMolData"
    assert journey.molecules.available_properties == []
    assert len(journey.molecules) == 2
    assert journey.molecules[0].metadata == {"name": "hydrogen", "nested": {"source": "test"}}
    assert journey.molecules[1].metadata["name"] == "He-1"
    assert len(journey.registry_.molecules) == 2


def test_molecules_journey_prepare_for_run_folder_flattens_concat_sources(tmp_path):
    file_backed = _make_file_backed_molecules(tmp_path / "source.db", 1, prefix="file")
    in_memory = _make_molecules(2)
    in_memory.update_row_metadata(0, source="memory-0")
    in_memory.update_row_metadata(1, source="memory-1")
    combined = ConcatChemDataSet(data_source=[file_backed, in_memory])
    journey = MoleculesJourney(
        molecules=combined,
        molecule_import=MoleculeImportConfig(enabled=True),
        task_checker_getter=_make_couples(1),
        max_attempts_per_task={"task-1": 0},
        batch_size=1,
    )

    journey.prepare_for_run_folder(tmp_path / "run")

    imported_rows = list(journey.molecules)
    assert [row.metadata["origin"] for row in imported_rows[:1]] == ["file-0"]
    assert [row.metadata["source"] for row in imported_rows[1:]] == ["memory-0", "memory-1"]


def test_molecules_journey_prepare_for_run_folder_preserves_effective_view_subset_and_transforms(tmp_path):
    molecules = _make_file_backed_molecules(tmp_path / "source.db", 3, prefix="subset")
    molecules = molecules.subset([2, 0])
    molecules.transforms.append(MetadataAddTransform(key_values={"tag": "prepared"}))
    journey = MoleculesJourney(
        molecules=molecules,
        molecule_import=MoleculeImportConfig(enabled=True),
        task_checker_getter=_make_couples(1),
        max_attempts_per_task={"task-1": 0},
        batch_size=1,
    )

    expected_rows = list(molecules)

    journey.prepare_for_run_folder(tmp_path / "run")

    imported_rows = list(journey.molecules)
    assert [row.metadata["origin"] for row in imported_rows] == [row.metadata["origin"] for row in expected_rows]
    assert [row.metadata["tag"] for row in imported_rows] == ["prepared", "prepared"]


def test_molecules_journey_prepare_for_run_folder_disabled_keeps_original_source(tmp_path):
    molecules = _make_file_backed_molecules(tmp_path / "source.db", 1)
    journey = MoleculesJourney(
        molecules=molecules,
        molecule_import=MoleculeImportConfig(enabled=False),
        task_checker_getter=_make_couples(1),
        max_attempts_per_task={"task-1": 0},
        batch_size=1,
    )

    journey.prepare_for_run_folder(tmp_path / "run")

    assert journey.molecules is molecules
    assert not (tmp_path / "run" / "external" / "_imported" / "journey_molecules.db").exists()


def test_molecules_journey_prepare_for_run_folder_skips_when_already_imported(tmp_path):
    molecules = _make_file_backed_molecules(tmp_path / "source.db", 1)
    journey = MoleculesJourney(
        molecules=molecules,
        molecule_import=MoleculeImportConfig(enabled=True),
        task_checker_getter=_make_couples(1),
        max_attempts_per_task={"task-1": 0},
        batch_size=1,
    )

    journey.prepare_for_run_folder(tmp_path / "run")

    first_data_source = journey.molecules.data_source
    first_original = journey.molecule_import.original_molecules

    journey.prepare_for_run_folder(tmp_path / "other_run")

    assert journey.molecules.data_source == first_data_source
    assert journey.molecule_import.original_molecules == first_original
    assert not (tmp_path / "other_run" / "external" / "_imported" / "journey_molecules.db").exists()


def test_molecules_journey_snapshot_reload_keeps_imported_dataset_and_skips_reimport(tmp_path):
    source_path = tmp_path / "source.db"
    molecules = _make_file_backed_molecules(source_path, 2)
    journey = MoleculesJourney(
        molecules=molecules,
        molecule_import=MoleculeImportConfig(enabled=True),
        task_checker_getter=_make_couples(1),
        max_attempts_per_task={"task-1": 0},
        batch_size=1,
    )

    journey.prepare_for_run_folder(tmp_path / "run")
    imported_path = Path(journey.molecules.data_source)
    snapshot_path = JourneySnapshot(journey=journey).write_model_json(tmp_path / "saved" / "journey.json")

    source_path.unlink()

    reloaded = JourneySnapshot.load_model_json(snapshot_path)

    assert reloaded.journey.molecules.type == "ASEMolData"
    assert Path(reloaded.journey.molecules.data_source) == imported_path
    assert reloaded.journey.molecule_import.original_molecules is not None
    assert reloaded.journey.molecule_import.original_molecules["data_source"] == source_path.resolve().as_posix()

    reloaded.journey.prepare_for_run_folder(tmp_path / "restarted_run")

    assert Path(reloaded.journey.molecules.data_source) == imported_path
    assert not (tmp_path / "restarted_run" / "external" / "_imported" / "journey_molecules.db").exists()
