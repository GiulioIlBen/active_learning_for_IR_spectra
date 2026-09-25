from __future__ import annotations

import numpy as np
from ase import Atoms
from scm.moliterate import PropertyInfo
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.interfaces.in_memory import InMemoryMolData

from scm.active_learning.engines import Engine
from scm.active_learning.mlip import DataAndSplittingStrategy
from scm.active_learning.results import EngineCheckResult, JourneyAdvanceResult


def _make_dataset(task_id: str = "AMSMDTask") -> InMemoryMolData:
    dataset = InMemoryMolData.create(available_properties=[PropertyInfo(name="energy", unit="eV")])
    base_atoms = Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])

    for idx in range(2):
        dataset.add_system(
            ChemDataEntry(
                system=base_atoms.copy(),
                properties={"energy": float(idx)},
                metadata={"task_id": task_id},
            )
        )
    return dataset


def _make_accuracy_result(success: bool, task_id: str = "AMSMDTask") -> list[EngineCheckResult]:
    return [
        EngineCheckResult(
            engine_id="engine",
            task_id=task_id,
            system_id="system",
            checker_id="accuracy",
            property="energy",
            metric="mae",
            value=0.0 if success else 1.0,
            target=0.5,
            n_entries=2,
            success="OK" if success else "FAIL",
        )
    ]


class _EngineWithDatasets(Engine):
    training_data: InMemoryMolData | None = None
    validation_data: InMemoryMolData | None = None

    @property
    def training_dataset(self):
        return self.training_data

    @property
    def validation_dataset(self):
        return self.validation_data


def test_all_tasks_converged_matches_skip_reason():
    assert JourneyAdvanceResult(skip_training_reason="All the tasks converged").all_tasks_converged is True
    assert JourneyAdvanceResult().all_tasks_converged is False
    assert JourneyAdvanceResult(skip_training_reason="Early stop").all_tasks_converged is False


def test_fail_in_train_keeps_splitter_output_for_successful_tasks():
    strategy = DataAndSplittingStrategy(
        dataset=InMemoryMolData.create(available_properties=[PropertyInfo(name="energy", unit="eV")]),
        splitting_policy="fail_in_train",
        splitter="sequential",
        validation_set_fraction=0.1,
    )
    dataset = _make_dataset()

    split = strategy.split(
        dataset=dataset,
        journey_advance_results=JourneyAdvanceResult(),
        task_results=None,
        accuracy_checks_results=_make_accuracy_result(True),
    )

    assert np.array_equal(split, np.array([strategy.validation_key, strategy.training_key]))


def test_fail_in_train_forces_failed_tasks_to_training():
    strategy = DataAndSplittingStrategy(
        dataset=InMemoryMolData.create(available_properties=[PropertyInfo(name="energy", unit="eV")]),
        splitting_policy="fail_in_train",
        splitter="sequential",
        validation_set_fraction=0.1,
    )
    dataset = _make_dataset()

    split = strategy.split(
        dataset=dataset,
        journey_advance_results=JourneyAdvanceResult(),
        task_results=None,
        accuracy_checks_results=_make_accuracy_result(False),
    )

    assert np.array_equal(split, np.array([strategy.training_key, strategy.training_key]))


def test_import_data_decorates_all_rows_without_strict_zip_failure():
    strategy = DataAndSplittingStrategy(
        dataset=InMemoryMolData.create(available_properties=[PropertyInfo(name="energy", unit="eV")]),
    )

    strategy.import_data(_make_dataset(), split=strategy.training_key, imported=True)

    imported_rows = list(strategy.dataset)
    assert len(imported_rows) == 2
    assert all(row.metadata[strategy.split_key] == strategy.training_key for row in imported_rows)
    assert all(row.metadata[strategy.iteration_al_key] == -1 for row in imported_rows)


def test_import_data_from_engine_preserves_training_and_validation_splits():
    strategy = DataAndSplittingStrategy(
        dataset=InMemoryMolData.create(available_properties=[PropertyInfo(name="energy", unit="eV")]),
    )
    engine = _EngineWithDatasets(
        engine_id="engine",
        training_data=_make_dataset(task_id="training_task"),
        validation_data=_make_dataset(task_id="validation_task"),
    )

    strategy.import_data_from_engine(engine, imported=True)

    imported_rows = list(strategy.dataset)
    assert len(imported_rows) == 4
    assert sum(row.metadata[strategy.split_key] == strategy.training_key for row in imported_rows) == 2
    assert sum(row.metadata[strategy.split_key] == strategy.validation_key for row in imported_rows) == 2
    assert all(row.metadata[strategy.iteration_al_key] == -1 for row in imported_rows)


def _make_grouped_dataset(n_per_group: int = 6) -> InMemoryMolData:
    dataset = InMemoryMolData.create(available_properties=[PropertyInfo(name="energy", unit="eV")])
    base_atoms = Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    for task_id in ("go", "md"):
        for system_id in ("M0000", "M0001", "M0002"):
            for idx in range(n_per_group):
                dataset.add_system(
                    ChemDataEntry(
                        system=base_atoms.copy(),
                        properties={"energy": float(idx)},
                        metadata={"task_id": task_id, "system_id": system_id},
                    )
                )
    return dataset


def test_stratify_keys_puts_one_validation_frame_per_task_and_system():
    strategy = DataAndSplittingStrategy(
        stratify_keys=["task_id", "system_id"],
        validation_set_fraction=1,
        seed=123,
    )
    dataset = _make_grouped_dataset()

    split = strategy.split(dataset, None, None, None)

    groups: dict[tuple[str, str], list[str]] = {}
    for row, label in zip(dataset, split, strict=True):
        groups.setdefault((row.metadata["task_id"], row.metadata["system_id"]), []).append(str(label))
    assert len(groups) == 6
    for labels in groups.values():
        assert labels.count(strategy.validation_key) == 1
        assert labels.count(strategy.training_key) == 5


def test_stratify_keys_default_groups_by_task_only():
    strategy = DataAndSplittingStrategy(validation_set_fraction=1, seed=123)
    dataset = _make_grouped_dataset()

    split = strategy.split(dataset, None, None, None)

    assert list(split).count(strategy.validation_key) == 2
