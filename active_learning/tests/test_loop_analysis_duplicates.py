from __future__ import annotations

from types import SimpleNamespace

import pytest
from ase import Atoms
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.interfaces.in_memory import InMemoryMolData

from scm.active_learning.loop.loop_analysis import ActiveLearningAnalysis, ALPlotter

pytest.importorskip("scipy")


def _analysis_from_dataset(dataset):
    loop = SimpleNamespace(splitting=SimpleNamespace(dataset=dataset))
    return ActiveLearningAnalysis.model_construct(loop=loop)


def _duplicate_dataset(with_iteration_metadata: bool = True):
    dataset = InMemoryMolData.create(available_properties=[])
    rows = [
        ChemDataEntry(
            system=Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            metadata={"iteration_al": 0} if with_iteration_metadata else {},
        ),
        ChemDataEntry(
            system=Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            metadata={"iteration_al": 0} if with_iteration_metadata else {},
        ),
        ChemDataEntry(
            system=Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            metadata={"iteration_al": 1} if with_iteration_metadata else {},
        ),
        ChemDataEntry(
            system=Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            metadata={"iteration_al": 1} if with_iteration_metadata else {},
        ),
        ChemDataEntry(
            system=Atoms(numbers=[1, 1], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
            metadata={"iteration_al": 1} if with_iteration_metadata else {},
        ),
        ChemDataEntry(
            system=Atoms(numbers=[2], positions=[[1.0, 1.0, 1.0]]),
            metadata={"iteration_al": 1} if with_iteration_metadata else {},
        ),
    ]
    dataset.add_systems(rows)
    return dataset


def test_collect_duplicates_reports_default_overall_and_metadata_groups():
    analysis = _analysis_from_dataset(_duplicate_dataset())

    rows = list(analysis.collect_duplicates())

    assert len(rows) == 4

    overall = rows[0]
    assert overall == {
        "scope": "overall",
        "group": "all",
        "tot_frames": 6,
        "n_frames_unique": 3,
        "n_removed": 3,
    }

    assert rows[1] == {
        "scope": "dataset",
        "group": "None",
        "tot_frames": 6,
        "n_frames_unique": 3,
        "n_removed": 3,
    }

    per_iteration = {row["group"]: row for row in rows[2:]}
    assert per_iteration["None%0"] == {
        "scope": "dataset%iteration_al",
        "group": "None%0",
        "tot_frames": 2,
        "n_frames_unique": 1,
        "n_removed": 1,
    }
    assert per_iteration["None%1"] == {
        "scope": "dataset%iteration_al",
        "group": "None%1",
        "tot_frames": 4,
        "n_frames_unique": 2,
        "n_removed": 2,
    }


def test_collect_duplicates_respects_custom_group_by_metadata():
    analysis = _analysis_from_dataset(_duplicate_dataset())

    rows = list(analysis.collect_duplicates(group_by_metadata=(("iteration_al",),)))

    assert [(row["scope"], row["group"]) for row in rows] == [
        ("overall", "all"),
        ("iteration_al", "0"),
        ("iteration_al", "1"),
    ]


def test_collect_duplicates_without_metadata_uses_none_groups():
    analysis = _analysis_from_dataset(_duplicate_dataset(with_iteration_metadata=False))

    rows = list(analysis.collect_duplicates())

    assert len(rows) == 3
    assert rows[0] == {
        "scope": "overall",
        "group": "all",
        "tot_frames": 6,
        "n_frames_unique": 3,
        "n_removed": 3,
    }
    assert rows[1] == {
        "scope": "dataset",
        "group": "None",
        "tot_frames": 6,
        "n_frames_unique": 3,
        "n_removed": 3,
    }
    assert rows[2] == {
        "scope": "dataset%iteration_al",
        "group": "None%None",
        "tot_frames": 6,
        "n_frames_unique": 3,
        "n_removed": 3,
    }


def test_view_duplicates_contains_expected_headers():
    analysis = _analysis_from_dataset(_duplicate_dataset())

    table = analysis.view_duplicates()

    assert "scope" in table
    assert "tot_frames" in table
    assert "n_frames_unique" in table
    assert "n_removed" in table


def test_accuracy_table_rejects_missing_accuracy_data():
    pytest.importorskip("pandas")
    pytest.importorskip("seaborn")
    analysis = SimpleNamespace(collect_table=lambda _option: iter(()))

    with pytest.raises(AssertionError, match="No accuracy data"):
        ALPlotter(analysis).accuracy_table()


def test_train_validation_frame_data_includes_cumulative_totals():
    pytest.importorskip("pandas")
    analysis = SimpleNamespace(
        collect_train_validation_frames=lambda: iter(
            [
                {
                    "scope": "iteration_al",
                    "group": "0",
                    "n_frames_training": 8,
                    "n_frames_validation": 2,
                    "tot_frames": 10,
                },
                {
                    "scope": "iteration_al",
                    "group": "1",
                    "n_frames_training": 7,
                    "n_frames_validation": 3,
                    "tot_frames": 10,
                },
            ]
        )
    )

    frame_data = ALPlotter(analysis)._train_validation_frames_dataframe()

    assert frame_data["total_frames_training"].tolist() == [8, 15]
    assert frame_data["total_frames_validation"].tolist() == [2, 5]
    assert frame_data["total_frames"].tolist() == [10, 20]
