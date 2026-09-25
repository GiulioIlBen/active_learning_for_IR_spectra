from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ase import Atoms
from scm.moliterate import PropertyInfo
from scm.moliterate.analysis import PairwiseDatasetMetrics, PairwiseResult
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.interfaces.in_memory import InMemoryMolData

from scm.active_learning.loop import AccuracyRun
from scm.active_learning.loop.iteration_state import IterationState
from scm.active_learning.tasks import SPLabeller


class DummyTask:
    def __init__(self, dataset, labelled_dataset=None, out_path: str = "labelled.db"):
        self.dataset = dataset
        self.labelled_dataset = labelled_dataset if labelled_dataset is not None else dataset
        self.task_id = "sp-task"
        self.sp_labeller = SimpleNamespace(out_path=out_path)
        self.calls = []

    def run(self, engine):
        self.calls.append(engine)
        return SimpleNamespace(
            task=SimpleNamespace(dataset=self.dataset),
            dataset=self.labelled_dataset,
            failed_simulations={},
        )


class DummyMetrics:
    def __init__(self):
        self.settings = []
        self.compare_two_calls = []
        self.compare_two_grouped_calls = []

    @staticmethod
    def _results():
        return [
            PairwiseResult(
                property="energy",
                metric="mae",
                units="eV",
                value=0.5,
                target=1.0,
                n_entries=2,
                success="OK",
            )
        ]

    def compare_two(self, dataset_ref, dataset_cmp):
        self.compare_two_calls.append((dataset_ref, dataset_cmp))
        return self._results()

    def compare_two_grouped(self, dataset_ref, dataset_cmp, metadata_grouping):
        self.compare_two_grouped_calls.append((dataset_ref, dataset_cmp, metadata_grouping))
        return self._results()


def test_accuracy_run_uses_resolved_inputs_and_writes_results(tmp_path, monkeypatch):
    dataset = ["frame-1", "frame-2"]
    labelled_dataset = ["labelled-1", "labelled-2"]
    resolved_engine = SimpleNamespace(type="dummy", engine_id="engine-1")
    task = DummyTask(dataset=["old"], labelled_dataset=labelled_dataset)
    metrics = DummyMetrics()
    output_path = tmp_path / "accuracy.json"

    monkeypatch.setattr("scm.active_learning.loop.accuracy_run.resolve_dataset", lambda value: dataset)
    monkeypatch.setattr("scm.active_learning.loop.accuracy_run.resolve_engine", lambda value: resolved_engine)

    run = AccuracyRun.model_construct(
        dataset=Path("dataset.db"),
        start_engine=Path("engine.json"),
        sp_task=task,
        metrics=metrics,
        groupby_metadata=[],
        output_path=output_path,
    )

    results = run.run()

    assert task.dataset == dataset
    assert task.calls == [resolved_engine]
    assert metrics.compare_two_calls == [(dataset, labelled_dataset)]
    assert len(results) == 1
    assert json.loads(output_path.read_text(encoding="utf-8")) == [results[0].model_dump(mode="json")]


def test_accuracy_run_falls_back_to_task_dataset_when_dataset_not_provided(monkeypatch, capsys):
    dataset = ["frame-1"]
    resolved_engine = SimpleNamespace(type="dummy", engine_id="engine-2")
    task = DummyTask(dataset=dataset, out_path="task.db")
    metrics = DummyMetrics()

    monkeypatch.setattr("scm.active_learning.loop.accuracy_run.resolve_engine", lambda value: resolved_engine)

    run = AccuracyRun.model_construct(
        dataset=None,
        start_engine="engine.json",
        sp_task=task,
        metrics=metrics,
        groupby_metadata=["dataset"],
        output_path=None,
        verbose=True,
    )

    run.run()
    captured = capsys.readouterr()

    assert "Dataset entries: 1" in captured.out
    assert "groupby_metadata=['dataset']" in captured.out
    assert "Accuracy Results" in captured.out
    assert "energy" in captured.out
    assert metrics.compare_two_grouped_calls == [(dataset, dataset, ["dataset"])]
    assert task.calls == [resolved_engine]


def test_iteration_state_accuracy_uses_sp_checker_with_retain_previous(monkeypatch):
    props = [PropertyInfo(name="energy", unit="eV")]
    dataset = InMemoryMolData.create(available_properties=props)
    labeller = SPLabeller(properties=props)
    accuracy_checker = PairwiseDatasetMetrics(
        settings=[PairwiseDatasetMetrics.PropMetricEv(property="energy", metric="mae", target=1.0)]
    )
    start_engine = SimpleNamespace(type="dummy", engine_id="engine-3")
    delegated_results = [SimpleNamespace(success="OK")]
    calls = {}

    def fake_labelling_run(self, engine, **kwargs):
        calls["labelling_engine"] = engine
        return SimpleNamespace(
            task=self,
            engine=engine,
            dataset=dataset,
            failed_simulations={},
        )

    def fake_checker_run(self, result, **kwargs):
        calls["groupby_metadata"] = self.groupby_metadata
        calls["collect_ids_from_groupinfo"] = self.collect_ids_from_groupinfo
        calls["checker_id"] = self.checker_id
        calls["result"] = result
        return delegated_results

    monkeypatch.setattr("scm.active_learning.tasks.SPLabellerData.run", fake_labelling_run)
    monkeypatch.setattr("scm.active_learning.loop.iteration_state.SPChecker.run", fake_checker_run)

    state = IterationState.model_construct(
        iteration_al=3,
        start_engine=start_engine,
        ds_post_filters=dataset,
        skip={},
        timings={},
    )

    state.accuracy(labeller=labeller, accuracy_checker=accuracy_checker)

    assert state.accuracy_checks_results == delegated_results
    assert calls["labelling_engine"] is start_engine
    assert calls["groupby_metadata"] == ["engine_id", "system_id", "task_id", "checker_id"]
    assert calls["collect_ids_from_groupinfo"] is True
    assert calls["checker_id"] == "accuracy_checker[03]"
    assert calls["result"].dataset is dataset


def test_iteration_state_label_raises_when_all_reference_single_points_fail(monkeypatch):
    props = [PropertyInfo(name="energy", unit="eV")]
    dataset = InMemoryMolData.create(available_properties=props)
    dataset.add_system(ChemDataEntry(system=Atoms("H"), properties={"energy": 0.0}))
    labeller = SPLabeller(properties=props)
    reference_engine = SimpleNamespace(type="dummy", engine_id="reference")

    def fake_labelling_run(self, engine, **kwargs):
        return SimpleNamespace(
            task=self,
            engine=engine,
            dataset=dataset,
            failed_simulations={0: "reference calculation crashed"},
        )

    monkeypatch.setattr("scm.active_learning.tasks.SPLabellerData.run", fake_labelling_run)
    post_filter_called = False

    def post_filter(labelled_dataset):
        nonlocal post_filter_called
        post_filter_called = True
        return labelled_dataset

    state = IterationState.model_construct(
        iteration_al=3,
        task_results=SimpleNamespace(get_final_dataset=lambda: dataset),
        skip={},
        timings={},
    )

    with pytest.raises(RuntimeError, match=r"All reference-engine single-point simulations failed \(1/1\)") as exc_info:
        state.label(
            pre_filter=None,
            labeller=labeller,
            labeller_engine=reference_engine,
            post_filter=post_filter,
        )

    assert "frame 0: reference calculation crashed" in str(exc_info.value)
    assert post_filter_called is False
    assert state.ds_post_filters is None


def test_iteration_state_label_continues_when_only_some_reference_single_points_fail(monkeypatch):
    props = [PropertyInfo(name="energy", unit="eV")]
    dataset = InMemoryMolData.create(available_properties=props)
    dataset.add_system(ChemDataEntry(system=Atoms("H"), properties={"energy": 0.0}))
    dataset.add_system(ChemDataEntry(system=Atoms("He"), properties={"energy": 0.0}))
    labeller = SPLabeller(properties=props)
    reference_engine = SimpleNamespace(type="dummy", engine_id="reference")

    def fake_labelling_run(self, engine, **kwargs):
        return SimpleNamespace(
            task=self,
            engine=engine,
            dataset=dataset,
            failed_simulations={0: "reference calculation crashed"},
        )

    monkeypatch.setattr("scm.active_learning.tasks.SPLabellerData.run", fake_labelling_run)

    state = IterationState.model_construct(
        iteration_al=3,
        task_results=SimpleNamespace(get_final_dataset=lambda: dataset),
        skip={},
        timings={},
    )

    state.label(
        pre_filter=None,
        labeller=labeller,
        labeller_engine=reference_engine,
        post_filter=None,
    )

    assert state.labelled.failed_simulations == {0: "reference calculation crashed"}
    assert state.ds_post_filters is not None
    assert len(state.ds_post_filters) == 1
    assert state.ds_post_filters.get_row(0).atoms.symbols == "He"
