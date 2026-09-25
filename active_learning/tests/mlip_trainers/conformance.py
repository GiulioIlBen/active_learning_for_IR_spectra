from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pytest
from scm.moliterate import ChemDataSetFormat, PropertyInfo, load_dataset
from scm.moliterate.analysis import PairwiseDatasetMetrics
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.interfaces.in_memory import InMemoryMolData

from scm.active_learning.engines import PLAMSMolecule
from scm.active_learning.loop import AccuracyRun
from scm.active_learning.mlip import DataAndSplittingStrategy
from scm.active_learning.mlip.core import MLIPTrainer
from scm.active_learning.results import MLIPTrainerResults
from scm.active_learning.tasks import ASEMolecularDynamics, SPLabeller, SPLabellerData


@dataclass(frozen=True)
class MLIPTrainerCase:
    name: str
    make_trainer: Callable[[Path], MLIPTrainer]
    training_properties: Sequence[PropertyInfo]
    accuracy_properties: Sequence[PropertyInfo]
    loss_metric_names: Sequence[str]
    engine_id: str

    def make_data_split(self) -> DataAndSplittingStrategy:
        return DataAndSplittingStrategy(dataset=get_ds(self.training_properties))


def get_ds(
    properties: Sequence[PropertyInfo] | None = None,
    *,
    max_training_rows: int = 4,
    max_validation_rows: int = 2,
) -> InMemoryMolData:
    source = load_dataset(Path(__file__).with_name("train_validation_AL.db"))
    if properties is None:
        properties = source.out_properties

    dataset = InMemoryMolData.create(
        available_properties=list(properties),
        distance_unit=source.distance_unit,
    )
    selected_counts = {"training_set": 0, "validation_set": 0}
    for row in source:
        split = row.metadata.get("dataset")
        if split == "training_set":
            if selected_counts[split] >= max_training_rows:
                continue
            selected_counts[split] += 1
        elif split == "validation_set":
            if selected_counts[split] >= max_validation_rows:
                continue
            selected_counts[split] += 1
        else:
            continue

        dataset.add_system(
            ChemDataEntry(
                system=row.atoms.copy(),
                properties={prop.name: _get_property_value(row.properties, prop.name) for prop in properties},
                metadata=dict(row.metadata),
            )
        )
        if (
            selected_counts["training_set"] >= max_training_rows
            and selected_counts["validation_set"] >= max_validation_rows
        ):
            break
    return dataset


def _get_property_value(properties: dict, requested_name: str):
    for candidate_name in _property_aliases(requested_name):
        if candidate_name in properties:
            return properties[candidate_name]
    raise KeyError(f"None of {_property_aliases(requested_name)!r} found in row properties {sorted(properties)}")


def _property_aliases(name: str) -> tuple[str, ...]:
    return {
        "atomic_polar_tensor": ("atomic_polar_tensor", "dipole_gradients", "dipole_derivatives", "apt"),
        "dipole_gradients": ("dipole_gradients", "dipole_derivatives", "atomic_polar_tensor", "apt"),
        "dipole_moment": ("dipole_moment", "dipole"),
        "dipole": ("dipole", "dipole_moment"),
    }.get(name, (name,))


def assert_training_loss_shows_learning(result: MLIPTrainerResults, metric_names: Iterable[str]) -> None:
    for metric_name in metric_names:
        values = _metric_values(result, metric_name)
        assert len(values) >= 2, f"{metric_name} should contain at least two finite epochs"
        assert min(values[1:]) < values[0], f"{metric_name} did not improve: {values}"


def _metric_values(result: MLIPTrainerResults, metric_name: str) -> list[float]:
    values = result.log_metrics.get(metric_name, [])
    if values:
        return _finite_metric_values(values)
    learning_curve_csv = result.train_infos.get("learning_curve_csv")
    if isinstance(learning_curve_csv, str):
        return _finite_metric_values(_read_csv_metric_column(Path(learning_curve_csv), metric_name))
    training_log_path = result.train_infos.get("training_log_path")
    if isinstance(training_log_path, str):
        return _finite_metric_values(_read_training_log_metric(Path(training_log_path), metric_name))
    run_directory = result.train_infos.get("run_directory")
    if isinstance(run_directory, str):
        return _finite_metric_values(_read_mace_log_metric(Path(run_directory), metric_name))
    return []


def _read_csv_metric_column(path: Path, metric_name: str) -> list[str]:
    values = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            value = row.get(metric_name)
            if value not in (None, ""):
                values.append(value)
    return values


def _read_mace_log_metric(run_dir: Path, metric_name: str) -> list[str]:
    if metric_name != "loss":
        return []
    train_logs = sorted((run_dir / "logs").glob("*.log"))
    if len(train_logs) == 0:
        return []
    pattern = re.compile(r"(?:Initial|Epoch\s+\d+):.*?\bloss=(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    return [match.group("value") for match in pattern.finditer(train_logs[-1].read_text(encoding="utf-8"))]


def _read_training_log_metric(path: Path, metric_name: str) -> list[str]:
    if metric_name != "loss" or not path.exists():
        return []
    pattern = re.compile(r"\bloss\s*[:=]\s*(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", flags=re.IGNORECASE)
    return [match.group("value") for match in pattern.finditer(path.read_text(encoding="utf-8", errors="ignore"))]


def _finite_metric_values(values: Iterable[object]) -> list[float]:
    finite_values = []
    for value in values:
        numeric = float(value)
        if np.isfinite(numeric):
            finite_values.append(numeric)
    return finite_values


def assert_engine_runs_in_accuracy_run(
    result: MLIPTrainerResults,
    *,
    dataset: InMemoryMolData,
    properties: Sequence[PropertyInfo],
    work_dir: Path,
) -> None:
    labeller = SPLabeller(
        properties=list(properties),
        out_fmt=ChemDataSetFormat.IN_MEMORY,
        out_path=str(work_dir / "accuracy_labels.db"),
        if_out_path_exists="overwrite",
    )
    task = SPLabellerData(task_id="mlip-accuracy-single-point", sp_labeller=labeller, dataset=dataset)
    metrics = PairwiseDatasetMetrics(
        settings=[
            PairwiseDatasetMetrics.PropMetricEv(property=prop.name, metric="mae", target=100.0) for prop in properties
        ]
    )

    labelling_results, accuracy_results = AccuracyRun(
        dataset=dataset,
        start_engine=result.engine,
        sp_task=task,
        metrics=metrics,
    ).run_with_labelling_results()

    assert labelling_results.failed_simulations == {}, labelling_results.failed_simulations
    assert len(accuracy_results) == len(properties)
    assert {accuracy_result.success for accuracy_result in accuracy_results} == {"OK"}
    for prop in properties:
        assert prop.name in labelling_results.dataset.get_row(0).properties


def assert_engine_runs_md(result: MLIPTrainerResults, *, work_dir: Path) -> None:
    task = ASEMolecularDynamics(
        task_id="mlip-md-smoke",
        atomistic_system=_fixture_plams_system(),
        save_folder=str(work_dir / "md"),
        nsteps=2,
        timestep=0.5,
        samplingfreq=1,
        thermostat=None,
        temperature=None,
        exit_condition_freq=None,
        minimum_distance=None,
        error_threshold=None,
    )

    md_result = task.run(result.engine)

    assert Path(md_result.trajectory_path).is_file()
    assert md_result.executed_steps == 2
    assert md_result.written_frames >= 1
    assert md_result.stop_reasons == []


def _fixture_plams_system() -> PLAMSMolecule:
    ds = get_ds()
    return PLAMSMolecule.from_molecules(system_id="fixture-system", systems=ds[0].molecule)


def skip_if_missing_modules(*module_names: str) -> None:
    for module_name in module_names:
        pytest.importorskip(module_name)
