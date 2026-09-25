from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from scm.moliterate import ConcreteInterfaces, load_dataset
from scm.moliterate.analysis import PairwiseDatasetMetrics, PairwiseResult
from tabulate import tabulate

from scm.active_learning.engines import ConcreteEngines
from scm.active_learning.logging.path_serializable_model import PathSerializableModel
from scm.active_learning.results import MLIPTrainerResults, SPLabellingResults
from scm.active_learning.tasks import SPLabellerData

DatasetInput = Union[str, Path, ConcreteInterfaces]
EngineInput = Union[str, Path, ConcreteEngines]


def _load_normalized_json(path: Union[str, Path]) -> dict:
    return PathSerializableModel._load_normalized_json(path)


def load_engine_from_json(path: Union[str, Path]) -> ConcreteEngines:
    payload = _load_normalized_json(path)
    try:
        return TypeAdapter(ConcreteEngines).validate_python(payload)
    except ValidationError:
        return TypeAdapter(MLIPTrainerResults).validate_python(payload).engine


def resolve_dataset(dataset: DatasetInput) -> ConcreteInterfaces:
    if isinstance(dataset, (str, Path)):
        return load_dataset(dataset)
    return dataset


def resolve_engine(start_engine: EngineInput) -> ConcreteEngines:
    if isinstance(start_engine, (str, Path)):
        return load_engine_from_json(start_engine)
    return start_engine


def summarize_accuracy_metrics(
    metrics: PairwiseDatasetMetrics,
    groupby_metadata: List[str],
) -> str:
    if len(metrics.settings) == 0:
        metrics_summary = "no metric settings"
    else:
        metrics_summary = ", ".join(f"{setting.property}:{setting.metric}" for setting in metrics.settings)
    return f"groupby_metadata={groupby_metadata} | metrics={metrics_summary}"


def run_pairwise_accuracy(
    start_engine: ConcreteEngines,
    sp_task: SPLabellerData,
    metrics: PairwiseDatasetMetrics,
    groupby_metadata: List[str],
) -> Tuple[SPLabellingResults, List[PairwiseResult]]:
    sp_labelling_results = sp_task.run(engine=start_engine)
    if len(groupby_metadata) == 0:
        pairwise_results = metrics.compare_two(sp_labelling_results.task.dataset, sp_labelling_results.dataset)
    else:
        pairwise_results = metrics.compare_two_grouped(
            sp_labelling_results.task.dataset,
            sp_labelling_results.dataset,
            metadata_grouping=groupby_metadata,
        )
    return sp_labelling_results, pairwise_results


def dump_accuracy_results(
    accuracy_results: List[PairwiseResult],
    output_path: Union[str, Path],
) -> None:
    text = json.dumps([result.model_dump(mode="json") for result in accuracy_results], indent=2)
    output_path = Path(output_path)
    output_path.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote {len(accuracy_results)} accuracy results to {output_path}")


def _pairwise_result_compact_dict(result: PairwiseResult) -> dict:
    row = result.model_dump(exclude={"lower_is_better"})
    if row.pop("per_n_atoms"):
        row["units"] += "/NAtoms"
    return row


def accuracy_results_table(accuracy_results: List[PairwiseResult]) -> str:
    if len(accuracy_results) == 0:
        return ""
    return tabulate([_pairwise_result_compact_dict(result) for result in accuracy_results], headers="keys")


class AccuracyRun(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    dataset: Optional[DatasetInput] = None
    start_engine: EngineInput
    sp_task: SPLabellerData
    metrics: PairwiseDatasetMetrics
    groupby_metadata: List[str] = []
    output_path: Optional[Union[str, Path]] = None
    verbose: bool = False

    def run_with_labelling_results(self) -> Tuple[SPLabellingResults, List[PairwiseResult]]:
        resolved_dataset = resolve_dataset(self.dataset) if self.dataset is not None else self.sp_task.dataset
        resolved_start_engine = resolve_engine(self.start_engine)
        self.sp_task.dataset = resolved_dataset

        if self.verbose:
            print(f"Dataset entries: {len(resolved_dataset)}")
            print(f"Start engine: {resolved_start_engine.type}:{resolved_start_engine.engine_id}")
            print(f"SP task: {self.sp_task.task_id} | out_path={self.sp_task.sp_labeller.out_path}")
            print(f"Accuracy metrics: {summarize_accuracy_metrics(self.metrics, self.groupby_metadata)}")

        sp_labelling_results, accuracy_results = run_pairwise_accuracy(
            start_engine=resolved_start_engine,
            sp_task=self.sp_task,
            metrics=self.metrics,
            groupby_metadata=self.groupby_metadata,
        )

        if self.verbose:
            accuracy_table = accuracy_results_table(accuracy_results)
            if accuracy_table:
                print("Accuracy Results")
                print(accuracy_table)

        if self.output_path is not None:
            dump_accuracy_results(accuracy_results, output_path=self.output_path)
        return sp_labelling_results, accuracy_results

    def run(self) -> List[PairwiseResult]:
        _, accuracy_results = self.run_with_labelling_results()
        return accuracy_results


__all__ = [
    "AccuracyRun",
    "DatasetInput",
    "EngineInput",
    "accuracy_results_table",
    "dump_accuracy_results",
    "load_engine_from_json",
    "resolve_dataset",
    "resolve_engine",
    "run_pairwise_accuracy",
    "summarize_accuracy_metrics",
]
