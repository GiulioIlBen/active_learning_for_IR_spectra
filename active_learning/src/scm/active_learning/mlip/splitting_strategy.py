from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, Iterable, List, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator
from scm.moliterate import (
    ChemDataEntry,
    ChemDataSetFormat,
    ConcreteInterfaces,
    ConcreteInterfacesWriters,
    create_dataset,
)
from scm.moliterate.filters.conditions_filter import ConditionsFilter
from scm.moliterate.partition_criteria import (
    MetadataCriterion,
    SequentialSplitCriterion,
    StratifiedSplitCriterion,
    TreePartitionCriterion,
)

from scm.active_learning.engines import Engine
from scm.active_learning.logging import log_level
from scm.active_learning.results import (
    CollectionCheckGetResults,
    EngineCheckResult,
    JourneyAdvanceResult,
)


class CoreDataAndSplittingStrategy(BaseModel):
    @abstractmethod
    def import_data_from_engine(self, engine: Engine) -> None: ...

    @abstractmethod
    def split_and_add(
        self,
        dataset: ConcreteInterfaces,
        journey_advance_results: Optional[JourneyAdvanceResult],
        task_results: Optional[CollectionCheckGetResults],
        accuracy_checks_results: Optional[list[EngineCheckResult]],
    ): ...


# TODO implement a remove duplicates of new dataset to add compared to the already present one
class DataAndSplittingStrategy(CoreDataAndSplittingStrategy):
    # TBN: this dataset should already contains the properties names
    #      in the format required by the to_data_format converter
    dataset: ConcreteInterfacesWriters = Field(
        default_factory=lambda: create_dataset("__memory__", fmt=ChemDataSetFormat.IN_MEMORY)
    )
    splitting_policy: Literal["success_in_val", "fail_in_train", "splitter_off", "policy_off"] = Field(
        default="policy_off",
        description=(
            "policy refers to how frames generated from tasks are splitted."
            " success_in_val: frames from successful tasks go to validation;"
            " fail_in_train: frames from failed tasks stay in training;"
            " splitter_off: no splitter, only successful frames from tasks"
            " go to validation and rest to training; policy_off: use splitter only."
        ),
    )
    splitter: Literal["sequential", "random"] = "random"
    validation_set_fraction: float = 0.1
    seed: Optional[int] = None
    split_key: str = "dataset"
    training_key: str = "training_set"
    validation_key: str = "validation_set"
    task_id_key: str = "task_id"
    stratify_keys: Optional[List[str]] = Field(
        default=None,
        description=(
            "Metadata keys whose combined values define the groups the splitter stratifies over "
            "(e.g. ['task_id', 'getter_id', 'system_id']); None groups by task_id_key only."
            " A validation_set_fraction >= 1 is read as an absolute number of validation frames per group."
        ),
    )

    iteration_al_key: ClassVar[str] = "iteration_al"

    # stratified_splitter: StratifiedSplitCriterion= StratifiedSplitCriterion(
    #     stratify_criterion=MetadataCriterion(metadata_keys=["task_id"]),
    #     labelA="training_set",
    #     labelB="validation_set",
    #     labelB_size_per_cat= 0.1,
    #     seed=None,
    # )

    @field_validator("dataset")
    @classmethod
    def _validate_dataset_units(cls, value):
        for prop in value.out_properties:
            if prop.unit is None:
                raise ValueError(f"Units are required! Missing unit for output property: {prop.name}")
        return value

    @staticmethod
    def decorate_metadata(
        dataset: Iterable[ChemDataEntry],
        split: Iterable[str],
        split_key: str,
        strict: bool = True,
        imported=False,
    ):
        for row, s in zip(dataset, split, strict=strict):
            row.metadata[split_key] = s
            if imported:
                row.metadata[DataAndSplittingStrategy.iteration_al_key] = -1
            yield row

    def import_data_from_engine(self, engine: Engine, imported=True) -> None:
        if engine.training_dataset is not None:
            log_level("Importing training set = {Ndata}", Ndata=len(engine.training_dataset))
            self.import_data(engine.training_dataset, split=self.training_key, imported=imported)
        if engine.validation_dataset is not None:
            log_level(
                "Importing validation set = {Ndata}",
                Ndata=len(engine.validation_dataset),
            )
            self.import_data(engine.validation_dataset, split=self.validation_key, imported=imported)

    def import_data(self, dataset: Iterable[ChemDataEntry], split: str, imported=False) -> None:
        def inf_split():
            while True:
                yield split

        self.dataset.add_systems(
            self.decorate_metadata(dataset, inf_split(), self.split_key, strict=False, imported=imported)
        )

    @property
    def train_validation_filters(self):
        train_filter = ConditionsFilter(conditions=f"{self.split_key}={self.training_key}", collect_from="metadata")
        valid_filter = ConditionsFilter(
            conditions=f"{self.split_key}={self.validation_key}",
            collect_from="metadata",
        )
        return train_filter, valid_filter

    def get_training_and_validation_datasets(self):
        train_filter, valid_filter = self.train_validation_filters
        return train_filter(self.dataset), valid_filter(self.dataset)

    def split_and_add(
        self,
        dataset: ConcreteInterfaces,
        journey_advance_results: Optional[JourneyAdvanceResult],
        task_results: Optional[CollectionCheckGetResults],
        accuracy_checks_results: Optional[list[EngineCheckResult]],
    ):
        split = self.split(dataset, journey_advance_results, task_results, accuracy_checks_results)
        self.dataset.add_systems(
            self.decorate_metadata(dataset, split, self.split_key),
            total=len(dataset),
            desc="SplitAndAdd",
        )

    @property
    def random_splitter(self):
        """stratified random splitter"""
        return StratifiedSplitCriterion(
            stratify_criterion=self.stratify_criterion,
            labelA=self.training_key,
            labelB=self.validation_key,
            labelB_size_per_cat=self.validation_set_fraction,
            seed=self.seed,
        )

    @property
    def sequential_splitter(self):
        """stratified sequential splitter"""
        return TreePartitionCriterion(
            partition_criteria=[
                self.stratify_criterion,
                SequentialSplitCriterion(
                    labelA=self.training_key,
                    labelB=self.validation_key,
                    labelB_size=self.validation_set_fraction,
                ),
            ],
            concat_symbol="KeepLast",
        )

    @property
    def by_task_criterion(self):
        return MetadataCriterion(metadata_keys=[self.task_id_key])

    @property
    def stratify_criterion(self):
        if self.stratify_keys is None:
            return self.by_task_criterion
        return MetadataCriterion(metadata_keys=list(self.stratify_keys))

    @property
    def splitter_obj(self):
        if self.splitter == "random":
            return self.random_splitter
        elif self.splitter == "sequential":
            return self.sequential_splitter
        raise ValueError

    def split(
        self,
        dataset: ConcreteInterfaces,
        journey_advance_results: Optional[JourneyAdvanceResult],
        task_results: Optional[CollectionCheckGetResults],
        accuracy_checks_results: Optional[list[EngineCheckResult]],
    ) -> Iterable[str]:
        if (
            self.splitting_policy in ["splitter_off", "success_in_val"]
            and journey_advance_results is not None
            and journey_advance_results.all_tasks_converged
        ):
            return np.array([self.validation_key] * len(dataset))

        if self.splitting_policy == "splitter_off":
            string_length = np.max([len(self.validation_key), len(self.training_key)])
            default_ret = np.full(len(dataset), self.training_key, dtype=f"U{string_length}")
        else:
            default_ret = self.splitter_obj(dataset).flatten()

        if self.splitting_policy != "policy_off":
            task_ids_arr = self.by_task_criterion(dataset).flatten()
            task_ids = JourneyAdvanceResult.get_passed_tasks(
                task_results=task_results,
                accuracy_checks_results=accuracy_checks_results,
            )
            for tid, is_success in task_ids.items():
                if is_success and self.splitting_policy in [
                    "splitter_off",
                    "success_in_val",
                ]:
                    default_ret[task_ids_arr == tid] = self.validation_key
                elif not is_success and self.splitting_policy == "fail_in_train":
                    default_ret[task_ids_arr == tid] = self.training_key
        log_level(
            "DataAndSplittingStrategy split of N data={ndata}:\n{table}",
            level="DEBUG",
            ndata=len(dataset),
            table=default_ret,
            # ab_idxs=dataset.select_indices(), #absolute idxs are not that relevant... we need origin one!
        )
        return default_ret
