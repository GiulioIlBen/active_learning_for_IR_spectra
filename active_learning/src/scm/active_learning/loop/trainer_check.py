from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, field_validator
from scm.moliterate import ConcreteInterfaces
from scm.moliterate.analysis import PairwiseDatasetMetrics, PairwiseResult

from scm.active_learning.engines import ConcreteEngines
from scm.active_learning.tasks import SPLabeller, SPLabellerData


class TrainerChecker(BaseModel):
    type: Literal["TrainerChecker"] = "TrainerChecker"
    n_failures_allowed: int = 0
    groupby_metadata: List[str] = ["dataset"]
    metrics: PairwiseDatasetMetrics = PairwiseDatasetMetrics(
        settings=[
            PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="max_scaled_abs", target=0.1),
            PairwiseDatasetMetrics.PropMetricEv(property="forces", metric="mae", target=0.01),
            PairwiseDatasetMetrics.PropMetricEv(property="dipole", metric="mae", target=0.01),
            PairwiseDatasetMetrics.PropMetricEv(property="energy", metric="mae", per_n_atoms=True, target=0.01),
        ]
    )

    @field_validator("n_failures_allowed", mode="after")
    @classmethod
    def _not_implemented_error(cls, value: int):
        if value != 0:
            raise NotImplementedError(
                "The n_failures_allowed>0 should implement a filter "
                "for the sp_labelling_results.task.dataset that is not yet implemented"
            )
        return value

    def _sp_failure_rejection_reason(self, failed_simulations: dict[int, str]) -> str:
        if len(failed_simulations) <= self.n_failures_allowed or self.n_failures_allowed < 0:
            return ""
        return (
            f"TrainerChecker: {len(failed_simulations)} single-point simulation(s) failed; "
            f"allowed={self.n_failures_allowed}"
        )

    def run(
        self,
        dataset: ConcreteInterfaces,
        engine: ConcreteEngines,
        labeller: SPLabeller,
    ) -> "TrainerCheckResult":
        sp_task = SPLabellerData(
            task_id="=SPModel=",
            sp_labeller=labeller,
            dataset=dataset,
        )
        sp_labelling_results = sp_task.run(engine=engine)
        rejection_reason: str = self._sp_failure_rejection_reason(sp_labelling_results.failed_simulations)
        if len(rejection_reason) > 0:
            raise ValueError(rejection_reason)

        if len(self.groupby_metadata) == 0:
            accuracy_checks_results = self.metrics.compare_two(
                sp_labelling_results.task.dataset,
                sp_labelling_results.dataset,
            )
        else:
            accuracy_checks_results = self.metrics.compare_two_grouped(
                sp_labelling_results.task.dataset,
                sp_labelling_results.dataset,
                metadata_grouping=self.groupby_metadata,
            )
        return TrainerCheckResult(
            failed_simulations=sp_labelling_results.failed_simulations,
            accuracy_checks_results=accuracy_checks_results,
        )


class TrainerCheckResult(BaseModel):
    failed_simulations: Dict[int, str] = {}
    accuracy_checks_results: List[PairwiseResult]

    def summary(self):
        n_train = len(self.accuracy_checks_results)
        n_train_ok = sum(check.success == "OK" for check in self.accuracy_checks_results)
        return {
            "n_train_checks": n_train,
            "n_train_success": n_train_ok,
        }
