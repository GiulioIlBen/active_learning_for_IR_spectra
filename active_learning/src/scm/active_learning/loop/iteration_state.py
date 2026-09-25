from __future__ import annotations

from contextlib import contextmanager
from datetime import timedelta
from time import perf_counter
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Union

from pydantic import BaseModel
from scm.moliterate import ConcreteInterfaces
from scm.moliterate.analysis import PairwiseDatasetMetrics
from scm.moliterate.filters import UnionFilters
from scm.moliterate.transforms import MetadataAddTransform

from scm.active_learning.checker_getters.checkers import SPChecker
from scm.active_learning.engines import ConcreteEngines
from scm.active_learning.journey_scheduler import JourneyScheduler
from scm.active_learning.logging import log_level
from scm.active_learning.loop.iteration_phase import IterPhase
from scm.active_learning.loop.trainer_check import (
    TrainerChecker,
    TrainerCheckResult,
)
from scm.active_learning.mlip import (
    ConcreteMLIPTrainer,
    DataAndSplittingStrategy,
)
from scm.active_learning.results import (
    CollectionCheckGetResults,
    EngineCheckResult,
    JourneyAdvanceResult,
    MLIPTrainerResults,
    SPLabellingResults,
)
from scm.active_learning.tasks import SPLabeller, SPLabellerData


@contextmanager
def iteration_logs(s: IterationState, phase: IterPhase) -> Iterator[Callable[[str], None]]:
    start = perf_counter()
    skipped_reason: Optional[str] = None

    def set_skipped_reason(reason: str) -> None:
        nonlocal skipped_reason
        skipped_reason = reason

    try:
        yield set_skipped_reason
    finally:
        dtime = timedelta(seconds=perf_counter() - start)

        total_seconds = dtime.total_seconds()
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds - (hours * 3600) - (minutes * 60)
        dtime_str = f"{hours}:{minutes:02}:{seconds:04.1f}"

        s.timings[phase] = dtime
        info = "Completed"
        level = "INFO"
        if skipped_reason is not None:
            info = "Skipped: " + skipped_reason
            level = "SUCCESS"
        log_level(
            "iAL:{iteration_al:02} | {fname:<15} | Took: {dtime} | {info} ",
            level=level,
            depth=3,
            iteration_al=s.iteration_al,
            fname=phase.value,
            dtime=dtime_str,
            info=info,
        )


class IterationState(BaseModel):
    iteration_al: int
    start_engine: Optional[ConcreteEngines] = None
    task_results: Optional[CollectionCheckGetResults] = None
    ds_pre_filters: Optional[ConcreteInterfaces] = None
    labelled: Optional[SPLabellingResults] = None
    ds_post_filters: Optional[ConcreteInterfaces] = None
    accuracy_checks_results: Optional[List[EngineCheckResult]] = None
    journey_advance_results: Optional[JourneyAdvanceResult] = None
    train_results: Optional[MLIPTrainerResults] = None
    trainer_check_result: Optional[TrainerCheckResult] = None
    skip: Dict[IterPhase, str] = {}
    timings: Dict[IterPhase, timedelta] = {}
    al_finished_reason: Optional[Union[str, Literal["CONVERGED"]]] = None

    @property
    def al_finished(self):
        """this property controls if the loop stops"""
        return self.al_finished_reason is not None

    def run_task(self, journey: JourneyScheduler):
        with iteration_logs(self, IterPhase.TASK) as set_skipped_reason:
            if IterPhase.TASK in self.skip:
                reason = self.skip[IterPhase.TASK]
                set_skipped_reason(reason)
                return reason
            if self.start_engine is None:
                raise ValueError("Something wrong happen here...")
            self.task_results = journey.run_current_batch(engine=self.start_engine)

    def label(
        self,
        pre_filter: Optional[UnionFilters],
        labeller: SPLabeller,
        labeller_engine: ConcreteEngines,
        post_filter: Optional[UnionFilters],
    ):
        with iteration_logs(self, IterPhase.LABEL) as set_skipped_reason:
            if IterPhase.LABEL in self.skip:
                reason = self.skip[IterPhase.LABEL]
                set_skipped_reason(reason)
                return reason
            if self.task_results is None:
                raise ValueError("Something wrong happen here...")
            dataset = self.task_results.get_final_dataset()
            if len(dataset) == 0:
                raise ValueError("Something wrong happen here...")
            # TODO make pre_filter aware of the training and validation set: this should be implemented in moliterate!
            self.ds_pre_filters = pre_filter(dataset) if pre_filter is not None else dataset
            self.labelled = SPLabellerData(
                task_id="=labelling=", sp_labeller=labeller, dataset=self.ds_pre_filters
            ).run(engine=labeller_engine)
            failed_simulations = self.labelled.failed_simulations
            n_simulations = len(self.ds_pre_filters)
            if failed_simulations and len(failed_simulations) == n_simulations:
                failures = "; ".join(f"frame {index}: {error}" for index, error in sorted(failed_simulations.items()))
                raise RuntimeError(
                    f"All reference-engine single-point simulations failed "
                    f"({len(failed_simulations)}/{n_simulations}): {failures}"
                )
            successful_indices = [
                index for index in range(len(self.labelled.dataset)) if index not in failed_simulations
            ]
            successfully_labelled = self.labelled.dataset.subset(indices=successful_indices)
            self.ds_post_filters = (
                post_filter(successfully_labelled) if post_filter is not None else successfully_labelled
            )

            log_level(
                (
                    "Labelling reducing N data: "
                    "{ndata_initial} -(pre)-> {ndata_pre} -(label)-> {ndata_label} -(post)-> {ndata_post}"
                ),
                level="DEBUG",
                ndata_initial=len(dataset),
                ndata_pre=len(self.ds_pre_filters),
                ndata_label=len(successfully_labelled),
                ndata_post=len(self.ds_post_filters),
            )

    def accuracy(self, labeller: SPLabeller, accuracy_checker: PairwiseDatasetMetrics):
        with iteration_logs(self, IterPhase.ACCURACY) as set_skipped_reason:
            if IterPhase.ACCURACY in self.skip:
                reason = self.skip[IterPhase.ACCURACY]
                set_skipped_reason(reason)
                return reason
            if self.ds_post_filters is None or self.start_engine is None:
                raise ValueError("Something wrong happen here...")
            # Model labelling
            sp_labelling_results = SPLabellerData(
                task_id="=Model=", sp_labeller=labeller, dataset=self.ds_post_filters
            ).run(engine=self.start_engine)
            # Ref-Model single point comparison
            self.accuracy_checks_results = SPChecker(
                checker_id=f"accuracy_checker[{self.iteration_al:02}]",
                metrics=accuracy_checker,
                n_failures_allowed=0,
                groupby_metadata="retain_previous",
                collect_ids_from_groupinfo=False,  # it must be False, so the information is coming from previous steps
            ).run(
                result=sp_labelling_results,
            )

    def convergence(self, journey: JourneyScheduler):
        with iteration_logs(self, IterPhase.CONVERGENCE) as set_skipped_reason:
            if IterPhase.CONVERGENCE in self.skip:
                reason = self.skip[IterPhase.CONVERGENCE]
                set_skipped_reason(reason)
                return reason
            if (self.task_results is None) or (self.accuracy_checks_results is None):
                raise ValueError("Something wrong happen here...")
            # Inactivate all valid and accurate tasks
            self.journey_advance_results = journey.advance(
                validity_results=self.task_results,
                accuracy_results=self.accuracy_checks_results,
            )
            if self.journey_advance_results.skip_training:
                self.skip[IterPhase.TRAIN] = self.journey_advance_results.skip_training_reason
            if journey.is_finished():
                r = journey.finished_reason()
                self.al_finished_reason = r
                if IterPhase.TRAIN in self.skip:
                    self.skip[IterPhase.TRAIN] += f" - AL {r}"
                else:
                    self.skip[IterPhase.TRAIN] = f"AL {r}"
                self.skip[IterPhase.SPLIT_AND_ADD] = f"AL {r}"

    def split_and_add(self, splitting: DataAndSplittingStrategy):
        with iteration_logs(self, IterPhase.SPLIT_AND_ADD) as set_skipped_reason:
            if IterPhase.SPLIT_AND_ADD in self.skip:
                reason = self.skip[IterPhase.SPLIT_AND_ADD]
                set_skipped_reason(reason)
                return reason
            if self.ds_post_filters is None:
                raise ValueError("Something wrong happen here...")
            ds_to_add = self.ds_post_filters
            ds_to_add.transforms.append(
                MetadataAddTransform(
                    key_values={"iteration_al": self.iteration_al},
                    overwrite=True,
                )
            )
            splitting.split_and_add(
                dataset=ds_to_add,
                journey_advance_results=self.journey_advance_results,
                task_results=self.task_results,
                accuracy_checks_results=self.accuracy_checks_results,
            )

    def train(self, splitting: DataAndSplittingStrategy, mlip_trainer: ConcreteMLIPTrainer):
        with iteration_logs(self, IterPhase.TRAIN) as set_skipped_reason:
            if IterPhase.TRAIN in self.skip:
                reason = self.skip[IterPhase.TRAIN]
                set_skipped_reason(reason)
                return reason
            if len(splitting.dataset) == 0:
                raise ValueError("Something went wrong here...")
            engine_id = f"{mlip_trainer.ml_name}{self.iteration_al:02}"
            if self.start_engine is not None and self.start_engine.is_finetunable():
                self.train_results = mlip_trainer.finetune(
                    engine=mlip_trainer.validate_engine(self.start_engine),
                    data_split=splitting,
                    engine_id=engine_id,
                )
            else:
                self.train_results = mlip_trainer.train(data_split=splitting, engine_id=engine_id)

    def train_check(
        self,
        splitting: DataAndSplittingStrategy,
        labeller: SPLabeller,
        trainer_checker: Optional[TrainerChecker],
    ):
        with iteration_logs(self, IterPhase.TRAIN_CHECK) as set_skipped_reason:
            if IterPhase.TRAIN_CHECK in self.skip:
                reason = self.skip[IterPhase.TRAIN_CHECK]
                set_skipped_reason(reason)
                return reason
            if trainer_checker is None:
                set_skipped_reason("trainer_checker disabled")
                return "trainer_checker disabled"
            if IterPhase.TRAIN in self.skip:
                reason = self.skip[IterPhase.TRAIN]
                set_skipped_reason(reason)
                return reason
            if self.train_results is None:
                set_skipped_reason("No train results")
                return "No train results"
            self.trainer_check_result = trainer_checker.run(
                dataset=splitting.dataset,
                engine=self.train_results.engine,
                labeller=labeller,
            )

    #########################################################################
    ############################ Analysis Methods ###########################
    #########################################################################

    def summary(self):
        ret1 = [{}]
        if self.task_results is not None:
            ret1 = self.task_results.summary()

        n_acc = 0
        n_acc_ok = 0
        if self.accuracy_checks_results is not None:
            n_acc = len(self.accuracy_checks_results)
            n_acc_ok = sum(check.success == "OK" for check in self.accuracy_checks_results)

        ret3 = {}
        if self.trainer_check_result is not None:
            ret3 = self.trainer_check_result.summary()
        return {**list(ret1)[0], "n_accuracy_checks": n_acc, "n_accuracy_success": n_acc_ok, **ret3}

    def dump_timings(self, add_skip: bool = True):
        ret = {}
        for k, v in self.timings.items():
            if k in self.skip and add_skip:
                ret[f"{k}[min]"] = self.skip[k]
            else:
                ret[f"{k}[min]"] = self._as_min(v)
        return ret

    @staticmethod
    def _as_min(value: Optional[Any]) -> float:
        if value is not None:
            ret = float(value.total_seconds()) // 60
            if ret == 0:
                return float(value.total_seconds()) / 60
            return ret
        return -1.0

    def dump_iteration_al(self):
        return {"iteration_al": self.iteration_al}

    def iter_dump_validities_summary(self, **kwargs):
        if self.task_results is None:
            return None
        return self.task_results.summary(**kwargs)

    def iter_dump_validities(self):
        if self.task_results is None:
            return None
        for r in self.task_results.validity_get_results:
            for x in r.validity_results:
                yield x.model_dump()

    def iter_dump_accuracies(self):
        if self.accuracy_checks_results is None:
            return None
        for x in self.accuracy_checks_results:
            yield x.model_dump()

    def iter_dump_getters(self):
        if self.task_results is None:
            return None
        for x in self.task_results.validity_get_results:
            getter = x.getter_results
            yield {
                **getter.model_dump(exclude={"dataset"}),
                "n_data": len(getter.dataset),
            }

    def iter_dump_train(self):
        train_res = self.train_results
        lens = []
        if train_res is not None:
            lens = [len(x) for x in train_res.log_metrics.values()]
        if len(lens) > 0:
            assert all(x == lens[0] for x in lens)
            for i in range(lens[0]):
                yield {
                    "trained": True,
                    "type": train_res.engine.type,
                    **{k: train_res.log_metrics[k][i] for k in train_res.log_metrics},
                }
        else:
            yield {
                "trained": train_res is not None,
                "type": train_res.engine.type if train_res is not None else None,
            }

    def iter_dump_train_check(self, compact: bool = False):
        train_res_c = self.trainer_check_result
        if train_res_c is None:
            return None
        for x in train_res_c.accuracy_checks_results:
            yield {
                **(x.compact_dump(exclude={"group_info"}) if compact else x.model_dump(exclude={"group_info"})),
                **x.group_info.copy(),
            }
