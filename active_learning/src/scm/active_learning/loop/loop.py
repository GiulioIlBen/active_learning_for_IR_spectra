from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, List, Literal, Optional, TextIO, Tuple, Type, TypeVar

from pydantic import ConfigDict, field_validator, model_validator
from scm.moliterate import ChemDataSetFormat, create_dataset
from scm.moliterate.analysis import PairwiseDatasetMetrics
from scm.moliterate.filters import UnionFilters
from scm.moliterate.utils.progress_bar import MOLITERATE_PROGRESS_BAR_CONFIG
from scm.plams import config as plams_config

from scm.active_learning.callbacks import (
    ALCallback,
    ALMinimalLogger,
    ALStateLogger,
    FolderManagerCallback,
    ImportStartEngineData,
    ManualStopper,
    SkipIf,
    UnionALCallbacks,
)
from scm.active_learning.engines import ConcreteEngines
from scm.active_learning.journey_scheduler import ConcreteJourneyScheduler
from scm.active_learning.logging import FormatOptions, LoguruLevel, add_sink, remove_sinks
from scm.active_learning.logging.loguru_calls_logs import log_level
from scm.active_learning.logging.path_serializable_model import PathSerializableModel
from scm.active_learning.loop.loop_result import IterationState, LoopResult
from scm.active_learning.loop.run_control import ActiveLearningRunControl
from scm.active_learning.loop.stoppable_counter import StoppableCounter
from scm.active_learning.loop.trainer_check import TrainerChecker
from scm.active_learning.mlip import (
    ConcreteMLIPTrainer,
    DataAndSplittingStrategy,
)
from scm.active_learning.tasks import SPLabeller

TCallback = TypeVar("TCallback", bound=ALCallback)


class ActiveLearningLoop(PathSerializableModel):
    model_config = ConfigDict(validate_assignment=True)

    tag: str = ""
    start_engine: Optional[ConcreteEngines] = None
    iterable_loop: StoppableCounter = StoppableCounter(stop=10)
    journey: ConcreteJourneyScheduler
    pre_filter: Optional[UnionFilters] = None  # can be incorporated in TaskCheckGetProtocol
    labeller: SPLabeller
    labeller_engine: ConcreteEngines
    post_filter: Optional[UnionFilters] = None  # can be incorporated in TaskCheckGetProtocol
    accuracy_checker: PairwiseDatasetMetrics = PairwiseDatasetMetrics()
    trainer_checker: Optional[TrainerChecker] = TrainerChecker()
    # with methods import_data and split_and_add and split
    splitting: DataAndSplittingStrategy = DataAndSplittingStrategy()
    mlip_trainer: ConcreteMLIPTrainer
    callbacks: List[UnionALCallbacks] = [
        FolderManagerCallback(),
        ALStateLogger(),
        ALMinimalLogger(),
        ImportStartEngineData(),
        SkipIf(),
        ManualStopper(),
    ]
    run_control: ActiveLearningRunControl = ActiveLearningRunControl()
    result: LoopResult = LoopResult()
    current_state: IterationState = IterationState(iteration_al=-1)

    ##########################
    ####### Validation #######
    ##########################

    @field_validator("iterable_loop", mode="before")
    @classmethod
    def _validate_iterable_loop(cls, value):
        if isinstance(value, int):
            return StoppableCounter(stop=value)
        return value

    @field_validator("labeller", mode="before")
    @classmethod
    def _validate_labeller(cls, value):
        if isinstance(value, list) and not isinstance(value, SPLabeller):
            return SPLabeller(properties=value)
        return value

    def model_post_init(self, __context: Any) -> None:
        if self.splitting.split_key == "__DefaultInPostInit__":
            Path("train_validation_AL.db").unlink(missing_ok=True)
            self.splitting = DataAndSplittingStrategy(
                dataset=create_dataset(
                    "train_validation_AL.db",
                    fmt=ChemDataSetFormat.ASE,
                    available_properties=self.labeller.properties,
                ),
                split_key="dataset",
                **self.splitting.model_dump(exclude={"dataset", "split_key"}),
            )
        self._clean_metrics_properties()
        self._sync_tagged_folder_manager_timestamp()

    def _clean_metrics_properties(self) -> None:
        labeller = getattr(self, "labeller", None)
        if labeller is None:
            return
        requested_properties = {prop.name for prop in labeller.properties}
        self._clean_metric_properties(self.accuracy_checker, requested_properties, "accuracy_checker.settings")
        if self.trainer_checker is not None:
            self._clean_metric_properties(
                self.trainer_checker.metrics,
                requested_properties,
                "trainer_checker.metrics.settings",
            )

    @staticmethod
    def _clean_metric_properties(
        metrics: PairwiseDatasetMetrics,
        requested_properties: set[str],
        settings_path: str,
    ) -> None:
        # TODO also check that if two metrics are the same between accuracy and training,
        # the training target should be always less or equal than the accuracy target.
        checked_properties = {setting.property for setting in metrics.settings}
        extra_properties = checked_properties - requested_properties
        missing_properties = requested_properties - checked_properties
        if extra_properties or missing_properties:
            parts = []
            if extra_properties:
                parts.append(f"ignored unlabelled metrics: {sorted(extra_properties)}")
            if missing_properties:
                parts.append(f"no metrics for labelled properties: {sorted(missing_properties)}")
            warnings.warn(
                f"ActiveLearningLoop property mismatch in {settings_path}: "
                f"{'; '.join(parts)}. Edit labeller.properties or {settings_path}.",
                stacklevel=2,
            )
        if extra_properties:
            metrics.settings = [setting for setting in metrics.settings if setting.property in requested_properties]

    def _sync_tagged_folder_manager_timestamp(self) -> None:
        if self.tag == "":
            return
        fms = self.query_callbacks(FolderManagerCallback)
        if len(fms) == 1:
            fm = fms[0]
            fm.root_dir.timestamp_format = f"%Y%m%d_%H%M%S-{self.tag}"

    @model_validator(mode="after")
    def _apply_tag_to_folder_manager(self) -> ActiveLearningLoop:
        self._sync_tagged_folder_manager_timestamp()
        return self

    ##########################
    #######  Callback  #######
    ##########################

    def _run_callbacks(
        self,
        method_name: Literal[
            "before_loop",
            "after_loop",
            "after_run_task",
            "start_iter_loop",
            "after_label",
            "after_accuracy",
            "after_convergence",
            "after_split_and_add",
            "after_train",
            "after_train_check",
        ],
        s: Optional[IterationState] = None,
    ) -> None:
        if not self.callbacks:
            return
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if callable(method):
                method(self, s)

    def query_callback(self, t: Type[TCallback], idx: int = 0) -> TCallback:
        return self.query_callbacks(t)[idx]

    def query_callbacks(self, t: Type[TCallback]) -> List[TCallback]:
        return [x for x in self.callbacks if isinstance(x, t)]

    def pop_callbacks(self, t: Type[TCallback]) -> List[TCallback]:
        popped: List[TCallback] = []
        kept: List[UnionALCallbacks] = []
        for callback in self.callbacks:
            if isinstance(callback, t):
                popped.append(callback)
            else:
                kept.append(callback)
        self.callbacks = kept
        return popped

    ##########################
    #######    Run     #######
    ##########################

    def run(self) -> ConcreteEngines:
        engine = self.run_control.start_engine(self)
        self.iterable_loop.start = self.run_control.start_iteration(self)
        self._run_callbacks("before_loop")
        iteration_al = -1
        for iteration_al in self.iterable_loop.run():
            iteration_state = self.run_iteration(
                engine=engine,
                iteration_al=iteration_al,
            )
            self.result.iterations.append(iteration_state)
            if iteration_state.train_results is not None:
                engine = iteration_state.train_results.engine
            if engine is None:
                raise ValueError("Something wrong happen here... There should be always an engine here")
            self.result.final_engine = engine
            last_reason = self.result.iterations[-1].al_finished_reason
            if last_reason is not None:
                self.iterable_loop.stop_next_iter(reason=last_reason)
        self.result.exit_message = (
            f"ActiveLearningLoop FINISHED! At iteration {iteration_al:02} because {self.iterable_loop.reason}"
        )
        self._run_callbacks("after_loop")
        log_level(
            self.result.exit_message,
            level="SUCCESS",
            depth=1,
        )
        if engine is None:
            raise ValueError("Something wrong happen here... There should be always an engine here")
        return engine

    def run_iteration(self, engine: Optional[ConcreteEngines], iteration_al: int) -> IterationState:
        self.current_state = IterationState(start_engine=engine, iteration_al=iteration_al)
        self._run_callbacks(method_name="start_iter_loop", s=self.current_state)
        self.current_state.run_task(self.journey)
        self._run_callbacks(method_name="after_run_task", s=self.current_state)
        self.current_state.label(self.pre_filter, self.labeller, self.labeller_engine, self.post_filter)
        self._run_callbacks(method_name="after_label", s=self.current_state)
        self.current_state.accuracy(self.labeller, self.accuracy_checker)
        self._run_callbacks(method_name="after_accuracy", s=self.current_state)
        self.current_state.convergence(self.journey)
        self._run_callbacks(method_name="after_convergence", s=self.current_state)
        self.current_state.split_and_add(self.splitting)
        self._run_callbacks(method_name="after_split_and_add", s=self.current_state)
        self.current_state.train(self.splitting, self.mlip_trainer)
        self._run_callbacks(method_name="after_train", s=self.current_state)
        self.current_state.train_check(self.splitting, self.labeller, self.trainer_checker)
        self._run_callbacks(method_name="after_train_check", s=self.current_state)
        return self.current_state

    @classmethod
    def logging_config(
        cls,
        clean_sinks: int | Literal["YES", "NO"] = "YES",
        new_sink: str | Path | TextIO | None = None,
        level: LoguruLevel = "INFO",
        format: FormatOptions = "simple",
        moliterate_on: bool | None = False,
        plams_log: Tuple[int, int, int] | None = (7, 5, 0),
    ):
        """
        Configure active_learning sinks (Loguru), Moliterate progress output, and PLAMS (csv,file,stdout) log levels.
        """
        if moliterate_on is not None:
            MOLITERATE_PROGRESS_BAR_CONFIG.disable = not moliterate_on

        if plams_log is not None:
            plams_config.log.csv = plams_log[0]
            plams_config.log.file = plams_log[1]
            plams_config.log.stdout = plams_log[2]

        if clean_sinks != "NO":
            remove_sinks(handler_id=clean_sinks if isinstance(clean_sinks, int) else None)
        if new_sink is not None:
            return add_sink(new_sink, level=level, format=format)

    ##########################
    #######  Post Run  #######
    ##########################

    @property
    def analysis(self):
        from scm.active_learning.loop.loop_analysis import ActiveLearningAnalysis

        return ActiveLearningAnalysis(loop=self)

    @property
    def results(self):
        # to be plams consistent
        return self.result
