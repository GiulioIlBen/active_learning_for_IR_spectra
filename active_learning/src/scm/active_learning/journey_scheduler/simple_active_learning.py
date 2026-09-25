from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional, Type, Union

import numpy as np
import scm.plams as plams
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
)
from scm.plams import AMSJob, Settings
from tabulate import tabulate
from typing_extensions import Annotated

from scm.active_learning.checker_getters import ConcreteCheckerGetter
from scm.active_learning.checker_getters.checkers.ams_traj_checkers import (
    AMSTrajChecker,
)
from scm.active_learning.checker_getters.getters.ams_traj_getter import AMSTrajGetter
from scm.active_learning.engines import ConcreteEngines, PLAMSMolecule
from scm.active_learning.journey_scheduler.core import JourneyScheduler
from scm.active_learning.logging import log_level
from scm.active_learning.results import (
    CheckGetResults,
    CollectionCheckGetResults,
    EngineCheckResult,
    JourneyAdvanceResult,
)
from scm.active_learning.results.task.ams_task import AMSTaskResult
from scm.active_learning.results.task.ase_md import ASEMolecularDynamicsResult
from scm.active_learning.task_parallelization import AMSSerialStrategy, ConcreteParallelizationStrategy
from scm.active_learning.tasks import ConcreteTask
from scm.active_learning.tasks.ams_task import AMSMDTask, AMSTask
from scm.active_learning.tasks.ase_md import ASEMolecularDynamics


class RestartGluer(BaseModel):
    type: Literal["RestartGluer"] = "RestartGluer"

    def run(self, current_job: AMSJob, previous_job: Optional[AMSJob] = None):
        if previous_job:
            input_molecules = previous_job.results.get_input_molecules()
            n_extra = 0
            current_job.molecule = None
            current_job.settings.input.ams.LoadSystem = []
            for key, _ in input_molecules.items():
                if key == "":
                    section = "Molecule"
                else:
                    n_extra += 1
                    section = f"InputMolecule({n_extra})"
                current_job.settings.input.ams.LoadSystem.append(
                    Settings(_h=key, file=previous_job.results.rkfpath(), section=section)
                )

            current_job.settings.input.ams.moleculardynamics.Restart = previous_job.results.rkfpath()
            current_job.settings.input.ams.moleculardynamics.CopyRestartTrajectory = "Yes"


class RestartingAMSTask(AMSTask):
    """AMS task wrapper that reapplies SAL restart state during parallelized execution.

    `SimpleActiveLearningJourney` builds this task for AMS-based steps. When
    `AMSParallelStrategy.run_tasks()` prepares the job, it calls `prepare_job(...)` on this
    wrapper, which in turn applies the configured `gluer` using the stored `previous_job`.
    This keeps SAL restart chaining working while the actual execution stays delegated to
    `task_parallelization.run_tasks(...)`.
    """

    type: Literal["RestartingAMSTask"] = "RestartingAMSTask"
    gluer: Any = Field(default=None, exclude=True)
    previous_job: Any = Field(default=None, exclude=True)

    def prepare_job(self, current_job: AMSJob) -> None:
        """Glue the current AMS job to the previous successful SAL job when available."""
        run = getattr(self.gluer, "run", None)
        if callable(run):
            run(current_job, self.previous_job)


SamplingType = Union[int, List[int]]


class Steps(BaseModel):
    md_nsteps: int
    # `sampling` can be one value for all AL steps or a list per AL step.
    sampling: SamplingType = 1
    # If set, reduce sampling frequency when needed to collect at least this many frames.
    min_frames: Optional[int] = 30

    @field_validator("md_nsteps")
    @classmethod
    def validate_md_nsteps(cls, value: int) -> int:
        if value < 1:
            raise ValueError(f"MolecularDynamics%NSteps must be positive, found: {value}")
        return value

    @staticmethod
    def _read_md_nsteps(md_settings: Settings) -> int:
        md_nsteps = md_settings.get_nested(
            ("input", "ams", "moleculardynamics", "nsteps"),
            default=None,
        )
        if md_nsteps is None:
            md_nsteps = md_settings.input.ams.MolecularDynamics.get("NSteps", default=None)
        if md_nsteps is None:
            raise ValueError("Could not read MolecularDynamics%NSteps from md_settings")
        return int(md_nsteps)

    @staticmethod
    def _read_md_sampling(md_settings: Settings) -> int:
        sampling = md_settings.get_nested(
            ("input", "ams", "moleculardynamics", "trajectory", "samplingfreq"),
            default=None,
        )
        if sampling is None:
            md = md_settings.input.ams.MolecularDynamics
            traj = md.get("Trajectory", default=None)
            if traj is not None and hasattr(traj, "get"):
                sampling = traj.get("SamplingFreq", default=None)
        if sampling is None:
            sampling = 1
        return int(sampling)

    @classmethod
    def _common_from_md_settings_kwargs(
        cls,
        md_settings: Settings,
        *,
        sampling: Optional[SamplingType] = None,
        min_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        resolved_sampling: SamplingType = sampling if sampling is not None else cls._read_md_sampling(md_settings)
        return {
            "md_nsteps": cls._read_md_nsteps(md_settings),
            "sampling": resolved_sampling,
            "min_frames": min_frames,
        }

    @classmethod
    def _common_from_md_task_kwargs(
        cls,
        md_task: Union[AMSTask, AMSMDTask, ASEMolecularDynamics],
        *,
        sampling: Optional[SamplingType] = None,
        min_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        if isinstance(md_task, ASEMolecularDynamics):
            resolved_sampling: SamplingType = sampling if sampling is not None else int(md_task.samplingfreq)
            return {
                "md_nsteps": int(md_task.nsteps),
                "sampling": resolved_sampling,
                "min_frames": min_frames,
            }
        return cls._common_from_md_settings_kwargs(
            md_settings=md_task.settings,
            sampling=sampling,
            min_frames=min_frames,
        )

    def _normalize_cumulative(self, cumulative_nsteps_per_step: List[int]) -> List[int]:
        if not cumulative_nsteps_per_step:
            raise ValueError("At least one active-learning step is required")

        nsteps_per_step = np.diff(np.array([0] + list(cumulative_nsteps_per_step), dtype=np.int64))

        # Make sure the sum of steps exactly equals the original number of frames.
        nsteps_per_step[-1] += self.md_nsteps - int(np.sum(nsteps_per_step))
        ret = np.cumsum(np.array(nsteps_per_step, dtype=np.int64))
        return [int(x) for x in ret]

    def actual_steps(self) -> List[int]:
        cumulative = self.cumulative_steps()
        return [int(x) for x in np.diff(np.array([0] + cumulative, dtype=np.int64))]

    def effective_steps(self) -> List[int]:
        return self.actual_steps()

    def cumulative_steps(self) -> List[int]:
        raise NotImplementedError

    def _base_sampling_frequencies(self) -> List[int]:
        n_steps = len(self.cumulative_steps())

        if isinstance(self.sampling, int):
            if self.sampling < 1:
                raise ValueError(f"Sampling frequency must be positive, found: {self.sampling}")
            return [int(self.sampling)] * n_steps

        if len(self.sampling) != n_steps:
            raise ValueError(f"sampling list must have length {n_steps}, found {len(self.sampling)}")
        if any(x < 1 for x in self.sampling):
            raise ValueError(f"All sampling frequencies must be positive, found: {self.sampling}")
        return [int(x) for x in self.sampling]

    def sampling_frequencies(self) -> List[int]:
        freqs = self._base_sampling_frequencies()

        if self.min_frames is None:
            return freqs
        if self.min_frames < 1:
            raise ValueError(f"min_frames must be positive, found: {self.min_frames}")

        adjusted = []
        for nsteps, freq in zip(self.actual_steps(), freqs, strict=True):
            # set sampling freq to have at least `min_frames` frames
            num_frames = nsteps // freq
            if num_frames < self.min_frames:
                freq = max(1, nsteps // self.min_frames)
            adjusted.append(int(freq))

        return adjusted


class ListSteps(Steps):
    type: Literal["ListSteps"] = "ListSteps"
    md_nsteps: int = 1
    cumulative_values: List[int]

    def model_post_init(self, __context: Any) -> None:
        self.md_nsteps = self.cumulative_values[-1]

    @classmethod
    def from_md_settings(
        cls,
        md_settings: Settings,
        cumulative_values: List[int],
        *,
        sampling: Optional[SamplingType] = None,
        min_frames: Optional[int] = 30,
    ) -> "ListSteps":
        return cls(
            cumulative_values=cumulative_values,
            **cls._common_from_md_settings_kwargs(
                md_settings=md_settings,
                sampling=sampling,
                min_frames=min_frames,
            ),
        )

    @classmethod
    def from_md_task(
        cls,
        md_task: Union[AMSTask, AMSMDTask, ASEMolecularDynamics],
        cumulative_values: List[int],
        *,
        sampling: Optional[SamplingType] = None,
        min_frames: Optional[int] = 30,
    ) -> "ListSteps":
        return cls(
            cumulative_values=cumulative_values,
            **cls._common_from_md_task_kwargs(
                md_task=md_task,
                sampling=sampling,
                min_frames=min_frames,
            ),
        )

    def cumulative_steps(self) -> List[int]:
        cumulative_nsteps_per_step = self.cumulative_values or []
        cumulative_nsteps_per_step = sorted(set(cumulative_nsteps_per_step))
        if any(x <= 0 for x in cumulative_nsteps_per_step):
            raise ValueError(f"In cumulative_values all values must be positive, found: {cumulative_nsteps_per_step}")

        cumulative_nsteps_per_step = [int(x) for x in cumulative_nsteps_per_step if x < self.md_nsteps]
        cumulative_nsteps_per_step.append(self.md_nsteps)
        return self._normalize_cumulative(cumulative_nsteps_per_step)


class GeometricSteps(Steps):
    type: Literal["GeometricSteps"] = "GeometricSteps"
    start: int
    num_steps: int

    @classmethod
    def from_md_settings(
        cls,
        md_settings: Settings,
        start: int,
        num_steps: int,
        *,
        sampling: Optional[SamplingType] = None,
        min_frames: Optional[int] = 30,
    ) -> "GeometricSteps":
        return cls(
            start=start,
            num_steps=num_steps,
            **cls._common_from_md_settings_kwargs(
                md_settings=md_settings,
                sampling=sampling,
                min_frames=min_frames,
            ),
        )

    @classmethod
    def from_md_task(
        cls,
        md_task: Union[AMSTask, AMSMDTask, ASEMolecularDynamics],
        start: int,
        num_steps: int,
        *,
        sampling: Optional[SamplingType] = None,
        min_frames: Optional[int] = 30,
    ) -> "GeometricSteps":
        return cls(
            start=start,
            num_steps=num_steps,
            **cls._common_from_md_task_kwargs(
                md_task=md_task,
                sampling=sampling,
                min_frames=min_frames,
            ),
        )

    def cumulative_steps(self) -> List[int]:
        if self.num_steps < 1:
            raise ValueError(f"Number of active learning steps must be >=1, found {self.num_steps}")
        if self.start < 1:
            raise ValueError(f"GeometricSteps start must be positive, found: {self.start}")
        if self.start > self.md_nsteps:
            raise ValueError("GeometricSteps start must <= md_nsteps")

        cumulative_nsteps_per_step = np.geomspace(
            self.start,
            self.md_nsteps,
            self.num_steps,
            endpoint=True,
            dtype=np.int64,
        ).tolist()
        return self._normalize_cumulative([int(x) for x in cumulative_nsteps_per_step])


class LinearSteps(Steps):
    type: Literal["LinearSteps"] = "LinearSteps"
    start: int
    step_size: int

    @classmethod
    def from_md_settings(
        cls,
        md_settings: Settings,
        start: int,
        step_size: int,
        *,
        sampling: Optional[SamplingType] = None,
        min_frames: Optional[int] = 30,
    ) -> "LinearSteps":
        return cls(
            start=start,
            step_size=step_size,
            **cls._common_from_md_settings_kwargs(
                md_settings=md_settings,
                sampling=sampling,
                min_frames=min_frames,
            ),
        )

    @classmethod
    def from_md_task(
        cls,
        md_task: Union[AMSTask, AMSMDTask, ASEMolecularDynamics],
        start: int,
        step_size: int,
        *,
        sampling: Optional[SamplingType] = None,
        min_frames: Optional[int] = 30,
    ) -> "LinearSteps":
        return cls(
            start=start,
            step_size=step_size,
            **cls._common_from_md_task_kwargs(
                md_task=md_task,
                sampling=sampling,
                min_frames=min_frames,
            ),
        )

    def cumulative_steps(self) -> List[int]:
        if self.start < 1:
            raise ValueError(f"ActiveLearning%Steps%Linear%Start must be positive, found: {self.start}")
        if self.start > self.md_nsteps:
            raise ValueError("ActiveLearning%Steps%Linear%Start must <= MolecularDynamics%NSteps")
        if self.step_size < 1:
            raise ValueError(f"ActiveLearning%Steps%Linear%StepSize must be positive, found: {self.step_size}")

        cumulative_nsteps_per_step = np.arange(self.start, self.md_nsteps + 1, self.step_size).tolist()
        if cumulative_nsteps_per_step[-1] < self.md_nsteps:
            cumulative_nsteps_per_step.append(self.md_nsteps)

        return self._normalize_cumulative([int(x) for x in cumulative_nsteps_per_step])


class SALState(BaseModel):
    type: Literal["SALState"] = "SALState"
    step: int = 0
    n_steps_left: int = -1
    attempt: int = 0
    previous_job: Optional[Path] = None
    finished_msg: str = ""
    finished: bool = False
    success: bool = False


class SimpleActiveLearningJourney(JourneyScheduler):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["SimpleActiveLearningJourney"] = "SimpleActiveLearningJourney"
    task_parallelization: ConcreteParallelizationStrategy = AMSSerialStrategy()
    checker_getter: ConcreteCheckerGetter = AMSTrajChecker() + AMSTrajGetter()
    task: Union[AMSTask, AMSMDTask, ASEMolecularDynamics] = AMSMDTask(
        atomistic_system=PLAMSMolecule.from_molecules(
            system_id="H2O",
            systems=plams.Molecule(
                string="3\n\nO 0 0 0\nH 0.9584 0 0\nH -0.2396 0.9271 0",
                inputformat="xyz",
            ),
        ),
        nsteps=10_000,
        samplingfreq=100,
    )
    steps: Annotated[Union[LinearSteps, ListSteps, GeometricSteps], Field(discriminator="type")] = GeometricSteps(
        md_nsteps=10_000, start=10, num_steps=5, min_frames=30
    )
    max_attempts: Union[int, List[int]] = 3
    finish_loop_on_max_attempts: Annotated[
        bool,
        Field(description="If True, stop the whole SAL journey when a step reaches max attempts and still fails"),
    ] = True
    gluer: Annotated[Union[RestartGluer, None], Field(discriminator="type")] = RestartGluer()
    first_step_train: Annotated[
        bool,
        Field(description="Useful to force train when you do not start with a finetunable model"),
    ] = False
    history: List[SALState] = []
    finish_reason: str = "CONVERGED"
    idx_: int = 0
    attempt_idx_: int = 0
    previous_job_: Optional[AMSJob] = None
    _current_attempt_job: Optional[AMSJob] = PrivateAttr(default=None)

    LinearSteps: ClassVar[Type[LinearSteps]] = LinearSteps
    ListSteps: ClassVar[Type[ListSteps]] = ListSteps
    GeometricSteps: ClassVar[Type[GeometricSteps]] = GeometricSteps

    @field_serializer("previous_job_", check_fields=False)
    def serialize_previous_job_path(self, previous_job_: Optional[AMSJob]) -> Optional[str]:
        if previous_job_ is None:
            return None
        return previous_job_.path

    @field_validator("previous_job_", mode="before", check_fields=False)
    @classmethod
    def valid_previous_job(cls, value):
        if isinstance(value, str):
            return AMSJob.load_external(value)
        return value

    def restart_journey(self):
        self.idx_ = 0
        self.attempt_idx_ = 0

    @property
    def max_attempts_list(self) -> List[int]:
        if isinstance(self.max_attempts, list):
            return self.max_attempts
        return [self.max_attempts] * self.n_steps()

    @property
    def max_n_steps(self) -> int:
        return sum(self.max_attempts_list)

    def n_steps(self) -> int:
        return len(self.steps.cumulative_steps())

    def n_steps_left(self) -> int:
        return self.n_steps() - self.idx_ - 1

    def is_finished(self) -> bool:
        return self.n_steps_left() < 0

    def register_promotion_next_step(self) -> None:
        self.idx_ += 1

    def register_attempt(self) -> None:
        self.attempt_idx_ += 1

    def reset_attempt(self):
        self.attempt_idx_ = 0

    def _current_state(self) -> SALState:
        return SALState(
            step=self.idx_,
            n_steps_left=self.n_steps_left(),
            attempt=self.attempt_idx_,
            previous_job=Path(self.previous_job_.path) if self.previous_job_ else None,
        )

    def _record_state(self, finished_msg: str = "", finished: bool = False, success: bool = False) -> SALState:
        state = self._current_state()
        state.finished_msg = finished_msg
        state.finished = finished
        state.success = success
        self.history.append(state)
        return state

    def _current_batch_tasks(self) -> List[ConcreteTask]:
        if isinstance(self.task, ASEMolecularDynamics):
            return [
                self.task.model_copy(
                    update={
                        "nsteps": self.steps.cumulative_steps()[self.idx_],
                        "samplingfreq": self.steps.sampling_frequencies()[self.idx_],
                    }
                )
            ]
        settings = self.task.settings
        settings.input.ams.MolecularDynamics.NSteps = self.steps.cumulative_steps()[self.idx_]
        settings.input.ams.MolecularDynamics.Trajectory.SamplingFreq = self.steps.sampling_frequencies()[self.idx_]
        return [
            RestartingAMSTask(
                task_id=self.task.task_id,
                source_settings=settings.as_dict(),
                atomistic_system=self.task.atomistic_system,
                gluer=self.gluer,
                previous_job=self.previous_job_,
            )
        ]

    def _current_batch_check_getters(self) -> List[ConcreteCheckerGetter]:
        return [self.checker_getter]

    def run_current_batch(self, engine: ConcreteEngines) -> CollectionCheckGetResults:
        self.register_attempt()
        task_results = self.task_parallelization.run_tasks(
            task_engine_coll=[(task_i, engine) for task_i in self._current_batch_tasks()]
        )
        coll = []
        for task_result, check_getter in zip(task_results, self._current_batch_check_getters(), strict=True):
            if isinstance(task_result, ASEMolecularDynamicsResult):
                log_level(
                    "step={step} | attempt={attempt}| ASE trajectory path:\n{path}",
                    level="DEBUG",
                    path=task_result.trajectory_path,
                    step=self.idx_,
                    attempt=self.attempt_idx_,
                )
            elif isinstance(task_result, AMSTaskResult):
                log_level(
                    "step={step} | attempt={attempt}| AMSJob path:\n{path}",
                    level="DEBUG",
                    path=task_result.plams_results.rkfpath(),
                    step=self.idx_,
                    attempt=self.attempt_idx_,
                )
                self._current_attempt_job = task_result.executed_job or getattr(task_result.plams_results, "job", None)
            else:
                raise TypeError(f"Unsupported task result type for SAL: {type(task_result)=}")
            coll.append(check_getter.run(result=task_result))
        return CollectionCheckGetResults(validity_get_results=coll)

    def advance(
        self,
        validity_results: CollectionCheckGetResults,
        accuracy_results: List[EngineCheckResult],
    ):
        max_attempt_reached = self.attempt_idx_ >= self.max_attempts_list[self.idx_]

        # seems reasonable to add here the if statement but it is better to try one last time
        # if we passed the checks
        # if max_attempt_reached:
        #     return self.build_results("Max attempts reached.", True)
        def collect_first_failure(vals) -> Optional[EngineCheckResult]:
            if isinstance(vals, list) and len(vals) > 0 and isinstance(vals[0], CheckGetResults):
                for many_r in vals:
                    for r in many_r.validity_results:
                        if not r.is_success():
                            return r
            else:
                for r in vals:
                    if not r.is_success():
                        return r
            return None

        def format_failure(failure: Optional[EngineCheckResult]):
            if failure is None:
                return ""
            val = failure.value
            if isinstance(val, float):
                val = f"{val:0.2f}"
            return f"{failure.property}|{failure.metric}:{val}"

        validity_passed = all(
            r.is_success() for many_r in validity_results.validity_get_results for r in many_r.validity_results
        )
        if not validity_passed:
            infos = format_failure(collect_first_failure(validity_results.validity_get_results))
            if max_attempt_reached:
                return self.build_results(
                    f"Max attempts reached. Validity not passed. {infos}",
                    True,
                    False,
                    finish_loop=self.finish_loop_on_max_attempts,
                )
            return self.build_results(f"Validity not passed. {infos}", False, False)
        accuracy_passed = all(r.is_success() for r in accuracy_results)
        if not accuracy_passed:
            infos = format_failure(collect_first_failure(accuracy_results))
            if max_attempt_reached:
                return self.build_results(
                    f"Max attempts reached. Accuracy not passed. {infos}",
                    True,
                    False,
                    finish_loop=self.finish_loop_on_max_attempts,
                )
            return self.build_results(f"Accuracy not passed. {infos}", False, False)

        ret = self.build_results("Validity AND Accuracy checks passed.", True, True)
        if self.first_step_train and self.idx_ == 1:  # 1 because if we are here the 0 is already become 1
            return ret
        # this skip is specific of SAL
        ret.skip_training_reason = "All the tasks converged"
        return ret

    def finished_reason(self):
        return self.finish_reason

    def build_results(
        self,
        finished_msg: str = "",
        finished: bool = False,
        success: bool = False,
        finish_loop: bool = False,
    ):
        # important step to decide when to transfer job or not
        if finished and self._current_attempt_job is not None:
            self.previous_job_ = self._current_attempt_job
        self._current_attempt_job = None

        self.log_failed(finished_msg, finished, success)
        state = self._record_state(finished_msg, finished, success)
        if finished:
            if finish_loop:
                self.idx_ = self.n_steps()
                self.finish_reason = finished_msg
            else:
                self.register_promotion_next_step()
                self.reset_attempt()
        # TODO: this results in a duplication of information both in self.history
        # and JourneyAdvanceResult.info where should be kept?
        # in SAL is useful know the history and seems the most reasonable
        # but the JourneyAdvanceResult then logs into LoopState,
        # which is also a great place to put this data, giving more context of the ALLoop
        default = JourneyAdvanceResult(info=state.model_dump(mode="json"))
        return default

    def log_failed(self, info: str, finished: bool, success: bool):
        finished_note = "FINISHED" if finished else "FAILED"
        finished_note = "SUCCESS" if success else finished_note
        log_level(
            "SimpleActiveLearningJourney: Step {idx} at attempt {attempt} {finished_note}! Because {info}",
            level="SUCCESS",
            idx=self.idx_,
            attempt=self.attempt_idx_,
            info=info,
            finished_note=finished_note,
            depth=2,
        )

    def journey_table(self) -> str:
        subtitle = (
            f"{self.task.task_id}|{self.task.system_id} -> "
            f"{self.checker_getter.checker_id}|{self.checker_getter.getter_id}"
        )
        table_mds = tabulate(
            {
                "MD NSteps": self.steps.effective_steps(),
                "MD NSteps - Cumulative": self.steps.cumulative_steps(),
                "MD SamplingFreq": self.steps.sampling_frequencies(),
                "MD MaxFrames": [
                    a // b for a, b in zip(self.steps.effective_steps(), self.steps.sampling_frequencies(), strict=True)
                ],
                "max_attempts": self.max_attempts_list,
            },
            headers="keys",
        )
        return self.wrap_table(table_mds, self.start_separator, subtitle=subtitle)

    def history_table(self) -> str:
        table_mds = tabulate(
            [{"Idx": ii, **x.model_dump(exclude={"previous_job"})} for ii, x in enumerate(self.history)],
            headers="keys",
        )
        return self.wrap_table(table_mds, self.history_separator)
