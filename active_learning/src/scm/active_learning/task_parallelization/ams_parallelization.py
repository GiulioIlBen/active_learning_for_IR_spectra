from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

import scm.plams as plams
from pydantic import PositiveInt

from scm.active_learning.logging import log_level

from .core import ParallelStrategy

if TYPE_CHECKING:
    from scm.active_learning.engines import ConcreteEngines
    from scm.active_learning.results.task import ConcreteTaskResult
    from scm.active_learning.tasks import AMSTask, ConcreteTask


# List[Tuple[AMSTask, Union[AMSEngine, ParAMSEngine]]]
# [Tuple[AMSTask, Union[AMSEngine, ParAMSEngine]], AMSTaskResult]
class AMSParallelStrategy(ParallelStrategy):
    # type: Literal["serial"] = "serial"  # It must have a type, and it is used as discriminator
    watch_ams_log_stdout: bool = False

    @abstractmethod
    def switch_parallel_plams(self, njobs: Optional[int] = -1) -> Tuple[Optional[plams.JobRunner], plams.Settings]: ...

    def prepare_job(self, task: "ConcreteTask", job: plams.AMSJob) -> None:
        """Apply optional task-specific changes to a built AMS job before execution.

        `run_tasks()` calls this hook after `task.build_job(...)` and before `job.run(...)`.
        The default implementation looks for a `prepare_job(job)` method on the task and
        invokes it when present. This lets journeys influence AMS execution through the
        task-parallelization path instead of bypassing it with custom run logic.
        """
        prepare_job = getattr(task, "prepare_job", None)
        if callable(prepare_job):
            prepare_job(job)

    def run_tasks(self, task_engine_coll: List[Tuple["ConcreteTask", "ConcreteEngines"]]):
        # avoid circular import
        from scm.active_learning.engines import (
            AMSEngine,
            ASEEngine,
            ParAMSEngine,
            to_ams_engine,
        )
        from scm.active_learning.results.task import AMSConformersResults, AMSTaskResult
        from scm.active_learning.tasks import AMSConformersTask, AMSGOIRTask, AMSMDTask, AMSTask

        AMSTasksCompatible = (AMSTask, AMSMDTask, AMSGOIRTask, AMSConformersTask)

        jr, s_nproc = self.switch_parallel_plams(njobs=len(task_engine_coll))

        def to_ams_task(tsk: ConcreteTask) -> AMSTask:
            if isinstance(tsk, AMSTasksCompatible):
                return tsk
            raise TypeError(f"Invalid task type: {type(tsk)=}")

        def is_parallelizable(tsk: ConcreteTask, eng: ConcreteEngines) -> bool:
            return isinstance(tsk, AMSTasksCompatible) and isinstance(eng, (AMSEngine, ParAMSEngine, ASEEngine))

        results: List[ConcreteTaskResult] = [None] * len(task_engine_coll)  # pyright: ignore[reportAssignmentType]
        parallel_items: List[Tuple[int, AMSTask, ConcreteEngines]] = []
        serial_items: List[Tuple[int, ConcreteTask, ConcreteEngines]] = []

        for i, (task_i, engine_i) in enumerate(task_engine_coll):
            if is_parallelizable(task_i, engine_i):
                parallel_items.append((i, to_ams_task(task_i), engine_i))
            else:
                serial_items.append((i, task_i, engine_i))

        for i, task_i, engine_i in parallel_items:
            job = task_i.build_job(engine=to_ams_engine(engine_i), extra=s_nproc)
            self.prepare_job(task_i, job)
            log_level(
                "AMS parallel job prepared: name={name} | runscript_settings={runscript_settings}\n{runscript}",
                level="DEBUG",
                name=job.name,
                runscript_settings=job.settings.as_dict().get("runscript"),
                runscript=job.get_runscript().rstrip(),
            )
            res = job.run(jobrunner=jr, watch=self.watch_ams_log_stdout)
            result_cls = AMSConformersResults if isinstance(task_i, AMSConformersTask) else AMSTaskResult
            results[i] = result_cls(task=task_i, engine=engine_i, plams_results=res)

        for i, task_i, engine_i in serial_items:
            results[i] = task_i.run(engine=engine_i)

        return results


class AMSSerialStrategy(AMSParallelStrategy):
    type: Literal["AMSSerialStrategy"] = "AMSSerialStrategy"
    # These are expert settings!
    nproc: Optional[int] = None
    OMP_NUM_THREADS: Optional[int] = None

    def switch_parallel_plams(self, njobs: Optional[int] = -1):
        del njobs
        s_job = plams.Settings()
        if self.nproc is not None:
            s_job.runscript.nproc = self.nproc
        if self.OMP_NUM_THREADS is not None:
            s_job.runscript.preamble_lines = [f"export OMP_NUM_THREADS={self.OMP_NUM_THREADS}"]
        return None, s_job


class AMSParallelCPU(AMSParallelStrategy):
    type: Literal["AMSParallelCPU"] = "AMSParallelCPU"  # It must have a type, and it is used as discriminator
    maxjobs: Optional[PositiveInt] = None
    nproc: Optional[PositiveInt] = None

    def switch_parallel_plams(self, njobs: Optional[int] = -1):
        jr = None
        if self.maxjobs:
            jr = plams.JobRunner(parallel=True, maxjobs=self.maxjobs)

        s_nproc = plams.Settings()
        if self.nproc is not None:
            s_nproc.runscript.nproc = self.nproc
        return jr, s_nproc


class AMSParallelNProcCPU(AMSParallelStrategy):
    """
    CPU base parallelization:
    I have n CPUs and m Jobs:
    - if n > m: run m Jobs sequentially each one parallelized on n CPUs
    - if n < m: run n Jobs in parallel each one running on 1 CPU
    """

    type: Literal["AMSParallelNProcCPU"] = "AMSParallelNProcCPU"  # It must have a type, and it is used as discriminator
    njobs: Optional[int] = None
    nproc: Optional[int] = None
    auto_maxjobs: bool = True

    def switch_parallel_plams(self, njobs: Optional[int] = -1):
        nj = njobs or self.njobs or -1
        jr = None
        nproc_ret = self.nproc
        if self.auto_maxjobs and self.nproc is not None and nj > self.nproc:
            jr = plams.JobRunner(parallel=True, maxjobs=self.nproc)
            nproc_ret = 1
            plams.log(f"Switch parallel made: maxjobs={self.nproc}|{nproc_ret=}", level=1)
        else:
            plams.log(f"NO switch parallel: maxjobs={1}|{nproc_ret=}", level=1)

        s_nproc = plams.Settings()
        if nproc_ret is not None:
            s_nproc.runscript.nproc = nproc_ret
        return jr, s_nproc


class AMSParallelMaxJobsGPU(AMSParallelStrategy):
    """
    GPU base parallelization:
    I have n MiB and m Jobs:
    - each Job consumes x MiB (1200): run n//x Jobs in parallel
    + recommended to have nproc=1
    + recommended to set export OMP_NUM_THREADS=1
    """

    type: Literal["AMSParallelMaxJobsGPU"] = (
        "AMSParallelMaxJobsGPU"  # It must have a type, and it is used as discriminator
    )
    maxjobs: Optional[int] = None
    auto_maxjobs: bool = False
    memory_avail: Optional[int] = None
    memory_each_job: Optional[int] = None
    # These are expert settings!
    nproc: Optional[int] = 1
    OMP_NUM_THREADS: Optional[int] = 1

    def switch_parallel_plams(self, njobs: Optional[int] = -1):
        jr = None
        if self.maxjobs is not None:
            jr = plams.JobRunner(parallel=True, maxjobs=self.maxjobs)
        elif self.auto_maxjobs and self.memory_avail is not None and self.memory_each_job is not None:
            jr = plams.JobRunner(parallel=True, maxjobs=self.memory_avail // self.memory_each_job)
            plams.log(f"Switch parallel made: {self.memory_avail=}|{jr.maxjobs=}", level=1)
        else:
            plams.log(f"NO switch parallel: {self.memory_avail=}|jr.maxjobs={None}", level=1)

        s_job = plams.Settings()
        if self.nproc is not None:
            s_job.runscript.nproc = self.nproc
        if self.OMP_NUM_THREADS is not None:
            s_job.runscript.preamble_lines = [f"export OMP_NUM_THREADS={self.OMP_NUM_THREADS}"]
        return jr, s_job
