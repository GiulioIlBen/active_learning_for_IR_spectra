from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import ClassVar, List

from pydantic import BaseModel

from scm.active_learning.checker_getters import ConcreteCheckerGetter
from scm.active_learning.engines import ConcreteEngines
from scm.active_learning.results import (
    CollectionCheckGetResults,
    EngineCheckResult,
    JourneyAdvanceResult,
)
from scm.active_learning.task_parallelization import (
    ConcreteParallelizationStrategy,
    SerialStrategy,
)
from scm.active_learning.tasks import ConcreteTask


class JourneyScheduler(BaseModel):
    # type: Literal
    task_parallelization: ConcreteParallelizationStrategy = SerialStrategy()
    start_separator: ClassVar[str] = " JourneyScheduler "
    history_separator: ClassVar[str] = " JourneyHistory "
    end_separator: ClassVar[str] = "="
    min_banner_width: ClassVar[int] = 26
    max_banner_width: ClassVar[int] = 120

    @abstractmethod
    def restart_journey(self) -> None:
        """
        Sometimes you need to do an initial round that you do not want to be counted,
        this method is useful to make restart, but keep the history!
        """
        ...

    @property
    @abstractmethod
    def max_n_steps(self) -> int: ...
    @abstractmethod
    def n_steps(self) -> int: ...
    @abstractmethod
    def n_steps_left(self) -> int:
        """if it is -1 means goes infinite"""
        ...

    def finished_reason(self) -> str:
        return "CONVERGED"

    @abstractmethod
    def is_finished(self) -> bool: ...
    @abstractmethod
    def register_attempt(self) -> None: ...
    @abstractmethod
    def _current_batch_tasks(self) -> List[ConcreteTask]: ...
    @abstractmethod
    def _current_batch_check_getters(self) -> List[ConcreteCheckerGetter]: ...
    @abstractmethod
    def advance(
        self,
        validity_results: CollectionCheckGetResults,
        accuracy_results: List[EngineCheckResult],
    ) -> JourneyAdvanceResult:
        """_summary_

        :param validity_results: _description_
        :type validity_results: CollectionCheckGetResults
        :param accuracy_results: _description_
        :type accuracy_results: List[EngineCheckResult]
        :return: return True if all the current tasks are successful. Therefore you can skip retraining the MLIP
        :rtype: bool
        """
        ...

    def run_current_batch(self, engine: ConcreteEngines) -> CollectionCheckGetResults:
        self.register_attempt()
        task_results = self.task_parallelization.run_tasks(
            task_engine_coll=[(task_i, engine) for task_i in self._current_batch_tasks()]
        )
        coll = []
        for task_res, check_getter in zip(task_results, self._current_batch_check_getters(), strict=True):
            coll.append(check_getter.run(result=task_res))
        return CollectionCheckGetResults(validity_get_results=coll)

    def journey_table(self) -> str:
        return self.wrap_table("", self.start_separator)

    def history_table(self) -> str:
        return self.wrap_table("", self.history_separator)

    def prepare_for_run_folder(self, run_dir: Path) -> None:
        return None

    def wrap_table(self, table: str, title: str, subtitle: str | None = None):
        banner_lines = table.split("\n") + [title] + ([subtitle] if subtitle is not None else [])
        banner_width = self._banner_width(banner_lines, title)
        side_width = max(1, (banner_width - len(title)) // 2)
        sep = self.end_separator * side_width
        top = sep + title + sep
        ret = [top]
        if subtitle is not None:
            ret.append(subtitle)
        ret.append(table)
        ret.append(self.end_separator * max(len(top), self.min_banner_width))
        return "\n".join(ret)

    def _banner_width(self, lines: List[str], title: str) -> int:
        max_len = max(map(len, lines), default=0)
        min_width = max(len(title) + 2, self.min_banner_width)
        return max(min(max_len, self.max_banner_width), min_width)
