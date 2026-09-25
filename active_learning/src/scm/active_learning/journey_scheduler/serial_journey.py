from typing import List, Literal, Tuple

from tabulate import tabulate

from scm.active_learning.checker_getters import ConcreteCheckerGetter
from scm.active_learning.results import (
    CollectionCheckGetResults,
    EngineCheckResult,
    JourneyAdvanceResult,
)
from scm.active_learning.tasks import ConcreteTask

from .core import JourneyScheduler


class SequentialStepsJourney(JourneyScheduler):
    type: Literal["SequentialStepsJourney"] = "SequentialStepsJourney"
    couples: List[Tuple[ConcreteTask, ConcreteCheckerGetter]]
    max_attempts: int
    idx_: int = 0
    attempt_idx_: int = 0

    def restart_journey(self):
        self.idx_ = 0
        self.attempt_idx_ = 0

    @property
    def max_n_steps(self):
        return self.n_steps() * self.max_attempts

    def n_steps(self) -> int:
        return len(self.couples)

    def n_steps_left(self) -> int:
        return self.n_steps() - self.idx_

    def is_finished(self):
        return self.idx_ >= len(self.couples)

    def register_attempt(self) -> None:
        self.attempt_idx_ += 1

    def _current_batch_tasks(self) -> List[ConcreteTask]:
        return [self.couples[self.idx_][0]]

    def _current_batch_check_getters(self) -> List[ConcreteCheckerGetter]:
        return [self.couples[self.idx_][1]]

    def advance(
        self,
        validity_results: CollectionCheckGetResults,
        accuracy_results: List[EngineCheckResult],
    ):
        default = JourneyAdvanceResult()
        reasonable_passed = all(
            r.is_success() for many_r in validity_results.validity_get_results for r in many_r.validity_results
        )
        if not reasonable_passed:
            return default
        accuracy_passed = all(r.is_success() for r in accuracy_results)
        if not accuracy_passed:
            return default
        if self.idx_ < self.max_attempts:
            return default
        self.idx_ += 1
        self.attempt_idx_ = 0
        return JourneyAdvanceResult(skip_training_reason="All the tasks converged")

    def journey_table(self) -> str:
        table = tabulate(
            [
                {
                    "task": x[0].task_id,
                    "system": x[0].system_id,
                    "checker": x[1].checker_id,
                    "max_attempts": self.max_attempts,
                }
                for x in self.couples
            ],
            headers="keys",
        )
        return "\n".join([self.start_separator, table, self.end_separator])
