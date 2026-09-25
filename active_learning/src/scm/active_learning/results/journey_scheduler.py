from __future__ import annotations

from typing import Dict, Literal, Optional, Union

from pydantic import BaseModel, JsonValue

from scm.active_learning.results.collection_check_get import (
    CollectionCheckGetResults,
)
from scm.active_learning.results.engine_check import (
    EngineCheckResult,
)


class JourneyAdvanceResult(BaseModel):
    info: Dict[str, JsonValue] = {}
    skip_training_reason: Union[str, Literal["All the tasks converged"]] = ""

    @property
    def skip_training(self):
        return self.skip_training_reason != ""

    @property
    def all_tasks_converged(self):
        return self.skip_training_reason == "All the tasks converged"

    @classmethod
    def get_passed_tasks(
        cls,
        task_results: Optional[CollectionCheckGetResults],
        accuracy_checks_results: Optional[list[EngineCheckResult]],
    ):
        task_ids = {}
        if task_results is not None:
            for x in task_results.validity_get_results:
                for xx in x.validity_results:
                    task_ids[xx.task_id] = task_ids.get(xx.task_id, True) and xx.is_success()
        if accuracy_checks_results is not None:
            for x in accuracy_checks_results:
                task_ids[x.task_id] = task_ids.get(x.task_id, True) and x.is_success()
        return task_ids
