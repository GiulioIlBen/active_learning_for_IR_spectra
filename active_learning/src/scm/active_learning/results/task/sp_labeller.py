from __future__ import annotations

from typing import Dict, Literal

from scm.moliterate import (
    ConcreteInterfaces,
)

from scm.active_learning.engines.core import Engine
from scm.active_learning.tasks import SPLabellerData

from .core import TaskResult


class SPLabellingResults(TaskResult):
    type: Literal["SPLabellingResults"] = "SPLabellingResults"
    task: SPLabellerData
    engine: Engine
    dataset: ConcreteInterfaces
    failed_simulations: Dict[int, str] = {}

    @property
    def from_task(self):
        return self.task

    @property
    def from_engine(self):
        return self.engine
