from __future__ import annotations

from typing import List

from pydantic import BaseModel
from tabulate import tabulate

from scm.active_learning.results.engine_check import EngineCheckResult
from scm.active_learning.results.frame_getter import FrameGetterResult


class CheckGetResults(BaseModel):
    validity_results: List[EngineCheckResult]
    getter_results: FrameGetterResult

    def validity_results_table(self):
        return tabulate([x.model_dump() for x in self.validity_results], headers="keys")
