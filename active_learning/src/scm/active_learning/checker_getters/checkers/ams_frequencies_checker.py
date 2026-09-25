from __future__ import annotations

from typing import ClassVar, List, Literal, Optional, Tuple, Type

import numpy as np

from scm.active_learning.checker_getters.checkers.core import Checker
from scm.active_learning.results import EngineCheckResult
from scm.active_learning.results.task import AMSTaskResult
from scm.active_learning.tasks import AMSGOIRTask, AMSTask, Task


class AMSFrequenciesChecker(Checker[AMSTaskResult]):
    """Check that an AMS result contains at least one usable normal mode."""

    type: Literal["AMSFrequenciesChecker"] = "AMSFrequenciesChecker"
    supported_tasks: ClassVar[Tuple[Type[Task], ...]] = (AMSTask, AMSGOIRTask)
    checker_id: str = "AMSFrequenciesChecker"
    min_wavenumber: float = 500.0
    engine_file: Optional[str] = None

    def _frequencies(self, result: AMSTaskResult) -> np.ndarray:
        jobresults = result.plams_results
        engine_file = self.engine_file
        if engine_file == "auto":
            engine_file = jobresults.rkfpath(file="engine")
            if engine_file is not None:
                from scm.plams import kftools

                engine_file = kftools.KFFile(engine_file).reader.main_section
        return np.asarray(jobresults.get_frequencies(engine=engine_file), dtype=float)

    def run(self, result: AMSTaskResult, **kwargs) -> List[EngineCheckResult]:
        try:
            frequencies = self._frequencies(result)
        except Exception:
            frequencies = np.asarray([], dtype=float)

        finite_frequencies = frequencies[np.isfinite(frequencies)]
        max_frequency = float(finite_frequencies.max()) if finite_frequencies.size else -1.0
        success = "OK" if np.any(finite_frequencies > self.min_wavenumber) else "NO_USABLE_FREQUENCIES"
        return [
            EngineCheckResult(
                **self.collect_ids(result=result),
                property="frequencies",
                units="cm^-1",
                metric="max_wavenumber",
                value=max_frequency,
                target=self.min_wavenumber,
                lower_is_better=False,
                n_entries=int(finite_frequencies.size),
                success=success,
            )
        ]
