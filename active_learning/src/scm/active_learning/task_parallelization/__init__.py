from __future__ import annotations

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from scm.active_learning.task_parallelization.ams_parallelization import (
    AMSParallelCPU,
    AMSParallelMaxJobsGPU,
    AMSParallelNProcCPU,
    AMSSerialStrategy,
)
from scm.active_learning.task_parallelization.core import ParallelStrategy
from scm.active_learning.task_parallelization.serial_parallelization import (
    SerialStrategy,
)

ConcreteAMSParallelStrategies = Annotated[
    Union[AMSSerialStrategy, AMSParallelCPU, AMSParallelNProcCPU, AMSParallelMaxJobsGPU],
    Field(discriminator="type"),
]
ConcreteParallelizationStrategy = Annotated[
    Union[SerialStrategy, AMSSerialStrategy, AMSParallelCPU, AMSParallelNProcCPU, AMSParallelMaxJobsGPU],
    Field(discriminator="type"),
]

__all__ = [
    "AMSSerialStrategy",
    "SerialStrategy",
    "ConcreteParallelizationStrategy",
    "ParallelStrategy",
    "ConcreteAMSParallelStrategies",
]
