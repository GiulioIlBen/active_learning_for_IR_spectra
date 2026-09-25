from __future__ import annotations

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from scm.active_learning.mlip.core import MLIPTrainer
from scm.active_learning.mlip.mace_trainer import MACETrainer
from scm.active_learning.mlip.params_trainer import ParAMSTrainer
from scm.active_learning.mlip.schnetpack_trainer import SchNetPackTrainer
from scm.active_learning.mlip.splitting_strategy import DataAndSplittingStrategy

ConcreteMLIPTrainer = Annotated[
    Union[
        ParAMSTrainer,
        SchNetPackTrainer,
        MACETrainer,
    ],
    Field(discriminator="type"),
]

__all__ = [
    "DataAndSplittingStrategy",
    "ConcreteMLIPTrainer",
    "MLIPTrainer",
    "ParAMSTrainer",
    "SchNetPackTrainer",
    "MACETrainer",
]
