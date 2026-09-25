from __future__ import annotations

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from scm.active_learning.journey_scheduler.core import JourneyScheduler
from scm.active_learning.journey_scheduler.molecule_journey import MoleculesJourney
from scm.active_learning.journey_scheduler.serial_journey import SequentialStepsJourney
from scm.active_learning.journey_scheduler.simple_active_learning import (
    SimpleActiveLearningJourney,
)

ConcreteJourneyScheduler = Annotated[
    Union[SequentialStepsJourney, SimpleActiveLearningJourney, MoleculesJourney],
    Field(discriminator="type"),
]

__all__ = [
    "ConcreteJourneyScheduler",
    "JourneyScheduler",
    "SequentialStepsJourney",
    "SimpleActiveLearningJourney",
    "MoleculesJourney",
]
