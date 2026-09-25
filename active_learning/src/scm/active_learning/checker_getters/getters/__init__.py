from __future__ import annotations

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from .ams_traj_getter import AMSTrajGetter
from .ase_traj_getter import ASETrajGetter
from .conf_traj_getter import ConfTrajGetter
from .conformers_getter import ConformersGetter
from .core import Getter
from .nms_getter import NMSGetters
from .sp_getter import SPGetter

ConcreteGetter = Annotated[
    Union[SPGetter, AMSTrajGetter, ASETrajGetter, ConfTrajGetter, ConformersGetter, NMSGetters],
    Field(discriminator="type"),
]


__all__ = [
    "ConcreteGetter",
    "Getter",
    "AMSTrajGetter",
    "ASETrajGetter",
    "ConfTrajGetter",
    "SPGetter",
    "ConformersGetter",
    "NMSGetters",
]
