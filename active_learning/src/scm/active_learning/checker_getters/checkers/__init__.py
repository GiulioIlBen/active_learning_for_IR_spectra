from __future__ import annotations

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from .ams_frequencies_checker import AMSFrequenciesChecker
from .ams_ir_committee_checker import AMSIRCommitteeAgreementChecker
from .ams_traj_checkers import AMSTrajChecker
from .ase_traj_checker import ASETrajChecker
from .conformers_energy_checker import (
    ConfJobFailChecker,
    ConformersChecker,
    ConformersEnergyChecker,
)
from .core import Checker
from .none_checker import NoneChecker
from .sp_checkers import SPChecker

ConcreteChecker = Annotated[
    Union[
        NoneChecker,
        SPChecker,
        AMSFrequenciesChecker,
        AMSTrajChecker,
        ASETrajChecker,
        AMSIRCommitteeAgreementChecker,
        ConformersEnergyChecker,
        ConfJobFailChecker,
        ConformersChecker,
    ],
    Field(discriminator="type"),
]


__all__ = [
    "ConcreteChecker",
    "Checker",
    "NoneChecker",
    "SPChecker",
    "AMSFrequenciesChecker",
    "AMSTrajChecker",
    "ASETrajChecker",
    "AMSIRCommitteeAgreementChecker",
    "ConformersEnergyChecker",
    "ConfJobFailChecker",
    "ConformersChecker",
]
