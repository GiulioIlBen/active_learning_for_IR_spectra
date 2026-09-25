from __future__ import annotations

from typing import Any, Dict, Generic, Iterable, List, Literal, Optional, TypeVar

from pydantic import BaseModel, field_validator
from scm.moliterate import ConcreteInterfaces, PropertyInfo
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.plams import Settings

from scm.active_learning.task_parallelization import ParallelStrategy

T = TypeVar("T", bound=ParallelStrategy)


class Engine(BaseModel, Generic[T]):
    # type: Literal[""] = ""
    engine_id: str

    @field_validator("engine_id")
    @classmethod
    def _validate_engine_id(cls, value: str) -> str:
        if "-" in value:
            raise ValueError("engine_id must not contain '-'")
        return value

    def _validate_properties(self, requested_properties: List[PropertyInfo]) -> List[str]: ...

    def is_finetunable(self) -> bool:
        return False

    @property
    def committee_members(self) -> List["Engine"]:
        return []

    def run_single_point(
        self,
        properties: List[PropertyInfo],
        dataset: ConcreteInterfaces,
        parallel_settings: ParallelStrategy,
        **kwargs,
    ) -> Iterable[Dict[str | Literal["FAILURE"], Any | str]]:
        """Very important for the implementation to be units consistent!"""
        ...

    @property
    def training_dataset(self) -> Optional[Iterable[ChemDataEntry]]:
        return None

    @property
    def validation_dataset(self) -> Optional[Iterable[ChemDataEntry]]:
        return None

    @property
    def settings(self):
        return Settings(self.model_dump(mode="json"))

    # TODO implement these settings
    # result.get_torch_model() # Interface to torchsim
    # result.get_calculator() # Interface to ASE
    # result.get_production_settings(committee=True) # Interface to AMS
