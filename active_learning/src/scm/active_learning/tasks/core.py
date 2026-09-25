from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from pydantic import BaseModel

if TYPE_CHECKING:
    from scm.active_learning.results.task import ConcreteTaskResult

from ..engines.core import Engine

T = TypeVar("T", bound=Engine)


class Task(BaseModel, Generic[T]):
    # type: Literal[""] = "" # discriminator
    task_id: str
    # source_settings: Dict[str, JsonValue] = {}

    @property
    @abstractmethod
    def system_id(self) -> str: ...

    # @abstractmethod
    # def validate_properties(self, requested_properties: List[PropertyInfo]): ...

    @abstractmethod
    def run(self, engine: T, *args, **kwargs) -> "ConcreteTaskResult": ...
