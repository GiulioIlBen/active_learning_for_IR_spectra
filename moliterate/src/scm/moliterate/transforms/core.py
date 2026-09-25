from abc import ABC, abstractmethod
from typing import List

from pydantic import BaseModel

from ..core.chem_data_entry import ChemDataEntry
from ..core.properties_info import PropertyInfo


class BaseEntryTransform(ABC, BaseModel):
    # type: Literal[""] = ""  # Field discriminator
    @abstractmethod
    def properties_in(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        """tells the user what are the properties that are used/needed as input for this transform

        :param properties_info: _description_
        :type properties_info: List[PropertyInfo]
        :return: _description_
        :rtype: List[PropertyInfo]
        """
        ...

    @abstractmethod
    def properties_out(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]: ...

    @abstractmethod
    def __call__(self, row: ChemDataEntry) -> ChemDataEntry: ...
