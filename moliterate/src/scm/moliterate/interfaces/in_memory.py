"""In-memory implementation of the molecule database interface."""

import copy
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional

from pydantic import PrivateAttr

from scm.moliterate.core.base_chem_dataset_writer import BaseChemDataSetWriter
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.transforms import UnionRowTransform


class InMemoryAtomsError(Exception):
    pass


class InMemoryAtomsWarning(Warning):
    pass


class InMemoryMolData(BaseChemDataSetWriter):
    type: Literal["InMemoryMolData"] = "InMemoryMolData"
    data_source: str = ""
    _data: List[ChemDataEntry] = PrivateAttr(default=[])
    _metadata: Dict[str, Any] = PrivateAttr(default={})

    @classmethod
    def create(
        cls,
        data_source: str = "",
        available_properties: Optional[List[PropertyInfo]] = None,
        distance_unit: str = "Ang",
        transforms: Optional[List[UnionRowTransform]] = None,
        **kwargs,
    ) -> "InMemoryMolData":
        ret = cls(data_source=data_source, transforms=transforms or [])
        ret.available_properties = available_properties or []
        ret.distance_unit = distance_unit
        return ret

    def add_system(self, mol_data_row: ChemDataEntry):
        mol_data_row.properties = {k.name: mol_data_row.properties[k.name] for k in self.available_properties}
        self._data.append(mol_data_row)

    def update_metadata(self, **kwargs):
        """update the metadata of the dataset"""
        self._metadata.update(kwargs)

    def update_row_metadata(self, idx: int, absolute: bool = False, **update):
        if absolute:
            self._data[idx].metadata.update(update)
        else:
            self.get_row(idx).metadata.update(update)

    def total_len(self) -> int:
        """method to get the total len of the data"""
        return len(self._data)

    @property
    def available_properties(self) -> List[PropertyInfo]:
        """Available properties in the dataset"""
        return self.metadata.get("__properties__", [])

    @available_properties.setter
    def available_properties(self, val: Optional[List[PropertyInfo]]):
        """Available properties in the dataset"""
        __properties__ = val or []
        self.update_metadata(__properties__=[PropertyInfo(**p) if isinstance(p, dict) else p for p in __properties__])

    @property
    def distance_unit(self) -> str:
        return self.metadata["__distance_unit__"]

    @distance_unit.setter
    def distance_unit(self, val: str):
        self.update_metadata(__distance_unit__=val)

    @property
    def metadata(self) -> Dict[str, Any]:
        """Global metadata"""
        return self._metadata.copy()

    def __iter__(self) -> Iterator[ChemDataEntry]:
        # TBN: I decided to copy the data from the list to avoid that composed_transform changes the
        # original data. I think is a better pattern
        trn = self.composed_transform
        for idx_absolute in map(int, self.select_indices()):
            row = copy.deepcopy(self._data[idx_absolute])
            row.idx_absolute = idx_absolute
            row.idx_origin = idx_absolute
            yield trn(row)

    def get_row(self, idx: int) -> ChemDataEntry:
        idxs = [x for x in self.select_indices(indices=idx)]
        idx_origin = idxs[0]
        row = self._data[idx_origin]
        row.idx_absolute = idx_origin
        row.idx_origin = idx_origin
        return self.composed_transform(row)
