from typing import List

from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.core.properties_info import PropertyInfo
from scm.moliterate.transforms import UnionRowTransform
from scm.moliterate.transforms.core import BaseEntryTransform


class ComposedTransform(BaseEntryTransform):
    transforms: List[UnionRowTransform]

    def properties_in(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        # disclaimer: chatGPT generated, not really looked into it, looks fine
        # current = list(properties_info or [])
        # required: List[PropertyInfo] = []
        # required_names = set()
        # for trn in self.transforms:
        #     needed = trn.properties_in(current)
        #     for prop in needed:
        #         if prop.name not in required_names:
        #             required.append(prop)
        #             required_names.add(prop.name)
        #     current = trn.properties_out(current)
        return properties_info

    def properties_out(self, properties_info: List[PropertyInfo]) -> List[PropertyInfo]:
        current = list(properties_info or [])
        for trn in self.transforms:
            current = trn.properties_out(current)
        return current

    def __call__(self, row: ChemDataEntry) -> ChemDataEntry:
        for trn in self.transforms:
            row = trn(row)
        return row
