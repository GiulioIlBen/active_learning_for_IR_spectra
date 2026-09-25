from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Literal

import numpy as np

from scm.moliterate.core.base_chem_dataset import BaseChemDataSet
from scm.moliterate.core.chem_data_entry import ChemDataEntry
from scm.moliterate.core.properties_info import PropertyInfo

if TYPE_CHECKING:
    from scm.moliterate import ConcreteInterfaces


class ConcatChemDataSet(BaseChemDataSet):
    type: Literal["ConcatChemDataSet"] = "ConcatChemDataSet"
    data_source: List["ConcreteInterfaces"]

    @property
    def len_collection(self):
        return [len(x) for x in self.data_source]

    def total_len(self) -> int:
        """method to get the total len of the data"""
        return sum(self.len_collection)

    @property
    def available_properties(self) -> List[PropertyInfo]:
        """Available uniform properties in the datasets taking into account possible name transforms"""
        all_properties = [y for x in self.data_source for y in x.out_properties]
        return list(set(all_properties))

    @property
    def metadata(self) -> Dict[str, Any]:
        """Global metadata"""
        ret = {}
        for x in self.data_source:
            ret.update(x.metadata)
        return ret

    def __iter__(self) -> Iterator[ChemDataEntry]:
        trn = self.composed_transform
        indices_list = self.flatten_idxs_to_each_iter()
        for iter_i, indices_i in zip(self.data_source, indices_list):
            if indices_i is not None and len(indices_i) == 0:
                continue
            iter_sub = iter_i.subset(indices_i)
            for row, ii in zip(iter_sub, indices_i):
                row.idx_absolute = int(ii)
                yield trn(row)

    def get_row(self, idx: int) -> ChemDataEntry:
        iter_sub = self.subset(idx)
        row = list(iter_sub)[0]
        row.idx_absolute = int(iter_sub.absolute_idxs[0])
        return self.composed_transform(row)

    def flatten_idxs_to_each_iter(self):
        # split the indexes for each iterator
        # for example: we have 2 iterators of self.len_coll = [5,3] so csum = [5,8]
        # the selected indexes are [1,6,7] therefore we want to split in
        # [[1], [6,7]] and bring back the second iterator to its indexes values
        # [[1], [1,2]] this should be the indices_list values
        if self.absolute_idxs is None:
            ret = np.arange(self.total_len(), dtype=np.int64)
        else:
            ret = np.asarray(self.absolute_idxs, dtype=np.int64)
        csum = np.cumsum(self.len_collection)
        indices_list = [
            ret[np.logical_and(m_in <= ret, ret < m_ax)] - m_in for m_ax, m_in in zip(csum, [0, *csum[:-1]])
        ]
        return indices_list

    # def __str__(self) -> str:
    #     def indent(text: str, prefix: str = "\t") -> str:
    #         return "\n".join(prefix + line for line in text.splitlines())

    #     ret = super().__str__()
    #     formatted_items = [indent(str(x), "\t") for x in self.data_source]
    #     ret += "\n\tColl:\n" + "\n".join(formatted_items)
    #     return ret

    # def __repr__(self) -> str:
    #     return super().__str__()
