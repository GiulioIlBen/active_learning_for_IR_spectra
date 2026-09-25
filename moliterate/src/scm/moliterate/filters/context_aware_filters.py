from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Iterable, Literal, Optional, Union

import numpy as np

from scm.moliterate.core import BaseChemDataSet, BaseFilter, ConcatChemDataSet
from scm.moliterate.interfaces import UnionInterfaces

if TYPE_CHECKING:
    from scm.moliterate.filters import UnionFilters

# TODO this SimpleContextFilter can made smarter: the problem is that we could filter out also context_dataset
# which we shouldn't, therefore somehow we should say to first look at the context_dataset and then filter only
# data from db. Some Filters backends can do this: ApricotFilter,
# with FacilityLocationSelection and FeatureBasedSelection
# has an entry: initial_subset=[1, 5, 6, 8, 10] which it might be good for us. Also AMSConformersFilter
# the backend can be used for that but since support groupby option makes less accessible


class ContextFilter(BaseFilter):
    """
    ContextFilter are filters that has as settings another dataset that we need to keep in consideration
    """

    context_dataset: Union[UnionInterfaces, ConcatChemDataSet]
    selection_filter: "UnionFilters"

    @abstractmethod
    def apply_filter(self, db: BaseChemDataSet) -> Optional[Iterable[int]]: ...

    def __hash__(self) -> int:  # Needed only for type hint
        return super().__hash__()


class SimpleContextFilter(ContextFilter):
    """
    Apply a filter on ``context_dataset + db`` and keep only selected indices belonging to ``db``.
    """

    type: Literal["SimpleContextFilter"] = "SimpleContextFilter"  # Field discriminator!
    context_dataset: Union[UnionInterfaces, ConcatChemDataSet]
    selection_filter: "UnionFilters"

    def apply_filter(self, db: BaseChemDataSet) -> Optional[Iterable[int]]:
        target_len = len(db)
        if target_len == 0:
            return None

        context_len = len(self.context_dataset)
        if context_len == 0:
            return self.selection_filter.apply_filter(db)

        concat_ds = ConcatChemDataSet.model_construct(data_source=[self.context_dataset, db])
        selected = self.selection_filter.apply_filter(concat_ds)
        if selected is None:
            return None
        selected_np = concat_ds.subset(selected).flatten_idxs_to_each_iter()
        ret = selected_np[1]
        if ret.size == 0:
            return ret
        return ret
