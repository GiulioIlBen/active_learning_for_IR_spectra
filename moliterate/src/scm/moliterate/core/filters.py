from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable, Optional, Union

from scm.moliterate.core.base_chem_dataset import BaseChemDataSet, TBaseChemDataSet
from scm.moliterate.utils.hashable_model import FrozenHashableModel

if TYPE_CHECKING:
    from scm.moliterate.filters import UnionFilters
    from scm.moliterate.filters.bitwise_filters import AndFilter, NotFilter, OrFilter, PipeFilter


class BaseFilterError(ValueError):
    pass


class BaseFilterWarning(Warning):
    pass


class BaseFilter(ABC, FrozenHashableModel):
    # type: Literal[""] = ""  # Field discriminator!

    def __call__(self, db: TBaseChemDataSet) -> TBaseChemDataSet:
        return db.subset(indices=self.apply_filter(db))

    @abstractmethod
    def apply_filter(self, db: BaseChemDataSet) -> Optional[Iterable[int]]:
        """
        returns RELATIVE idx position to the BaseIterAtomsData provided.
        BUT if you can't return relative there is a _parse_relative_idxs but not preferred

        :param db: _description_
        :type db: BaseIterAtomsData
        :return: it returns RELATIVE idx position to the BaseIterAtomsData provided! they are 0-based index
        :rtype: Iterable[int]
        """
        pass

    def __iter__(self, db: BaseChemDataSet):  # pyright: ignore[reportIncompatibleMethodOverride]
        """Overright this implementation for custom made filter iterators (like for sql)"""
        for row in self(db):
            yield row

    @staticmethod
    def _parse_subset_n_data(subset_n_data: Union[float, int], total: int) -> int:
        if subset_n_data >= 1:
            return int(subset_n_data)
        return max(int(total * float(subset_n_data)), 1)

    def __and__(self, other: "UnionFilters") -> "AndFilter":
        from scm.moliterate.filters.bitwise_filters import AndFilter

        return AndFilter(filters={self, other})  # pyright: ignore[reportArgumentType, reportUnhashable]

    def __or__(self, other: "UnionFilters") -> "OrFilter":
        from scm.moliterate.filters.bitwise_filters import OrFilter

        return OrFilter(filters={self, other})  # pyright: ignore[reportArgumentType, reportUnhashable]

    def __invert__(self) -> "NotFilter":
        from scm.moliterate.filters.bitwise_filters import NotFilter

        return NotFilter(filter=self)  # pyright: ignore[reportArgumentType]

    def __rshift__(self, other: "UnionFilters") -> "PipeFilter":
        from scm.moliterate.filters.bitwise_filters import PipeFilter

        return PipeFilter(filters=[self, other])  # pyright: ignore[reportArgumentType]

    def __hash__(self) -> int:  # Needed only for type hint
        return super().__hash__()
