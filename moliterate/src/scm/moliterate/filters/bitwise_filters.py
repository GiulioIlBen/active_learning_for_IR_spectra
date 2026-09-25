from __future__ import annotations

import warnings
from functools import reduce
from typing import TYPE_CHECKING, Any, Iterable, List, Literal, Optional, Set, Union

import numpy as np
from pydantic import field_serializer, field_validator

from scm.moliterate.core import BaseChemDataSet, BaseFilter
from scm.moliterate.utils.hashable_model import hash_set

if TYPE_CHECKING:
    from scm.moliterate.filters import UnionFilters


# this warning is due to serialization of the filters with Set type
def __filter_known_warnings():
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"scm\.moliterate\.filters\.bitwise_filters",
        message=(
            r"Pydantic serializer warnings.*\n.*Expected `set\[.*\]` but got `list` with value `\[.*\]`"
            r" - serialized value may not be as expected"
        ),
    )


__filter_known_warnings()


def model_post_init_flatten(filter_with_filters: Union[AndFilter, OrFilter]) -> None:
    """for AndFilter and OrFilter it is better to flatten the simple cases"""
    if not any(isinstance(f, filter_with_filters.__class__) for f in filter_with_filters.filters):
        return

    flattened: Set["UnionFilters"] = set()
    stack = list(filter_with_filters.filters)
    for f in stack:
        if isinstance(f, filter_with_filters.__class__):
            flattened = flattened.union(f.filters)
        else:
            flattened.add(f)
    object.__setattr__(filter_with_filters, "filters", flattened)


class AndFilter(BaseFilter):
    type: Literal["AndFilter"] = "AndFilter"  # Field discriminator!
    filters: Set["UnionFilters"]

    def apply_filter(self, db: BaseChemDataSet) -> Optional[Iterable[int]]:
        """return RELATIVE indices of the dataset in input"""
        to_union = [f.apply_filter(db) for f in self.filters]
        not_none = [x for x in to_union if x is not None]
        if len(not_none) == 0:
            return None
        result = reduce(np.intersect1d, map(np.asarray, not_none))
        return result

    def model_post_init(self, __context: Any) -> None:
        model_post_init_flatten(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filters={self.filters})"

    def hash_payload(self) -> Any:
        ret = self.model_dump(
            mode="python",
            exclude={
                "filters",
            },
        )
        ret["filters"] = hash_set(self.filters)
        return ret

    def __hash__(self) -> int:  # Needed only for type hint
        return super().__hash__()

    @field_serializer("filters", when_used="always", mode="wrap")
    def _filters_python_serializer(self, value: Set["UnionFilters"], handler):
        return handler(sorted(value, key=hash))

    # trying to make it more flexible with option to add on runtime BaseFilter implementations, but difficult
    # since you want to keep json serialization and parsing, therefore you must have pre defined UnionFilters
    # @field_validator("filters", mode="before")
    # @classmethod
    # def _coerce_filters(cls, value):
    #     return coerce_field_with_base_filter(value)


class OrFilter(BaseFilter):
    type: Literal["OrFilter"] = "OrFilter"  # Field discriminator!
    filters: Set["UnionFilters"]

    def apply_filter(self, db: BaseChemDataSet) -> Optional[Iterable[int]]:
        """return RELATIVE indices of the dataset in input"""
        to_union = [f.apply_filter(db) for f in self.filters]
        not_none = [x for x in to_union if x is not None]
        if len(not_none) == 0:
            return None
        return np.unique(np.concatenate([np.asarray(x) for x in not_none]))

    def model_post_init(self, __context: Any) -> None:
        model_post_init_flatten(self)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filters={self.filters})"

    def hash_payload(self) -> Any:
        ret = self.model_dump(
            mode="python",
            exclude={
                "filters",
            },
        )
        ret["filters"] = hash_set(self.filters)
        return ret

    def __hash__(self) -> int:  # Needed only for typehint
        return super().__hash__()

    @field_serializer("filters", when_used="always", mode="wrap")
    def _filters_python_serializer(self, value: Set["UnionFilters"], handler):
        return handler(sorted(value, key=hash))


class NotFilter(BaseFilter):
    type: Literal["NotFilter"] = "NotFilter"  # Field discriminator!
    filter: "UnionFilters"

    def apply_filter(self, db: BaseChemDataSet) -> Optional[Iterable[int]]:
        """return RELATIVE indices of the dataset in input"""
        to_ret_opposite = self.filter.apply_filter(db)
        if to_ret_opposite is None:
            return []
        total = np.arange(len(db))
        return np.setdiff1d(total, np.asarray(to_ret_opposite))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filter={self.filter})"

    def __hash__(self) -> int:  # Needed only for typehint
        return super().__hash__()


class PipeFilter(BaseFilter):
    type: Literal["PipeFilter"] = "PipeFilter"  # Field discriminator!
    filters: List["UnionFilters"]

    def apply_filter(self, db: BaseChemDataSet) -> Optional[Iterable[int]]:
        """return RELATIVE indices of the dataset in input"""
        filtered_f = db
        for f in self.filters:
            filtered_f = f(filtered_f)
        return db.from_absolute_to_relative(filtered_f.select_indices())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(filters={self.filters})"

    def __hash__(self) -> int:  # Needed only for typehint
        return super().__hash__()
