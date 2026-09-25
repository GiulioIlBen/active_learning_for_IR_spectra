import warnings
from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from scm.moliterate.filters.apricot_interface import ApricotFilter
from scm.moliterate.filters.bitwise_filters import AndFilter, NotFilter, OrFilter, PipeFilter
from scm.moliterate.filters.conditions_filter import ConditionsFilter
from scm.moliterate.filters.conformers_ams import AMSConformersFilter
from scm.moliterate.filters.context_aware_filters import SimpleContextFilter
from scm.moliterate.filters.duplicates_filters import RemoveDuplicates
from scm.moliterate.filters.scalar_filters import FarthestPointFilter, LinearSteppedFilter, RandomFilter, TopNFilter

UnionFilters = Annotated[
    Union[
        TopNFilter,
        FarthestPointFilter,
        RandomFilter,
        LinearSteppedFilter,
        ConditionsFilter,
        AMSConformersFilter,
        SimpleContextFilter,
        RemoveDuplicates,
        ApricotFilter,
        AndFilter,
        NotFilter,
        OrFilter,
        PipeFilter,
    ],
    Field(discriminator="type"),
]

PipeFilter.model_rebuild()
AndFilter.model_rebuild()
OrFilter.model_rebuild()
NotFilter.model_rebuild()
RemoveDuplicates.model_rebuild()
SimpleContextFilter.model_rebuild()

__all__ = [
    "UnionFilters",
    # scalar
    "TopNFilter",
    "FarthestPointFilter",
    "RandomFilter",
    "LinearSteppedFilter",
    # bitwise
    "AndFilter",
    "NotFilter",
    "OrFilter",
    "PipeFilter",
    # multi condition
    "ConditionsFilter",
    "SimpleContextFilter",
    "RemoveDuplicates",
    "ApricotFilter",
    "AMSConformersFilter",
]
