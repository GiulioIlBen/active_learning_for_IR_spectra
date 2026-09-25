from __future__ import annotations

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from .concat_check_getter import ConcatCheckerGetter
from .core import CheckerGetter
from .decoupled_check_getter import DecoupledCheckerGetter
from .union_check_getter import UnionCheckerGetter, UnionMember

ConcreteCheckerGetter = Annotated[
    Union[ConcatCheckerGetter, DecoupledCheckerGetter, UnionCheckerGetter], Field(discriminator="type")
]


__all__ = ["ConcreteCheckerGetter", "CheckerGetter", "DecoupledCheckerGetter", "UnionCheckerGetter", "UnionMember"]
