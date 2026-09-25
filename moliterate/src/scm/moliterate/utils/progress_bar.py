from __future__ import annotations

from collections.abc import Sized
from typing import Iterable, Optional, TypeVar

from pydantic import BaseModel
from tqdm import tqdm as _tqdm

T = TypeVar("T")


class TqdmConfig(BaseModel):
    # it would be better to be enable instead of disable, but I prefer to map 1to1 with tqdm settings
    disable: bool = False


MOLITERATE_PROGRESS_BAR_CONFIG = TqdmConfig()


def moliterate_tqdm(iterable: Optional[Iterable[T]] = None, **kwargs):
    global MOLITERATE_PROGRESS_BAR_CONFIG
    settings_tqdm = {}
    try:
        settings_tqdm["total"] = len(iterable)
    except Exception:
        pass

    settings_tqdm.update(MOLITERATE_PROGRESS_BAR_CONFIG.model_dump())
    if kwargs:
        settings_tqdm.update(kwargs)

    return _tqdm(iterable, **settings_tqdm)


__all__ = [
    "TqdmConfig",
    "MOLITERATE_PROGRESS_BAR_CONFIG",
    "moliterate_tqdm",
]
