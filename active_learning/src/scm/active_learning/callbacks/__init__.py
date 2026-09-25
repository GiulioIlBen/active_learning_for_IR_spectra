from __future__ import annotations

from typing import Union

from pydantic import Field
from typing_extensions import Annotated

from scm.active_learning.callbacks.al_minimal_logger import ALMinimalLogger
from scm.active_learning.callbacks.al_state_logger import ALStateLogger
from scm.active_learning.callbacks.core import ALCallback
from scm.active_learning.callbacks.early_stop_task import EarlyStopAccuracy, EarlyStopValidity
from scm.active_learning.callbacks.folder_manager import FolderManagerCallback
from scm.active_learning.callbacks.import_start_engine_data import ImportStartEngineData
from scm.active_learning.callbacks.manual_stopper import ManualStopper
from scm.active_learning.callbacks.no_post_filter_data import NoPostFilterData
from scm.active_learning.callbacks.skip_if import SkipIf
from scm.active_learning.callbacks.stop_below_active_systems import StopBelowActiveSystems

UnionALCallbacks = Annotated[
    Union[
        ALMinimalLogger,
        ALStateLogger,
        EarlyStopAccuracy,
        EarlyStopValidity,
        ManualStopper,
        NoPostFilterData,
        FolderManagerCallback,
        ImportStartEngineData,
        SkipIf,
        StopBelowActiveSystems,
    ],
    Field(discriminator="type"),
]

__all__ = [
    "UnionALCallbacks",
    "ALCallback",
    "ALStateLogger",
    "ALMinimalLogger",
    "EarlyStopAccuracy",
    "EarlyStopValidity",
    "ManualStopper",
    "NoPostFilterData",
    "FolderManagerCallback",
    "ImportStartEngineData",
    "SkipIf",
    "StopBelowActiveSystems",
]
