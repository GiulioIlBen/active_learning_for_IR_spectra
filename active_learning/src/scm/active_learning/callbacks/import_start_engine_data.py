from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from scm.active_learning.callbacks.core import ALCallback
from scm.active_learning.loop.run_control import ActiveLearningRunControl

if TYPE_CHECKING:
    from scm.active_learning.loop.loop import ActiveLearningLoop


class ImportStartEngineData(ALCallback):
    type: Literal["ImportStartEngineData"] = "ImportStartEngineData"

    def before_loop(self, al_loop: "ActiveLearningLoop", state=None) -> None:
        run_control = getattr(al_loop, "run_control", ActiveLearningRunControl())
        if not run_control.should_import_start_engine_data():
            return
        if al_loop.start_engine is None:
            return
        al_loop.splitting.import_data_from_engine(al_loop.start_engine, imported=True)
