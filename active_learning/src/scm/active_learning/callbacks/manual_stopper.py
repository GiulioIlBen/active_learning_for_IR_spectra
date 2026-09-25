from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from scm.active_learning.callbacks.core import ALCallback
from scm.active_learning.callbacks.folder_manager import FolderManagerCallback
from scm.active_learning.logging import log_level
from scm.active_learning.loop.iteration_phase import IterPhase

if TYPE_CHECKING:
    from scm.active_learning.loop.iteration_state import IterationState
    from scm.active_learning.loop.loop import ActiveLearningLoop


class ManualStopper(ALCallback):
    type: Literal["ManualStopper"] = "ManualStopper"
    file_name: str = "STOP_ACTIVE_LEARNING"
    folder_path: Optional[Path] = None
    message_prefix: str = "ManualStopper: stop requested"
    skip_message: str = "ManualStopper"

    def before_loop(self, al_loop: "ActiveLearningLoop", state=None) -> None:
        stop_file = self.stop_file(al_loop)
        log_level(
            'ManualStopper: to stop the loop manually, run:\n echo "message" > {stop_file}',
            stop_file=stop_file,
        )

    def start_iter_loop(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        self.check(al_loop=al_loop, state=state)

    def after_convergence(self, al_loop, state):
        self.check(al_loop=al_loop, state=state)

    def check(self, al_loop: "ActiveLearningLoop", state: "IterationState") -> None:
        stop_message = self._read_stop_message(al_loop)
        if stop_message is None:
            return

        state.al_finished_reason = f"{self.message_prefix}. {stop_message}"
        state.skip.update({phase: self.skip_message for phase in IterPhase})
        log_level(
            "{reason}",
            level="SUCCESS",
            reason=state.al_finished_reason,
        )

    def stop_file(self, al_loop: "ActiveLearningLoop") -> Path:
        folder_path = self.folder_path
        if folder_path is None:
            folder_callbacks = al_loop.query_callbacks(FolderManagerCallback)
            folder_path = folder_callbacks[0].run_dir() if folder_callbacks else Path.cwd()
        return folder_path.expanduser().resolve(strict=False) / self.file_name

    def _read_stop_message(self, al_loop: "ActiveLearningLoop") -> str | None:
        stop_file = self.stop_file(al_loop)
        if not stop_file.is_file():
            return None

        message = stop_file.read_text(encoding="utf-8").strip()
        if not message:
            return None
        return message
