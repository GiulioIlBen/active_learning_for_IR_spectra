from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, PrivateAttr


class StoppableCounter(BaseModel):
    start: int = 0
    stop: int = 100
    iteration_al: int = 0
    reason: Union[str, Literal["CONVERGED"]] = ""
    _stop: bool = PrivateAttr(default=False)  # Stop condition flag

    def stop_next_iter(self, reason: str | Literal["CONVERGED"]):
        """Externally stop the iterator."""
        self._stop = True
        self.set_reason(reason)

    def set_reason(self, reason: str | Literal["CONVERGED"]):
        self.reason = f"AL stopped: {reason}"

    def run(self):
        for iteration_al in range(self.start, self.stop):  # Infinite counter
            if self._stop:
                return  # Stops when flag is set
            self.iteration_al = iteration_al
            yield iteration_al
        if self.reason == "":
            self.set_reason("max AL iterations reached")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(start={self.start}, stop={self.stop})"
