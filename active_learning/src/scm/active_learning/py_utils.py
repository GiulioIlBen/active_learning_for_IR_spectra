from contextlib import contextmanager
from datetime import datetime
from enum import Enum


class strEnum(str, Enum):
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        return name

    def __str__(self) -> str:
        return self.value


@contextmanager
def context_timer():
    time_infos = {}
    time_infos["start"] = datetime.now()
    try:
        yield time_infos
    finally:
        time_infos["dtime"] = datetime.now() - time_infos["start"]
