from __future__ import annotations

from datetime import datetime
from enum import auto

from scm.active_learning.py_utils import context_timer, strEnum


class Color(strEnum):
    RED = auto()
    BLUE = auto()


def test_str_enum_auto_values_and_str():
    assert Color.RED.value == "RED"
    assert str(Color.BLUE) == "BLUE"


def test_context_timer_records_elapsed():
    with context_timer() as info:
        assert isinstance(info["start"], datetime)

    assert "dtime" in info
    assert info["dtime"].total_seconds() >= 0
