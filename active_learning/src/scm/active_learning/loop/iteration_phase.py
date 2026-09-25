from __future__ import annotations

from scm.active_learning.py_utils import strEnum


class IterPhase(strEnum):
    TASK = "TASK"
    LABEL = "LABEL"
    ACCURACY = "ACCURACY"
    CONVERGENCE = "CONVERGENCE"
    SPLIT_AND_ADD = "SPLIT_AND_ADD"
    TRAIN = "TRAIN"
    TRAIN_CHECK = "TRAIN_CHECK"
