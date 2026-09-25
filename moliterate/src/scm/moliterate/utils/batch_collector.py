from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterable, List, TypeVar

T = TypeVar("T")


@contextmanager
def batch_collector(
    flush: Callable[[Iterable[T]], None],
    batch_size: int = 1000,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    batch: List[T] = []

    def add(item: T) -> None:
        batch.append(item)
        if len(batch) >= batch_size:
            flush(batch)
            batch.clear()

    try:
        yield add
    finally:
        if batch:
            flush(batch)
            batch.clear()
